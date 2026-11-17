#include <WiFi.h>
#include <esp_now.h>
#include "time.h"
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ThreeWire.h>  
#include <RtcDS1302.h>
#include <Adafruit_SHT31.h> 
#include <SPI.h>
#include <SD.h>
#include <math.h>

// --- 1. DATOS WI-FI ---
const char* ssid = "TemakaMB";
const char* password = "pass1234";

const char* ntpServer = "pool.ntp.org";
const long  gmtOffset_sec = -21600; // -6 horas
const int   daylightOffset_sec = 0;

const bool ENABLE_SCREEN = true;

// --- 2. CONFIGURACIÓN PANTALLA OLED ---
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define I2C_SDA 21 
#define I2C_SCL 22 
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// --- 3. CONFIGURACIÓN RELOJ DS1302 ---
ThreeWire myWire(25, 26, 27); // DAT(25), CLK(26), RST(27)
RtcDS1302<ThreeWire> Rtc(myWire);

// --- 4. CONFIGURACIÓN SENSOR SHT3X Y SD ---
Adafruit_SHT31 sht31 = Adafruit_SHT31();
#define SD_CS 5 // Pin CS de la MicroSD
int ultimoMinuto = -1; 
bool sdOK = false;

// --- 5. CONFIGURACIÓN ESP-NOW ---
typedef struct {
  uint32_t contador;      // se incrementa en cada envío
  uint32_t uptime_ms;     // millis() del emisor
  float    temperatura;   // sensor
  float    humedad;       // humedad
  uint32_t heap_libre;    // ESP.getFreeHeap() del emisor
  uint8_t  estado;        // 0=ok, 1=advertencia, 2=error
} __attribute__((packed)) Telemetria;

Telemetria datosTelemetria;
uint8_t macReceptor[] = {0x70, 0x4B, 0xCA, 0x23, 0xB7, 0xBC};
esp_now_peer_info_t peerInfo;

// --- 6. PREDICCIÓN GM(1,1) + FOURIER CON LAS ÚLTIMAS N_HIST MUESTRAS ---
/*
 * Tamaño de la ventana (una muestra por minuto).
 * Igual que en el código de Python:  T = n - 1   y   z = int((n - 1)/2 - 1)
 * El artículo pide que z sea entero, es decir, n impar:
 *   N_HIST = 4  -> z = 0  (Fourier solo aporta un término constante a0/2)
 *   N_HIST = 5  -> z = 1  (a0, a1, b1)
 *   N_HIST = 7  -> z = 2  (a0, a1, b1, a2, b2)
 */
#define N_HIST 5

// 1 = además del resumen imprime RPE punto a punto y los coeficientes de Fourier
#define METRICAS_DETALLADAS 1

struct ResultadoGMF {
  double a;
  double b;
  int    T;                       // periodo de la serie de Fourier (n - 1)
  int    z;                       // número de armónicos
  int    nCoef;                   // 2*z + 1
  double coef[N_HIST];            // [a0, a1, b1, a2, b2, ...]
  double rpeGM[N_HIST];           // RPE (%) del GM(1,1), k = 2..n
  double rpeF[N_HIST];            // RPE (%) del GM(1,1)+Fourier, k = 2..n
  double ARPE_GM;                 // error porcentual relativo promedio, GM(1,1)
  double ARPE_F;                  // error porcentual relativo promedio, GM(1,1)+Fourier
  float  predGM;                  // pronóstico GM(1,1) puro para k = n+1
  float  predF;                   // pronóstico GM(1,1)+Fourier para k = n+1
  bool   ok;
};

float histT[N_HIST];              // historial ordenado: [0] = más antigua, [N-1] = más reciente
float histH[N_HIST];
int   histCount = 0;

float predT = NAN;                // predicción vigente para el minuto actual
float predH = NAN;
bool  prediccionValida = false;

// Matriz aumentada del sistema normal de Fourier (global para no pasar arreglos 2D)
static double gA[N_HIST][N_HIST + 1];

// Base de Fourier: j=0 -> 1/2 ; j=1 -> cos(2*pi*1*k/T) ; j=2 -> sin(2*pi*1*k/T) ;
//                  j=3 -> cos(2*pi*2*k/T) ; j=4 -> sin(2*pi*2*k/T) ; ...
// Mismo orden de columnas que la matriz P del código de Python.
static double baseFourier(int j, int k, int T) {
  if (j == 0) return 0.5;
  int i = (j + 1) / 2;
  double ang = 2.0 * M_PI * (double)i * (double)k / (double)T;
  return (j & 1) ? cos(ang) : sin(ang);
}

// Resuelve gA * x = rhs (matriz aumentada p x (p+1)) por eliminación gaussiana
// con pivoteo parcial. Devuelve false si el sistema es singular.
static bool resolverSistema(int p, double* x) {
  for (int c = 0; c < p; c++) {
    int piv = c;
    for (int r = c + 1; r < p; r++) {
      if (fabs(gA[r][c]) > fabs(gA[piv][c])) piv = r;
    }
    if (fabs(gA[piv][c]) < 1e-12) return false;
    if (piv != c) {
      for (int j = c; j <= p; j++) {
        double t = gA[c][j]; gA[c][j] = gA[piv][j]; gA[piv][j] = t;
      }
    }
    for (int r = c + 1; r < p; r++) {
      double f = gA[r][c] / gA[c][c];
      for (int j = c; j <= p; j++) gA[r][j] -= f * gA[c][j];
    }
  }
  for (int r = p - 1; r >= 0; r--) {
    double s = gA[r][p];
    for (int j = r + 1; j < p; j++) s -= gA[r][j] * x[j];
    x[r] = s / gA[r][r];
  }
  return true;
}

/*
 * calcularGM11Fourier
 * -------------------
 * GM(1,1) modificado con series de Fourier sobre los residuos
 * (ecuaciones 22-30 del artículo, igual que el código de Python).
 *   X0_in : datos originales (n <= N_HIST)
 * Devuelve a, b, coeficientes de Fourier, RPE/ARPE del ajuste (GM y GM+Fourier)
 * y el pronóstico para k = n+1 (GM puro y GM+Fourier).
 *
 * Índices: k es 1-based como en el artículo; el arreglo X0[k-1] guarda x(k).
 */
ResultadoGMF calcularGM11Fourier(const float* X0_in, int n) {
  ResultadoGMF res = {};
  res.predGM = NAN;
  res.predF  = NAN;
  if (n < 2 || n > N_HIST) return res;

  // GM(1,1) necesita datos > 0. Si hay valores <= 0 se desplaza la serie
  // y el desplazamiento se revierte en el pronóstico.
  float minimo = X0_in[0];
  for (int i = 1; i < n; i++) if (X0_in[i] < minimo) minimo = X0_in[i];
  double desp = (minimo <= 0) ? (1.0 - minimo) : 0.0;

  double X0[N_HIST], X1[N_HIST];
  for (int i = 0; i < n; i++) X0[i] = (double)X0_in[i] + desp;

  // 1. Acumulación (AGO)
  X1[0] = X0[0];
  for (int i = 1; i < n; i++) X1[i] = X1[i - 1] + X0[i];

  // 2. Mínimos cuadrados: Y = -a*Z1 + b, con Z1 = media de X1 consecutivos
  const int m = n - 1;
  double sz = 0, szz = 0, sy = 0, szy = 0;
  for (int i = 0; i < m; i++) {
    double z1 = 0.5 * (X1[i] + X1[i + 1]);
    double y  = X0[i + 1];
    sz  += z1;
    szz += z1 * z1;
    sy  += y;
    szy += z1 * y;
  }
  double det = szz * m - sz * sz;
  if (fabs(det) < 1e-9) return res;

  double a = (sz * sy - m * szy) / det;
  double b = (szz * sy - sz * szy) / det;
  res.a = a;
  res.b = b;

  // 3. Serie simulada GM(1,1) y residuos (k = 2..n). Forma estable de
  //    x_p(k) = (x(1) - b/a) * exp(-a*(k-1)) * (1 - exp(a))
  double g    = (fabs(a) < 1e-9) ? 1.0 : expm1(a) / a;
  double base = (b - a * X0[0]) * g;

  double eps[N_HIST];               // eps[k-2] = x(k) - x_p(k)
  double sumaGM = 0.0;
  for (int k = 2; k <= n; k++) {
    double xhat = base * exp(-a * (k - 1));
    eps[k - 2] = X0[k - 1] - xhat;
    res.rpeGM[k - 2] = fabs(eps[k - 2]) / fabs(X0[k - 1]) * 100.0;
    sumaGM += res.rpeGM[k - 2];
  }
  res.ARPE_GM = sumaGM / m;

  // 4. Serie de Fourier sobre los residuos: eps ≈ P*C, C = (P'P)^-1 P' eps
  const int T = n - 1;
  int z = (int)((n - 1) / 2.0 - 1.0);   // trunca hacia 0, igual que int() en Python
  if (z < 0) z = 0;
  const int p = 2 * z + 1;
  res.T = T;
  res.z = z;
  res.nCoef = p;

  for (int r = 0; r < p; r++) {
    for (int c = 0; c < p; c++) {
      double s = 0.0;
      for (int k = 2; k <= n; k++) s += baseFourier(r, k, T) * baseFourier(c, k, T);
      gA[r][c] = s;
    }
    double s = 0.0;
    for (int k = 2; k <= n; k++) s += baseFourier(r, k, T) * eps[k - 2];
    gA[r][p] = s;
  }
  if (!resolverSistema(p, res.coef)) return res;

  // 5. Serie corregida x_F(k) = x_p(k) + eps_hat(k) y su error
  double sumaF = 0.0;
  for (int k = 2; k <= n; k++) {
    double epsHat = 0.0;
    for (int j = 0; j < p; j++) epsHat += res.coef[j] * baseFourier(j, k, T);
    double xF = base * exp(-a * (k - 1)) + epsHat;
    res.rpeF[k - 2] = fabs(X0[k - 1] - xF) / fabs(X0[k - 1]) * 100.0;
    sumaF += res.rpeF[k - 2];
  }
  res.ARPE_F = sumaF / m;

  // 6. Pronóstico del siguiente valor (k = n+1): GM(1,1) + eps_hat(n+1)
  double epsNext = 0.0;
  for (int j = 0; j < p; j++) epsNext += res.coef[j] * baseFourier(j, n + 1, T);
  double pGM = base * exp(-a * n) - desp;
  double pF  = pGM + epsNext;

  if (isfinite(pGM) && isfinite(pF) && isfinite(res.ARPE_GM) && isfinite(res.ARPE_F)) {
    res.predGM = (float)pGM;
    res.predF  = (float)pF;
    res.ok = true;
  }
  return res;
}

// Imprime por Serial las métricas de error de una variable ya evaluada.
//   - ARPE (y RPE) de AJUSTE: serie original vs modelo, k = 2..n (como en Python)
//   - Error del PRONÓSTICO contra la lectura real que acaba de llegar
void imprimirMetricas(const char* var, const ResultadoGMF& r, float real) {
  Serial.printf(">>> [%s] GM(1,1)+Fourier | n=%d T=%d z=%d | a=%.8f b=%.6f\n",
                var, N_HIST, r.T, r.z, r.a, r.b);
  Serial.printf("    ARPE ajuste : GM=%.6f%%  GM+Fourier=%.6f%%  reduccion=%+.6f pp\n",
                r.ARPE_GM, r.ARPE_F, r.ARPE_GM - r.ARPE_F);

#if METRICAS_DETALLADAS
  Serial.print("    Coef Fourier [a0,a1,b1,...]: ");
  for (int j = 0; j < r.nCoef; j++) Serial.printf("%.6f%s", r.coef[j], (j < r.nCoef - 1) ? ", " : "\n");
  for (int k = 2; k <= N_HIST; k++) {
    Serial.printf("    k=%d  RPE GM=%.6f%%  RPE GM+Fourier=%.6f%%\n",
                  k, r.rpeGM[k - 2], r.rpeF[k - 2]);
  }
#endif

  float errGM = real - r.predGM;
  float errF  = real - r.predF;
  float den   = fabsf(real);
  float rpeGM = (den > 1e-9f) ? fabsf(errGM) / den * 100.0f : NAN;
  float rpeF  = (den > 1e-9f) ? fabsf(errF)  / den * 100.0f : NAN;
  Serial.printf("    Pronostico  : Real=%.3f | GM=%.3f (err %+.3f, RPE %.3f%%) | GM+Fourier=%.3f (err %+.3f, RPE %.3f%%)\n",
                real, r.predGM, errGM, rpeGM, r.predF, errF, rpeF);
}

// Agrega una muestra al historial; si está lleno, descarta la más antigua
void pushHist(float t, float h) {
  if (histCount == N_HIST) {
    for (int i = 0; i < N_HIST - 1; i++) {
      histT[i] = histT[i + 1];
      histH[i] = histH[i + 1];
    }
    histCount = N_HIST - 1;
  }
  histT[histCount] = t;
  histH[histCount] = h;
  histCount++;
}

void setup() {
  Serial.begin(115200);

  // Inicializar contador ESP-NOW
  datosTelemetria.contador = 0;

  // -- INICIAR PANTALLA E I2C --
  if(ENABLE_SCREEN){
    Wire.begin(I2C_SDA, I2C_SCL);
    if(display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
      display.clearDisplay();
      display.setTextColor(SSD1306_WHITE);
      display.setTextSize(1);
      display.setCursor(0, 0);
      display.println("Iniciando Sistema...");
      display.display();
    }
  } else {
    // Si la pantalla está deshabilitada de todas formas iniciamos I2C para el SHT3X
    Wire.begin(I2C_SDA, I2C_SCL);
  }

  // -- INICIAR MICRO SD --
  if (!SD.begin(SD_CS)) {
    Serial.println("Fallo al montar la SD o no hay tarjeta.");
    if(ENABLE_SCREEN){
      display.println("Error SD!");
      display.display();
    }
    delay(2000);
  } else {
    sdOK = true;
    if(ENABLE_SCREEN){
      display.println("SD OK!");
      display.display();
    }
    Serial.println("Tarjeta SD iniciada correctamente.");
    
    // Crear el archivo con los encabezados si no existe
    File file = SD.open("/datos_clima.csv", FILE_APPEND);
    if(file) {
      if(file.size() == 0) {
         file.println("Fecha,Hora,temp_original,humedad_original,temp_correjida,humedad_correjida,temp_pronosticada,humedad_pronosticada");
      }
      file.close();
    }
  }

  // -- INICIAR RELOJ --
  Rtc.Begin();
  Rtc.SetIsWriteProtected(false);
  if (!Rtc.GetIsRunning()) {
    Rtc.SetIsRunning(true);
  }
  
  // -- INICIAR SHT3X --
  if (!sht31.begin(0x44)) { // 0x44 es la dirección por defecto, usa 0x45 si el pin ADDR está en HIGH
    Serial.println(F("Error: No se encontro el SHT3X."));
  }

  // -- LÓGICA DE ARRANQUE INTELIGENTE (WIFI y ESP-NOW) --
  // ESP-NOW requiere que el WiFi esté en modo Station (WIFI_STA) permanentemente
  WiFi.mode(WIFI_STA);

  RtcDateTime now = Rtc.GetDateTime();
  
  // Formato: RtcDateTime(Año, Mes, Día, Hora, Minuto, Segundo)
  // Sincronizar con fecha de referencia
  RtcDateTime fechaReferencia(2026, 10, 01, 10, 0, 0);

  if (!now.IsValid() || now < fechaReferencia) {

    if(ENABLE_SCREEN){
      display.println("Buscando WiFi...");
      display.display();
    }
    
    WiFi.begin(ssid, password);
    int intentos = 0;
    while (WiFi.status() != WL_CONNECTED && intentos < 20) {
      delay(500);
      intentos++;
    }

    if (WiFi.status() == WL_CONNECTED) {
      configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
      struct tm timeinfo;
      if (getLocalTime(&timeinfo, 5000)) { 
        RtcDateTime internetTime = RtcDateTime(
          timeinfo.tm_year + 1900, timeinfo.tm_mon + 1, timeinfo.tm_mday, 
          timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec
        );
        Rtc.SetDateTime(internetTime);
      }
      // Desconectamos del router pero mantenemos el modo STA para ESP-NOW
      WiFi.disconnect(); 
    }
  }

  // -- INICIALIZAR ESP-NOW --
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error inicializando ESP-NOW");
    return;
  }
  
  memcpy(peerInfo.peer_addr, macReceptor, 6);
  peerInfo.channel = 0;  
  peerInfo.encrypt = false;
  
  if (esp_now_add_peer(&peerInfo) != ESP_OK){
    Serial.println("Failed to add peer");
    return;
  }
}

void loop() {
  RtcDateTime now = Rtc.GetDateTime();
  
  // LECTURA DIRECTA DEL SENSOR
  float ref_temperatura = sht31.readTemperature();
  float ref_humedad = sht31.readHumidity();

  // CORRECCION DE DATOS (sin cambios respecto al original)
  // OJO: la humedad se calcula con ref_temperatura, igual que en el original.
  float temperatura = ( 5.301868409 + (0.702422691 * ref_temperatura));
  float humedad = ( 32.60425363 + (0.6384588546 * ref_humedad));

  // ==========================================
  // 1. VISUALIZACIÓN EN TIEMPO REAL (Cada 500ms)
  // ==========================================
  if (ENABLE_SCREEN) {
    // --- MODO PANTALLA ---
    display.clearDisplay();

    // --- MOSTRAR FECHA ---
    display.setTextSize(1);
    display.setCursor(28, 0); 
    if (now.IsValid()) {
      display.printf("%02d/%02d/%04d", now.Day(), now.Month(), now.Year());
    } else {
      display.print("Error RTC");
    }

    // --- MOSTRAR HORA ---
    display.setTextSize(2);
    display.setCursor(15, 10); 
    if (now.IsValid()) {
      display.printf("%02d:%02d:%02d", now.Hour(), now.Minute(), now.Second());
    } else {
      display.print("--:--:--");
    }

    // --- LECTURA CORREGIDA (tiempo real) ---
    display.setTextSize(1);
    display.setCursor(0, 30); 
    if (isnan(temperatura) || isnan(humedad)) {
      display.print("Lect: Error SHT3X");
    } else {
      display.printf("Lect T:%.1fC H:%.0f%%", temperatura, humedad);
    }

    // --- PREDICCIÓN GM(1,1)+FOURIER ---
    display.setCursor(0, 41);
    if (prediccionValida) {
      display.printf("Pred T:%.1fC H:%.0f%%", predT, predH);
    } else {
      // Faltan minutos para tener N_HIST muestras previas + la muestra actual
      display.printf("Pred: en %d min", N_HIST + 1 - histCount);
    }

    // --- LÍNEA INFERIOR: ESTADO SD E HISTORIAL ---
    display.setCursor(0, 54);
    display.print(sdOK ? "SD: OK" : "SD: ERROR");
    display.printf("  Hist:%d/%d", histCount, N_HIST);

    display.display();

  } else {
    // --- MODO SERIAL ---
    if (now.IsValid()) {
      Serial.printf("[%02d/%02d/%04d %02d:%02d:%02d] ", 
                    now.Day(), now.Month(), now.Year(),
                    now.Hour(), now.Minute(), now.Second());
    } else {
      Serial.print("[Error RTC] ");
    }

    if (isnan(temperatura) || isnan(humedad)) {
      Serial.print("Error SHT3X | ");
    } else {
      Serial.printf("T: %.1f C  H: %.0f %% | ", temperatura, humedad);
    }

    // predicción
    if (prediccionValida) {
      Serial.printf("Pred T: %.1f C  H: %.0f %% | ", predT, predH);
    } else {
      Serial.printf("Pred: en %d min | ", N_HIST + 1 - histCount);
    }

    if (sdOK) {
      Serial.println("SD: OK");
    } else {
      Serial.println("SD: ERROR");
    }
  }

  // ==========================================
  // 2. ENVIAR ESP-NOW Y GUARDAR EN SD CADA MINUTO
  // ==========================================
  if (now.IsValid() && !isnan(temperatura) && !isnan(humedad)) {
    if (now.Minute() != ultimoMinuto) {

      // 0. Predecir el valor actual con las N_HIST muestras anteriores (GM(1,1)+Fourier),
      //    enviar las métricas de error por Serial y después agregar la lectura actual al historial.
      int salto = (ultimoMinuto < 0) ? 1 : (now.Minute() - ultimoMinuto + 60) % 60;
      if (salto != 1) histCount = 0;   // hubo huecos: la serie ya no sería equiespaciada

      prediccionValida = false;
      if (histCount == N_HIST) {
        ResultadoGMF gmT = calcularGM11Fourier(histT, N_HIST);
        ResultadoGMF gmH = calcularGM11Fourier(histH, N_HIST);
        if (gmT.ok && gmH.ok) {
          gmH.predGM = constrain(gmH.predGM, 0.0f, 100.0f);
          gmH.predF  = constrain(gmH.predF,  0.0f, 100.0f);
          predT = gmT.predF;
          predH = gmH.predF;
          prediccionValida = true;

          // Métricas de error del modelo evaluado (T y H)
          imprimirMetricas("T", gmT, temperatura);
          imprimirMetricas("H", gmH, humedad);
        }
      }
      pushHist(temperatura, humedad);
      
      // 1. Preparar y Enviar por ESP-NOW
      datosTelemetria.contador++;
      datosTelemetria.uptime_ms = millis();
      datosTelemetria.temperatura = temperatura;
      datosTelemetria.humedad = humedad;
      datosTelemetria.heap_libre = ESP.getFreeHeap();
      datosTelemetria.estado = (isnan(temperatura) || isnan(humedad)) ? 2 : 0; // 0=OK, 2=Error
      
      esp_err_t result = esp_now_send(macReceptor, (uint8_t *) &datosTelemetria, sizeof(datosTelemetria));
      
      if (result == ESP_OK) {
        Serial.println("-> ESP-NOW: Datos enviados con exito");
      } else {
        Serial.println("-> ESP-NOW: Error en el envío");
      }

      // 2. Enviar por TX Serial
      Serial.printf(">>> TX: %02d/%02d/%04d,%02d:%02d:%02d,%.1f,%.0f\n", 
                    now.Day(), now.Month(), now.Year(),
                    now.Hour(), now.Minute(), now.Second(),
                    temperatura, humedad);

      // 3. Guardar en la MicroSD
      //    Columnas: Fecha,Hora,temp_original,humedad_original,
      //              temp_correjida,humedad_correjida,
      //              temp_pronosticada,humedad_pronosticada
      //    Si todavía no hay predicción (primeros minutos), los dos últimos campos van vacíos.
      if (sdOK) {
        File dataFile = SD.open("/datos_clima.csv", FILE_APPEND);
        if (dataFile) {
          dataFile.printf("%02d/%02d/%04d,%02d:%02d:%02d,%.2f,%.2f,%.2f,%.2f,",
                          now.Day(), now.Month(), now.Year(),
                          now.Hour(), now.Minute(), now.Second(),
                          ref_temperatura, ref_humedad,
                          temperatura, humedad);
          if (prediccionValida) {
            dataFile.printf("%.2f,%.2f\n", predT, predH);
          } else {
            dataFile.print(",\n");
          }
          dataFile.close();
          Serial.println("-> Dato guardado en SD");
        } else {
          Serial.println("-> Error abriendo datos_clima.csv");
        }
      }
                    
      ultimoMinuto = now.Minute(); 
    }
  }

  delay(500); 
}
