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

// --- 6. PREDICCIÓN GM(1,1) CON LAS ÚLTIMAS 4 MUESTRAS ---
#define N_HIST 4                  // muestras que se conservan (una por minuto)

struct ResultadoGM {
  double a;
  double b;
  double ARPE;                    // error porcentual promedio del ajuste
  float  prediccion;              // pronóstico para k = n (el siguiente valor)
  bool   ok;
};

float histT[N_HIST];              // historial ordenado: [0] = más antigua, [N-1] = más reciente
float histH[N_HIST];
int   histCount = 0;

float predT = NAN;                // predicción vigente para el minuto actual
float predH = NAN;
bool  prediccionValida = false;

/*
 * calcularGM11
 * ------------
 * Misma matemática que el ejemplo con Eigen, pero resolviendo directamente las
 * ecuaciones normales 2x2 (sin Eigen, sin memoria dinámica).
 *   X0_in : datos originales (n <= N_HIST)
 * Devuelve a, b, ARPE del ajuste y el pronóstico para k = n.
 */
ResultadoGM calcularGM11(const float* X0_in, int n) {
  ResultadoGM res = {0.0, 0.0, 0.0, NAN, false};
  if (n < 2 || n > N_HIST) return res;

  // GM(1,1) necesita datos > 0. Si hay valores <= 0 se desplaza la serie
  // y el desplazamiento se revierte al final.
  float minimo = X0_in[0];
  for (int i = 1; i < n; i++) if (X0_in[i] < minimo) minimo = X0_in[i];
  double desp = (minimo <= 0) ? (1.0 - minimo) : 0.0;

  double X0[N_HIST], X1[N_HIST];
  for (int i = 0; i < n; i++) X0[i] = (double)X0_in[i] + desp;

  // 1. Acumulación (cumsum)
  X1[0] = X0[0];
  for (int i = 1; i < n; i++) X1[i] = X1[i - 1] + X0[i];

  // 2. Mínimos cuadrados: Y = -a*Z1 + b, con Z1 = media de X1 consecutivos
  //    parametros = inv(B'B) * B'Y, con B = [-Z1, 1] e Y = X0[1..]
  const int m = n - 1;
  double sz = 0, szz = 0, sy = 0, szy = 0;
  for (int i = 0; i < m; i++) {
    double z = 0.5 * (X1[i] + X1[i + 1]);
    double y = X0[i + 1];
    sz  += z;
    szz += z * z;
    sy  += y;
    szy += z * y;
  }
  double det = szz * m - sz * sz;
  if (fabs(det) < 1e-9) return res;

  double a = (sz * sy - m * szy) / det;
  double b = (szz * sy - sz * szy) / det;

  // 3. Serie simulada. Forma equivalente y estable de
  //    X0_hat[k] = (x0_1 - b/a) * exp(-a*k) * (1 - exp(a))
  //    (evita dividir entre cero cuando a ~ 0, p. ej. datos casi constantes)
  double g    = (fabs(a) < 1e-9) ? 1.0 : expm1(a) / a;
  double base = (b - a * X0[0]) * g;

  double sumaRPE = 0.0;
  for (int k = 1; k < n; k++) {
    double xhat = base * exp(-a * k);
    sumaRPE += fabs(X0[k] - xhat) / X0[k] * 100.0;
  }

  // 4. Pronóstico del siguiente valor (k = n)
  double pred = base * exp(-a * n) - desp;

  res.a = a;
  res.b = b;
  res.ARPE = sumaRPE / (n - 1);
  if (isfinite(pred)) {
    res.prediccion = (float)pred;
    res.ok = true;
  }
  return res;
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

    // --- PREDICCIÓN GM(1,1) ---
    display.setCursor(0, 41);
    if (prediccionValida) {
      display.printf("Pred T:%.1fC H:%.0f%%", predT, predH);
    } else {
      // Faltan minutos para tener 4 muestras previas + la muestra actual
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

      // 0. Predecir el valor actual con las 4 muestras anteriores
      //    y después agregar la lectura actual al historial.
      int salto = (ultimoMinuto < 0) ? 1 : (now.Minute() - ultimoMinuto + 60) % 60;
      if (salto != 1) histCount = 0;   // hubo huecos: la serie ya no sería equiespaciada

      prediccionValida = false;
      if (histCount == N_HIST) {
        ResultadoGM gmT = calcularGM11(histT, N_HIST);
        ResultadoGM gmH = calcularGM11(histH, N_HIST);
        if (gmT.ok && gmH.ok) {
          predT = gmT.prediccion;
          predH = constrain(gmH.prediccion, 0.0f, 100.0f);
          prediccionValida = true;
          Serial.printf(">>> GM(1,1) | Lect: T=%.2f H=%.1f | Pred: T=%.2f H=%.1f | Err: T=%+.2f H=%+.1f | ARPE ajuste: T=%.3f%% H=%.3f%%\n",
                        temperatura, humedad, predT, predH,
                        temperatura - predT, humedad - predH,
                        gmT.ARPE, gmH.ARPE);
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
