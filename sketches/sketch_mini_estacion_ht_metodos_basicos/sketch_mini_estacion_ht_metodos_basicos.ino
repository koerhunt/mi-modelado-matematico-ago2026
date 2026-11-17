/*
 * MINI ESTACIÓN T/H - MÉTODOS BÁSICOS DE PRONÓSTICO
 * ==================================================
 * Implementación en ESP32 DevKit del cuaderno 06_serie_tiempo_02_datos_clima:
 *
 *   1. Average         y_hat(T+h) = promedio de los últimos N valores
 *   2. Naive           y_hat(T+h) = y(T)
 *   3. Seasonal Naive  y_hat(T+h) = y(T + h - m)          (m = 288, ciclo diario)
 *   4. Drift           y_hat(T+h) = y(T) + h * (y(T) - y(1)) / (T - 1)
 *
 * - Se toma una muestra cada INTERVALO_MIN = 5 minutos y se guardan en memoria
 *   los últimos N valores de temperatura y humedad en un buffer circular.
 * - En cada muestra, ANTES de agregar la lectura nueva, se compara contra la
 *   predicción que cada método hizo en la muestra anterior (pronóstico a 1 paso)
 *   y se acumulan las métricas del cuaderno: MAE, RMSE, MAPE y R².
 * - Al arrancar se recargan los últimos N registros de /datos_clima.csv (SD)
 *   para no esperar 24 h a que el Seasonal Naive tenga una temporada completa.
 *
 * PANTALLAS (pulsador en PIN_BOTON contra GND):
 *   Principal -> [Average: datos, gráfica T, gráfica H] -> [Naive ...]
 *             -> [S.Naive ...] -> [Drift ...] -> Resumen -> Principal
 *   - Pulsación corta : siguiente pantalla (después del Resumen regresa a la principal).
 *   - Pulsación larga : regresa directo a la pantalla principal.
 *   - Sin pulsar durante TIMEOUT_PANTALLA_MS: regresa a la principal.
 *
 * GRÁFICAS: línea continua = valores reales, puntos = predicciones a 1 paso
 * que hizo el método, línea punteada vertical = "ahora", línea discontinua a
 * la derecha = pronóstico de las próximas GRAF_FUT muestras.
 */

#include <WiFi.h>
#include <esp_now.h>
#include "time.h"
#include <Wire.h>
#include <Adafruit_GFX.h> // Adafruit GFX Library
#include <Adafruit_SSD1306.h> // Adafruit SSD13066
#include <ThreeWire.h> // RTC by Makuna
#include <RtcDS1302.h>
#include <Adafruit_SHT31.h>
#include <SPI.h>
#include <SD.h>
#include <math.h>
#include <string.h>
// SENSOR Library: Adafruit SHT31 Library

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
#define SD_CS 5 // Pin CS de la MicroSD (SPI: SCK 18, MISO 19, MOSI 23)
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

// --- 6. PULSADOR ---
// Pulsador entre PIN_BOTON y GND (se usa la resistencia pull-up interna).
// Alternativa sin cableado: usar el botón BOOT de la DevKit (GPIO 0).
#define PIN_BOTON              32
#define DEBOUNCE_MS            40
#define PULSACION_LARGA_MS     800
#define TIMEOUT_PANTALLA_MS    60000UL

// --- 7. PARÁMETROS DE LOS MÉTODOS ---
// INTERVALO_MIN : minutos entre muestras del modelo (el sensor se sigue leyendo
//                 cada 500 ms para la pantalla, pero al modelo, a la SD y a
//                 ESP-NOW solo entra una muestra cada INTERVALO_MIN minutos,
//                 alineada al reloj: hh:00, hh:05, hh:10, ...).
#define INTERVALO_MIN    5
#define MUESTRAS_DIA     (24 * 60 / INTERVALO_MIN)     // 288 muestras = 24 h
// N_VALORES     : valores que se mantienen en memoria por variable.
// M_ESTACIONAL  : periodo estacional en muestras (cada 5 min: m = 288 -> 24 h,
//                 igual que el cuaderno con datos de 5 min). Para probar en
//                 clase sin esperar un día se puede usar 12 (1 hora).
// MAX_HUECO_MUESTRAS: si faltan <= este número de muestras se rellenan
//                 repitiendo el último valor; si faltan más, se reinicia la
//                 serie (ya no sería equiespaciada y el Seasonal Naive quedaría
//                 desfasado).
#define N_VALORES          MUESTRAS_DIA
#define M_ESTACIONAL       MUESTRAS_DIA
#define MAX_HUECO_MUESTRAS 3                           // 15 min
#define PRECARGAR_SD       true
static_assert(N_VALORES >= M_ESTACIONAL, "N_VALORES debe ser >= M_ESTACIONAL");

// Gráficas: GRAF_HIST muestras de historia (8 h) + GRAF_FUT de pronóstico (~2.7 h) = 128 px
#define GRAF_HIST  96
#define GRAF_FUT   (SCREEN_WIDTH - GRAF_HIST)

#define NUM_METODOS 4
enum Metodo { AVERAGE = 0, NAIVE = 1, SNAIVE = 2, DRIFT = 3 };
const char* NOMBRE_METODO[NUM_METODOS] = {"AVERAGE", "NAIVE", "S.NAIVE", "DRIFT"};
const char* NOMBRE_CORTO[NUM_METODOS]  = {"Avg", "Naive", "SNaive", "Drift"};

// ============================================================
// BUFFER CIRCULAR: últimos N valores
// ============================================================
struct Serie {
  float    buf[N_VALORES];
  int      head  = 0;        // siguiente posición de escritura
  int      count = 0;        // valores almacenados
  double   suma  = 0.0;      // suma corriente (para Average en O(1))

  void limpiar() { head = 0; count = 0; suma = 0.0; }

  void push(float x) {
    if (count == N_VALORES) suma -= buf[head];   // sale el más antiguo
    else count++;
    buf[head] = x;
    suma += x;
    head = (head + 1) % N_VALORES;
  }

  // i = 0 -> más antiguo, i = count-1 -> más reciente
  float get(int i) const { return buf[(head - count + i + 2 * N_VALORES) % N_VALORES]; }
  float primero()  const { return get(0); }
  float ultimo()   const { return get(count - 1); }
};

// ============================================================
// MÉTRICAS ACUMULADAS (igual que metricas() del cuaderno)
// ============================================================
struct Metricas {
  uint32_t n = 0, nApe = 0;
  double sAbs = 0, sSq = 0, sApe = 0, sY = 0, sY2 = 0;

  void agregar(float real, float pred) {
    double e = (double)real - (double)pred;           // e_t = y_t - y_hat_t
    n++;
    sAbs += fabs(e);
    sSq  += e * e;
    sY   += real;
    sY2  += (double)real * real;
    if (real != 0.0f) { sApe += fabs(e / real); nApe++; }
  }
  float mae()  const { return n ? (float)(sAbs / n) : NAN; }
  float rmse() const { return n ? (float)sqrt(sSq / n) : NAN; }
  float mape() const { return nApe ? (float)(sApe / nApe * 100.0) : NAN; }
  float r2()   const {                                  // R² = 1 - SSE / SST
    if (n < 2) return NAN;
    double sst = sY2 - sY * sY / n;
    if (sst < 1e-9) return NAN;
    return (float)(1.0 - sSq / sst);
  }
};

// ============================================================
// VARIABLE PRONOSTICADA (temperatura o humedad)
// ============================================================
struct Variable {
  const char* nombre;
  const char* letra;     // "T" / "H"
  const char* unidad;
  float lo, hi;          // límites físicos del pronóstico

  Serie    serie;
  float    predSig[NUM_METODOS];      // pronóstico para la próxima muestra
  float    predEval[NUM_METODOS];     // pronóstico que se evaluó en la última muestra
  Metricas met[NUM_METODOS];

  // Historia corta para las gráficas
  float histReal[GRAF_HIST];
  float histPred[NUM_METODOS][GRAF_HIST];
  int   histHead = 0, histCount = 0;

  Variable(const char* n, const char* l, const char* u, float lo_, float hi_)
    : nombre(n), letra(l), unidad(u), lo(lo_), hi(hi_) {
    for (int k = 0; k < NUM_METODOS; k++) predSig[k] = predEval[k] = NAN;
  }

  // Pronóstico a h pasos (h >= 1) usando los datos actuales del buffer
  float pronostico(int metodo, int h) const {
    const int c = serie.count;
    float p = NAN;
    if (c == 0) return NAN;
    switch (metodo) {
      case AVERAGE:                            // media de los últimos N valores
        p = (float)(serie.suma / c);
        break;
      case NAIVE:                              // último valor observado
        p = serie.ultimo();
        break;
      case SNAIVE:                             // mismo instante de la temporada anterior
        if (c >= M_ESTACIONAL) p = serie.get(c - M_ESTACIONAL + ((h - 1) % M_ESTACIONAL));
        break;
      case DRIFT:                              // recta entre el primero y el último
        if (c >= 2) p = serie.ultimo() + h * (serie.ultimo() - serie.primero()) / (float)(c - 1);
        break;
    }
    if (!isnan(p)) { if (p < lo) p = lo; if (p > hi) p = hi; }
    return p;
  }

  float histRealAt(int i) const { return histReal[(histHead - histCount + i + 2 * GRAF_HIST) % GRAF_HIST]; }
  float histPredAt(int k, int i) const { return histPred[k][(histHead - histCount + i + 2 * GRAF_HIST) % GRAF_HIST]; }

  // Registra una lectura real. Si evaluar = true se acumulan los errores de
  // la predicción que cada método hizo para esta muestra.
  void actualizar(float x, bool evaluar) {
    for (int k = 0; k < NUM_METODOS; k++) {
      predEval[k] = evaluar ? predSig[k] : NAN;
      if (evaluar && !isnan(predSig[k])) met[k].agregar(x, predSig[k]);
      histPred[k][histHead] = predEval[k];
    }
    histReal[histHead] = x;
    histHead = (histHead + 1) % GRAF_HIST;
    if (histCount < GRAF_HIST) histCount++;

    serie.push(x);
    for (int k = 0; k < NUM_METODOS; k++) predSig[k] = pronostico(k, 1);
  }

  // Hubo un hueco largo: la serie deja de ser equiespaciada y se reinicia.
  // Las métricas acumuladas se conservan.
  void reiniciar() {
    serie.limpiar();
    histHead = histCount = 0;
    for (int k = 0; k < NUM_METODOS; k++) predSig[k] = predEval[k] = NAN;
  }
};

Variable varT("Temperatura", "T", "C", -INFINITY, INFINITY);
Variable varH("Humedad", "H", "%", 0.0f, 100.0f);

uint32_t ultimaMuestraAbs = 0;   // nº de intervalo (desde 2000-01-01) de la última muestra
bool     hayMuestra = false;

/*
 * registrarMuestra
 * ----------------
 * Agrega una muestra (una por intervalo de INTERVALO_MIN minutos) a ambas
 * variables manejando los huecos. Devuelve true si se procesó (intervalo nuevo).
 */
bool registrarMuestra(uint32_t muestraAbs, float t, float h) {
  if (hayMuestra) {
    int64_t salto = (int64_t)muestraAbs - (int64_t)ultimaMuestraAbs;
    if (salto == 0) return false;                       // mismo intervalo
    if (salto < 0 || salto > MAX_HUECO_MUESTRAS) {           // reloj hacia atrás o hueco largo
      Serial.printf("-> Hueco de %ld min: se reinicia la serie\n", (long)(salto * INTERVALO_MIN));
      varT.reiniciar();
      varH.reiniciar();
    } else {
      // Huecos cortos: se repite el último valor (sin evaluar errores)
      for (int64_t i = 1; i < salto; i++) {
        varT.actualizar(varT.serie.ultimo(), false);
        varH.actualizar(varH.serie.ultimo(), false);
      }
    }
  }
  varT.actualizar(t, true);
  varH.actualizar(h, true);
  ultimaMuestraAbs = muestraAbs;
  hayMuestra = true;
  return true;
}

/*
 * precargarDesdeSD
 * ----------------
 * Lee el final de /datos_clima.csv y vuelve a pasar los últimos ~N registros
 * por registrarMuestra(). Acepta los dos formatos usados por los sketches:
 *   Fecha,Hora,T,H                                     (sketch base)
 *   Fecha,Hora,T_orig,H_orig,T_corr,H_corr[,pred...]   (sketches GM / este)
 */
void precargarDesdeSD() {
  File f = SD.open("/datos_clima.csv", FILE_READ);
  if (!f) return;

  const size_t BYTES_POR_LINEA = 80;   // estimación holgada
  size_t tam = f.size();
  // Se leen hasta N*INTERVALO_MIN renglones por si el archivo tiene registros
  // de cada minuto (sketches anteriores): solo entra el primero de cada intervalo.
  size_t bytesNecesarios = (size_t)(N_VALORES * INTERVALO_MIN + 10) * BYTES_POR_LINEA;
  if (tam > bytesNecesarios) {
    f.seek(tam - bytesNecesarios);
    f.readStringUntil('\n');           // descarta la línea parcial
  }

  char linea[128];
  uint32_t leidas = 0;
  while (f.available()) {
    size_t len = f.readBytesUntil('\n', linea, sizeof(linea) - 1);
    linea[len] = '\0';
    int d, mo, y, hh, mi, ss;
    float a, b, c, e;
    int r = sscanf(linea, "%d/%d/%d,%d:%d:%d,%f,%f,%f,%f", &d, &mo, &y, &hh, &mi, &ss, &a, &b, &c, &e);
    float t, h;
    if (r >= 10)     { t = c; h = e; }   // columnas corregidas
    else if (r == 8) { t = a; h = b; }
    else continue;                       // encabezado o línea inválida
    if (isnan(t) || isnan(h)) continue;

    RtcDateTime fecha(y, mo, d, hh, mi, ss);
    if (registrarMuestra(fecha.TotalSeconds() / (60UL * INTERVALO_MIN), t, h)) leidas++;
  }
  f.close();
  Serial.printf("-> Precarga SD: %lu registros, en memoria %d/%d\n",
                (unsigned long)leidas, varT.serie.count, N_VALORES);
}

// ============================================================
// PANTALLAS
// ============================================================
// 0 = principal; 1..12 = (método, sub-pantalla); 13 = resumen
#define SUBPANTALLAS     3       // datos, gráfica T, gráfica H
#define PANTALLA_RESUMEN (1 + NUM_METODOS * SUBPANTALLAS)
#define NUM_PANTALLAS    (PANTALLA_RESUMEN + 1)

int pantalla = 0;
bool redibujar = true;
unsigned long ultimoBotonMs = 0;

void fmt(char* out, size_t len, float v, int dec) {
  if (isnan(v) || isinf(v)) snprintf(out, len, "--");
  else snprintf(out, len, "%.*f", dec, v);
}

void filaTabla(const char* etiqueta, float vt, float vh, int dec) {
  char a[12], b[12];
  fmt(a, sizeof(a), vt, dec);
  fmt(b, sizeof(b), vh, dec);
  display.printf("%-5s%7s%8s\n", etiqueta, a, b);
}

void dibujarPrincipal(const RtcDateTime& now, float temperatura, float humedad) {
  display.setTextSize(1);
  display.setCursor(28, 0);
  if (now.IsValid()) display.printf("%02d/%02d/%04d", now.Day(), now.Month(), now.Year());
  else display.print("Error RTC");

  display.setTextSize(2);
  display.setCursor(15, 10);
  if (now.IsValid()) display.printf("%02d:%02d:%02d", now.Hour(), now.Minute(), now.Second());
  else display.print("--:--:--");

  display.setTextSize(1);
  display.setCursor(0, 30);
  if (isnan(temperatura) || isnan(humedad)) display.print("Lect: Error SHT3X");
  else display.printf("Lect T:%.1fC H:%.0f%%", temperatura, humedad);

  display.setCursor(0, 41);
  display.printf("Hist:%d/%d", varT.serie.count, N_VALORES);
  if (varT.serie.count < M_ESTACIONAL)
    display.printf(" SN:%dm", (M_ESTACIONAL - varT.serie.count) * INTERVALO_MIN);  // minutos que faltan

  display.setCursor(0, 54);
  display.print(sdOK ? "SD:OK" : "SD:ERR");
  display.print("  [BTN] metodos");
}

void dibujarDatos(int k) {
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.printf("%-8s  datos   %d/%d\n", NOMBRE_METODO[k], k + 1, NUM_METODOS);
  display.println("        Temp     Hum");

  if (k == SNAIVE && varT.serie.count < M_ESTACIONAL) {
    display.printf("Pred faltan %d min\n", (M_ESTACIONAL - varT.serie.count) * INTERVALO_MIN);
  } else {
    filaTabla("Pred", varT.predSig[k], varH.predSig[k], 2);
  }
  filaTabla("MAE",   varT.met[k].mae(),  varH.met[k].mae(),  3);
  filaTabla("RMSE",  varT.met[k].rmse(), varH.met[k].rmse(), 3);
  filaTabla("MAPE%", varT.met[k].mape(), varH.met[k].mape(), 2);
  filaTabla("R2",    varT.met[k].r2(),   varH.met[k].r2(),   3);
  display.printf("n=%lu  hist %d", (unsigned long)varT.met[k].n, varT.serie.count);
}

void dibujarGrafica(int k, const Variable& v) {
  const int Y0 = 10, Y1 = SCREEN_HEIGHT - 1;
  const int nh = v.histCount;

  float fut[GRAF_FUT + 1];
  for (int h = 1; h <= GRAF_FUT; h++) fut[h] = v.pronostico(k, h);

  // Escala vertical automática
  float mn = INFINITY, mx = -INFINITY;
  auto incluir = [&](float y) { if (!isnan(y)) { if (y < mn) mn = y; if (y > mx) mx = y; } };
  for (int i = 0; i < nh; i++) { incluir(v.histRealAt(i)); incluir(v.histPredAt(k, i)); }
  for (int h = 1; h <= GRAF_FUT; h++) incluir(fut[h]);

  display.setTextSize(1);
  display.setCursor(0, 0);
  if (isinf(mn)) {
    display.printf("%s %s sin datos", NOMBRE_CORTO[k], v.letra);
    return;
  }
  display.printf("%s %s %.1f-%.1f%s", NOMBRE_CORTO[k], v.letra, mn, mx, v.unidad);
  if (mx - mn < 0.5f) { float c = 0.5f * (mx + mn); mn = c - 0.25f; mx = c + 0.25f; }

  auto mapY = [&](float y) -> int {
    return Y1 - (int)lroundf((y - mn) / (mx - mn) * (Y1 - Y0));
  };

  // Línea vertical punteada = "ahora"
  for (int y = Y0; y <= Y1; y += 3) display.drawPixel(GRAF_HIST - 1, y, SSD1306_WHITE);

  // Historia: real (línea) y predicción a 1 paso (puntos)
  int x0 = GRAF_HIST - nh;
  for (int i = 0; i < nh; i++) {
    float r = v.histRealAt(i);
    if (i > 0) {
      float rp = v.histRealAt(i - 1);
      display.drawLine(x0 + i - 1, mapY(rp), x0 + i, mapY(r), SSD1306_WHITE);
    }
    float p = v.histPredAt(k, i);
    if (!isnan(p) && (i % 2 == 0)) display.drawPixel(x0 + i, mapY(p), SSD1306_WHITE);
  }

  // Pronóstico futuro (línea discontinua)
  float prev = (nh > 0) ? v.histRealAt(nh - 1) : NAN;
  for (int h = 1; h <= GRAF_FUT; h++) {
    if (!isnan(prev) && !isnan(fut[h]) && ((h / 2) % 2 == 0))
      display.drawLine(GRAF_HIST - 2 + h, mapY(prev), GRAF_HIST - 1 + h, mapY(fut[h]), SSD1306_WHITE);
    prev = fut[h];
  }
}

void dibujarResumen() {
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("RESUMEN  RMSE (1 paso)");
  display.println("        Temp     Hum");

  int mejorT = -1, mejorH = -1;
  for (int k = 0; k < NUM_METODOS; k++) {
    float rt = varT.met[k].rmse(), rh = varH.met[k].rmse();
    if (!isnan(rt) && (mejorT < 0 || rt < varT.met[mejorT].rmse())) mejorT = k;
    if (!isnan(rh) && (mejorH < 0 || rh < varH.met[mejorH].rmse())) mejorH = k;
  }
  for (int k = 0; k < NUM_METODOS; k++) {
    char a[12], b[12];
    fmt(a, sizeof(a), varT.met[k].rmse(), 3);
    fmt(b, sizeof(b), varH.met[k].rmse(), 3);
    display.printf("%-6s%6s%c%7s%c\n", NOMBRE_CORTO[k], a, (k == mejorT) ? '*' : ' ',
                   b, (k == mejorH) ? '*' : ' ');
  }
  display.printf("Mejor T: %s\n", mejorT >= 0 ? NOMBRE_METODO[mejorT] : "--");
  display.printf("Mejor H: %s",   mejorH >= 0 ? NOMBRE_METODO[mejorH] : "--");
}

void dibujarPantalla(const RtcDateTime& now, float temperatura, float humedad) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  if (pantalla == 0) {
    dibujarPrincipal(now, temperatura, humedad);
  } else if (pantalla == PANTALLA_RESUMEN) {
    dibujarResumen();
  } else {
    int k   = (pantalla - 1) / SUBPANTALLAS;
    int sub = (pantalla - 1) % SUBPANTALLAS;
    if (sub == 0)      dibujarDatos(k);
    else if (sub == 1) dibujarGrafica(k, varT);
    else               dibujarGrafica(k, varH);
  }
  display.display();
}

// ============================================================
// PULSADOR (sondeo con antirrebote, corta / larga)
// ============================================================
void leerBoton() {
  static bool estadoEstable = HIGH, ultimoCrudo = HIGH;
  static unsigned long cambioMs = 0, presionadoMs = 0;
  static bool largaAtendida = false;

  bool crudo = digitalRead(PIN_BOTON);
  unsigned long ahora = millis();
  if (crudo != ultimoCrudo) { ultimoCrudo = crudo; cambioMs = ahora; }

  if ((ahora - cambioMs) > DEBOUNCE_MS && crudo != estadoEstable) {
    estadoEstable = crudo;
    if (estadoEstable == LOW) {                  // se presionó
      presionadoMs = ahora;
      largaAtendida = false;
    } else if (!largaAtendida) {                 // se soltó: pulsación corta
      pantalla = (pantalla + 1) % NUM_PANTALLAS; // tras el resumen vuelve a 0
      redibujar = true;
      ultimoBotonMs = ahora;
    }
  }

  // Pulsación larga: se atiende mientras se mantiene presionado
  if (estadoEstable == LOW && !largaAtendida && (ahora - presionadoMs) >= PULSACION_LARGA_MS) {
    largaAtendida = true;
    pantalla = 0;
    redibujar = true;
    ultimoBotonMs = ahora;
  }

  // Sin actividad: regresar a la pantalla principal de captura
  if (pantalla != 0 && (ahora - ultimoBotonMs) >= TIMEOUT_PANTALLA_MS) {
    pantalla = 0;
    redibujar = true;
  }
}

// ============================================================
// SALIDA SERIAL Y SD
// ============================================================
void imprimirMetricas(const RtcDateTime& t) {
  Serial.println("==============================================================");
  Serial.printf("METODOS BASICOS | %02d/%02d/%04d %02d:%02d | cada %d min | hist %d/%d | m=%d\n",
                t.Day(), t.Month(), t.Year(), t.Hour(), t.Minute(), INTERVALO_MIN,
                varT.serie.count, N_VALORES, M_ESTACIONAL);
  const Variable* vars[2] = {&varT, &varH};
  for (const Variable* v : vars) {
    Serial.printf("--- %s (real = %.2f %s) ---\n", v->nombre, v->serie.ultimo(), v->unidad);
    Serial.println("Modelo     PredUsada  PredSig      MAE     RMSE   MAPE(%)       R2      n");
    for (int k = 0; k < NUM_METODOS; k++) {
      const Metricas& m = v->met[k];
      Serial.printf("%-9s %9.3f %8.3f %8.4f %8.4f %9.4f %8.4f %6lu\n",
                    NOMBRE_METODO[k], v->predEval[k], v->predSig[k],
                    m.mae(), m.rmse(), m.mape(), m.r2(), (unsigned long)m.n);
    }
  }
  Serial.println("==============================================================");
}

void guardarSD(const RtcDateTime& now, float refT, float refH, float t, float h) {
  if (!sdOK) return;

  // Mismo formato que los otros sketches (las columnas de pronóstico van vacías;
  // los pronósticos de cada método se guardan en /pronosticos_basicos.csv)
  File f = SD.open("/datos_clima.csv", FILE_APPEND);
  if (f) {
    f.printf("%02d/%02d/%04d,%02d:%02d:%02d,%.2f,%.2f,%.2f,%.2f,,\n",
             now.Day(), now.Month(), now.Year(), now.Hour(), now.Minute(), now.Second(),
             refT, refH, t, h);
    f.close();
  } else {
    Serial.println("-> Error abriendo datos_clima.csv");
  }

  // Pronóstico que cada método hizo para ESTA muestra (permite recalcular
  // las métricas en Python)
  f = SD.open("/pronosticos_basicos.csv", FILE_APPEND);
  if (f) {
    if (f.size() == 0)
      f.println("Fecha,Hora,T,H,T_Average,H_Average,T_Naive,H_Naive,"
                "T_Seasonal_Naive,H_Seasonal_Naive,T_Drift,H_Drift");
    f.printf("%02d/%02d/%04d,%02d:%02d:%02d,%.2f,%.2f",
             now.Day(), now.Month(), now.Year(), now.Hour(), now.Minute(), now.Second(), t, h);
    for (int k = 0; k < NUM_METODOS; k++) {
      if (isnan(varT.predEval[k])) f.print(",");
      else f.printf(",%.3f", varT.predEval[k]);
      if (isnan(varH.predEval[k])) f.print(",");
      else f.printf(",%.3f", varH.predEval[k]);
    }
    f.println();
    f.close();
  }
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  pinMode(PIN_BOTON, INPUT_PULLUP);

  datosTelemetria.contador = 0;

  // -- INICIAR PANTALLA E I2C --
  Wire.begin(I2C_SDA, I2C_SCL);
  if (ENABLE_SCREEN && display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("Iniciando Sistema...");
    display.display();
  }

  // -- INICIAR MICRO SD --
  if (!SD.begin(SD_CS)) {
    Serial.println("Fallo al montar la SD o no hay tarjeta.");
    if (ENABLE_SCREEN) { display.println("Error SD!"); display.display(); }
    delay(2000);
  } else {
    sdOK = true;
    if (ENABLE_SCREEN) { display.println("SD OK!"); display.display(); }
    Serial.println("Tarjeta SD iniciada correctamente.");

    File file = SD.open("/datos_clima.csv", FILE_APPEND);
    if (file) {
      if (file.size() == 0) {
        file.println("Fecha,Hora,temp_original,humedad_original,temp_correjida,humedad_correjida,temp_pronosticada,humedad_pronosticada");
      }
      file.close();
    }
  }

  // -- INICIAR RELOJ --
  Rtc.Begin();
  Rtc.SetIsWriteProtected(false);
  if (!Rtc.GetIsRunning()) Rtc.SetIsRunning(true);

  // -- INICIAR SHT3X --
  if (!sht31.begin(0x44)) Serial.println(F("Error: No se encontro el SHT3X."));

  // -- WIFI / NTP (solo si el RTC no tiene una hora válida) --
  WiFi.mode(WIFI_STA);
  RtcDateTime now = Rtc.GetDateTime();
  RtcDateTime fechaReferencia(2026, 10, 01, 10, 0, 0);

  if (!now.IsValid() || now < fechaReferencia) {
    if (ENABLE_SCREEN) { display.println("Buscando WiFi..."); display.display(); }
    WiFi.begin(ssid, password);
    int intentos = 0;
    while (WiFi.status() != WL_CONNECTED && intentos < 20) { delay(500); intentos++; }

    if (WiFi.status() == WL_CONNECTED) {
      configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
      struct tm timeinfo;
      if (getLocalTime(&timeinfo, 5000)) {
        RtcDateTime internetTime(timeinfo.tm_year + 1900, timeinfo.tm_mon + 1, timeinfo.tm_mday,
                                 timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec);
        Rtc.SetDateTime(internetTime);
      }
      WiFi.disconnect();
    }
  }

  // -- PRECARGA DE LOS ÚLTIMOS N VALORES DESDE LA SD --
  if (PRECARGAR_SD && sdOK) {
    if (ENABLE_SCREEN) { display.println("Precargando SD..."); display.display(); }
    precargarDesdeSD();
  }

  // -- INICIALIZAR ESP-NOW --
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error inicializando ESP-NOW");
    return;
  }
  memcpy(peerInfo.peer_addr, macReceptor, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  if (esp_now_add_peer(&peerInfo) != ESP_OK) Serial.println("Failed to add peer");
}

// ============================================================
// LOOP
// ============================================================
void loop() {
  static unsigned long ultimaLecturaMs = 0;
  static RtcDateTime now;
  static float ref_temperatura = NAN, ref_humedad = NAN, temperatura = NAN, humedad = NAN;

  leerBoton();   // se sondea en cada vuelta (~10 ms) para no perder pulsaciones

  if (millis() - ultimaLecturaMs >= 500) {
    ultimaLecturaMs = millis();
    redibujar = true;

    now = Rtc.GetDateTime();
    ref_temperatura = sht31.readTemperature();
    ref_humedad = sht31.readHumidity();

    // CORRECCIÓN DE DATOS (igual que en los otros sketches)
    temperatura = (5.301868409 + (0.702422691 * ref_temperatura));
    humedad = (32.60425363 + (0.6384588546 * ref_humedad));

    if (!ENABLE_SCREEN) {
      if (now.IsValid())
        Serial.printf("[%02d/%02d/%04d %02d:%02d:%02d] ", now.Day(), now.Month(), now.Year(),
                      now.Hour(), now.Minute(), now.Second());
      else Serial.print("[Error RTC] ");
      if (isnan(temperatura) || isnan(humedad)) Serial.print("Error SHT3X | ");
      else Serial.printf("T: %.1f C  H: %.0f %% | ", temperatura, humedad);
      Serial.printf("Hist: %d/%d | ", varT.serie.count, N_VALORES);
      Serial.println(sdOK ? "SD: OK" : "SD: ERROR");
    }

    // ==========================================
    // UNA MUESTRA CADA INTERVALO_MIN MINUTOS: métodos, ESP-NOW y SD
    // ==========================================
    if (now.IsValid() && !isnan(temperatura) && !isnan(humedad)) {
      uint32_t muestraAbs = now.TotalSeconds() / (60UL * INTERVALO_MIN);
      // 0. Evalúa la predicción de cada método para esta muestra, agrega la
      //    lectura al buffer y calcula el pronóstico siguiente. Devuelve false
      //    si el intervalo ya estaba registrado (p. ej. venía en la precarga SD).
      if (registrarMuestra(muestraAbs, temperatura, humedad)) {
        imprimirMetricas(now);

        // 1. ESP-NOW
        datosTelemetria.contador++;
        datosTelemetria.uptime_ms = millis();
        datosTelemetria.temperatura = temperatura;
        datosTelemetria.humedad = humedad;
        datosTelemetria.heap_libre = ESP.getFreeHeap();
        datosTelemetria.estado = 0;
        esp_err_t result = esp_now_send(macReceptor, (uint8_t*)&datosTelemetria, sizeof(datosTelemetria));
        Serial.println(result == ESP_OK ? "-> ESP-NOW: Datos enviados con exito"
                                        : "-> ESP-NOW: Error en el envío");

        // 2. TX Serial
        Serial.printf(">>> TX: %02d/%02d/%04d,%02d:%02d:%02d,%.1f,%.0f\n",
                      now.Day(), now.Month(), now.Year(),
                      now.Hour(), now.Minute(), now.Second(), temperatura, humedad);

        // 3. MicroSD
        guardarSD(now, ref_temperatura, ref_humedad, temperatura, humedad);
      }
    }
  }

  if (ENABLE_SCREEN && redibujar) {
    redibujar = false;
    dibujarPantalla(now, temperatura, humedad);
  }

  delay(10);
}
