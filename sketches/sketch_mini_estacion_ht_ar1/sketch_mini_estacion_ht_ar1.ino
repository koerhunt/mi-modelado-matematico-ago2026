/*
 * MINI ESTACIÓN T/H - MODELO AUTORREGRESIVO AR(1)
 * ===============================================
 * Implementación en ESP32 DevKit del cuaderno 08_modelo_ar:
 *
 *        X(t) = c + phi * X(t-1) + e(t)
 *
 *   1. Se toma una muestra cada INTERVALO_MIN = 5 minutos y se guardan en
 *      memoria los últimos N valores (288 = 24 h).
 *   2. Con ellos se forman los pares (X = X(t-1), Y = X(t)).
 *   3. Mínimos cuadrados ordinarios (igual que el cuaderno):
 *          phi = Sxy / Sxx      c = Ymedia - phi * Xmedia
 *      con Sxx = sum (X - Xmedia)^2   y   Sxy = sum (X - Xmedia)(Y - Ymedia)
 *   4. Métricas de AJUSTE sobre la ventana (RMSE, MAE, R2 del cuaderno).
 *   5. Además, en cada muestra se compara la lectura nueva contra la predicción
 *      a 1 paso (5 min) hecha en la muestra anterior: métricas FUERA DE MUESTRA.
 *   6. Pronóstico a h pasos: X(t+h) = c + phi * X(t+h-1)  -> tiende a
 *      la media del proceso mu = c / (1 - phi) si |phi| < 1.
 *
 * El primer ajuste se hace al completar un ciclo de 24 h (288 muestras); después
 * el modelo se re-estima en cada muestra (ventana móvil de las últimas 24 h).
 * Al arrancar se recargan los últimos N registros de /datos_clima.csv.
 *
 * PANTALLAS (pulsador en PIN_BOTON contra GND):
 *   Principal -> [Temperatura: modelo, errores, gráfica, dispersión]
 *             -> [Humedad: modelo, errores, gráfica, dispersión] -> Principal
 *   - Pulsación corta : siguiente pantalla (después de la última regresa a la principal).
 *   - Pulsación larga : regresa directo a la pantalla principal.
 *   - Sin pulsar durante TIMEOUT_PANTALLA_MS: regresa a la principal.
 */

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
#include <string.h>

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

// --- 7. PARÁMETROS DEL MODELO ---
// INTERVALO_MIN : minutos entre muestras del modelo (el sensor se sigue leyendo
//                 cada 500 ms para la pantalla, pero al modelo, a la SD y a
//                 ESP-NOW solo entra una muestra cada INTERVALO_MIN minutos,
//                 alineada al reloj: hh:00, hh:05, hh:10, ...).
#define INTERVALO_MIN    5
#define MUESTRAS_DIA     (24 * 60 / INTERVALO_MIN)     // 288 muestras = 24 h
// N_VALORES     : valores en memoria por variable = ventana de ajuste (24 h).
// N_MIN_AJUSTE  : valores mínimos para estimar el AR(1). Se espera un ciclo
//                 completo de 24 h (288 muestras) antes de estimar el modelo y
//                 empezar a pronosticar el siguiente ciclo.
// MAX_HUECO_MUESTRAS: huecos <= este número de muestras se rellenan con el
//                 último valor; si son mayores se reinicia la serie.
#define N_VALORES          MUESTRAS_DIA
#define N_MIN_AJUSTE       MUESTRAS_DIA                // 288 muestras = 24 h
#define MAX_HUECO_MUESTRAS 3                           // 15 min
#define PRECARGAR_SD       true

// Gráficas: GRAF_HIST muestras de historia (8 h) + GRAF_FUT de pronóstico (~2.7 h) = 128 px
#define GRAF_HIST  96
#define GRAF_FUT   (SCREEN_WIDTH - GRAF_HIST)

// ============================================================
// BUFFER CIRCULAR: últimos N valores
// ============================================================
struct Serie {
  float buf[N_VALORES];
  int   head  = 0;        // siguiente posición de escritura
  int   count = 0;        // valores almacenados

  void limpiar() { head = 0; count = 0; }

  void push(float x) {
    buf[head] = x;
    head = (head + 1) % N_VALORES;
    if (count < N_VALORES) count++;
  }

  // i = 0 -> más antiguo, i = count-1 -> más reciente
  float get(int i) const { return buf[(head - count + i + 2 * N_VALORES) % N_VALORES]; }
  float ultimo()   const { return get(count - 1); }
};

// ============================================================
// MÉTRICAS ACUMULADAS: MAE, RMSE, MAPE, R2
// ============================================================
struct Metricas {
  uint32_t n = 0, nApe = 0;
  double sAbs = 0, sSq = 0, sApe = 0, sY = 0, sY2 = 0;

  void limpiar() { *this = Metricas(); }

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
  float r2()   const {                                  // R2 = 1 - SSE / SST
    if (n < 2) return NAN;
    double sst = sY2 - sY * sY / n;
    if (sst < 1e-9) return NAN;
    return (float)(1.0 - sSq / sst);
  }
};

// ============================================================
// MODELO AR(1) DE UNA VARIABLE (temperatura o humedad)
// ============================================================
struct ModeloAR1 {
  const char* nombre;
  const char* letra;     // "T" / "H"
  const char* unidad;
  float lo, hi;          // límites físicos del pronóstico

  Serie serie;

  // Parámetros estimados (cuaderno 08, pasos 3 a 10)
  bool   ok = false;
  int    nPares = 0;
  double xMedia = 0, yMedia = 0, sxx = 0, sxy = 0;
  double phi = NAN, c = NAN;
  Metricas ajuste;       // residuales dentro de la ventana (in-sample)

  // Evaluación fuera de muestra a 1 paso
  float    predSig  = NAN;   // pronóstico para la próxima muestra
  float    predEval = NAN;   // pronóstico que se evaluó en la última muestra
  Metricas unPaso;

  // Historia corta para la gráfica
  float histReal[GRAF_HIST];
  float histPred[GRAF_HIST];
  int   histHead = 0, histCount = 0;

  ModeloAR1(const char* n, const char* l, const char* u, float lo_, float hi_)
    : nombre(n), letra(l), unidad(u), lo(lo_), hi(hi_) {}

  float media() const {                       // mu = c / (1 - phi)
    if (!ok || fabs(1.0 - phi) < 1e-9) return NAN;
    return (float)(c / (1.0 - phi));
  }
  bool estacionario() const { return ok && fabs(phi) < 1.0; }

  // Mínimos cuadrados ordinarios sobre los pares de la ventana
  void estimar() {
    ok = false;
    nPares = serie.count - 1;
    if (serie.count < N_MIN_AJUSTE) return;

    // Medias de X = X(t-1) e Y = X(t)
    double sX = 0, sY = 0;
    for (int i = 0; i < nPares; i++) { sX += serie.get(i); sY += serie.get(i + 1); }
    xMedia = sX / nPares;
    yMedia = sY / nPares;

    // Sxx y Sxy con desviaciones respecto a las medias
    sxx = 0; sxy = 0;
    for (int i = 0; i < nPares; i++) {
      double dx = serie.get(i) - xMedia;
      double dy = serie.get(i + 1) - yMedia;
      sxx += dx * dx;
      sxy += dx * dy;
    }
    if (sxx < 1e-9) return;                   // serie constante: phi no está definido

    phi = sxy / sxx;
    c   = yMedia - phi * xMedia;
    ok  = true;

    // Residuales del ajuste: RMSE, MAE, MAPE, R2
    ajuste.limpiar();
    for (int i = 0; i < nPares; i++) {
      float x = serie.get(i), y = serie.get(i + 1);
      ajuste.agregar(y, (float)(c + phi * x));
    }
  }

  // Pronóstico a h pasos: se itera X(t+1) = c + phi * X(t)
  float pronostico(int h) const {
    if (!ok || serie.count == 0) return NAN;
    double x = serie.ultimo();
    for (int i = 0; i < h; i++) x = c + phi * x;
    float p = (float)x;
    if (p < lo) p = lo;
    if (p > hi) p = hi;
    return p;
  }

  float histRealAt(int i) const { return histReal[(histHead - histCount + i + 2 * GRAF_HIST) % GRAF_HIST]; }
  float histPredAt(int i) const { return histPred[(histHead - histCount + i + 2 * GRAF_HIST) % GRAF_HIST]; }

  // Registra una lectura real. Si evaluar = true se acumula el error de la
  // predicción hecha para esta muestra.
  void actualizar(float x, bool evaluar) {
    predEval = evaluar ? predSig : NAN;
    if (!isnan(predEval)) unPaso.agregar(x, predEval);

    histReal[histHead] = x;
    histPred[histHead] = predEval;
    histHead = (histHead + 1) % GRAF_HIST;
    if (histCount < GRAF_HIST) histCount++;

    serie.push(x);
    estimar();
    predSig = pronostico(1);
  }

  // Hubo un hueco largo: la serie deja de ser equiespaciada y se reinicia.
  // Las métricas a 1 paso acumuladas se conservan.
  void reiniciar() {
    serie.limpiar();
    histHead = histCount = 0;
    ok = false;
    predSig = predEval = NAN;
  }
};

ModeloAR1 arT("Temperatura", "T", "C", -INFINITY, INFINITY);
ModeloAR1 arH("Humedad", "H", "%", 0.0f, 100.0f);

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
      arT.reiniciar();
      arH.reiniciar();
    } else {
      // Huecos cortos: se repite el último valor (sin evaluar errores)
      for (int64_t i = 1; i < salto; i++) {
        arT.actualizar(arT.serie.ultimo(), false);
        arH.actualizar(arH.serie.ultimo(), false);
      }
    }
  }
  arT.actualizar(t, true);
  arH.actualizar(h, true);
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
                (unsigned long)leidas, arT.serie.count, N_VALORES);
}

// ============================================================
// PANTALLAS
// ============================================================
// 0 = principal; 1..4 = temperatura; 5..8 = humedad
#define SUBPANTALLAS   4       // modelo, errores, gráfica, dispersión
#define NUM_PANTALLAS  (1 + 2 * SUBPANTALLAS)

int pantalla = 0;
bool redibujar = true;
unsigned long ultimoBotonMs = 0;

// Minutos que faltan para completar N_MIN_AJUSTE muestras, como "23h55m"
void fmtFaltan(char* out, size_t len, int muestrasEnMemoria) {
  int minutos = (N_MIN_AJUSTE - muestrasEnMemoria) * INTERVALO_MIN;
  if (minutos < 0) minutos = 0;
  snprintf(out, len, "%dh%02dm", minutos / 60, minutos % 60);
}

void fmt(char* out, size_t len, float v, int dec) {
  if (isnan(v) || isinf(v)) snprintf(out, len, "--");
  else snprintf(out, len, "%.*f", dec, v);
}

void filaTabla(const char* etiqueta, float a, float b, int dec) {
  char sa[12], sb[12];
  fmt(sa, sizeof(sa), a, dec);
  fmt(sb, sizeof(sb), b, dec);
  display.printf("%-5s%8s%8s\n", etiqueta, sa, sb);
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
  if (arT.ok && arH.ok) display.printf("AR1  T:%.1fC H:%.0f%%", arT.predSig, arH.predSig);
  else {
    char f[12];
    fmtFaltan(f, sizeof(f), arT.serie.count);
    display.printf("AR1: faltan %s", f);
  }

  display.setCursor(0, 54);
  display.print(sdOK ? "SD:OK" : "SD:ERR");
  display.print("  [BTN] AR(1)");
}

void dibujarModelo(const ModeloAR1& m) {
  char s[16];
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.printf("AR(1) %s\n", m.nombre);
  if (!m.ok) {
    if (m.serie.count < N_MIN_AJUSTE) {
      char f[12];
      fmtFaltan(f, sizeof(f), m.serie.count);
      display.printf("\nCiclo de 24h: faltan\n%s\n", f);
    }
    else display.println("\nSxx = 0 (serie\nconstante)");
    display.printf("\nHist: %d/%d", m.serie.count, N_VALORES);
    return;
  }
  display.printf("phi = %.6f\n", m.phi);
  display.printf("c   = %.6f\n", m.c);
  fmt(s, sizeof(s), m.media(), 2);
  display.printf("mu=c/(1-phi) %s\n", s);
  display.printf("|phi|<1: %s\n", m.estacionario() ? "SI estac." : "NO");
  fmt(s, sizeof(s), m.predSig, 2);
  display.printf("Pred +%dm: %s %s\n", INTERVALO_MIN, s, m.unidad);
  display.printf("Pares: %d\n", m.nPares);
  display.print("X(t)=c+phi*X(t-1)");
}

void dibujarErrores(const ModeloAR1& m) {
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.printf("AR(1) %s  errores\n", m.letra);
  display.println("       Ajuste  1 paso");
  bool a = m.ok;
  filaTabla("MAE",   a ? m.ajuste.mae()  : NAN, m.unPaso.mae(),  4);
  filaTabla("RMSE",  a ? m.ajuste.rmse() : NAN, m.unPaso.rmse(), 4);
  filaTabla("MAPE%", a ? m.ajuste.mape() : NAN, m.unPaso.mape(), 3);
  filaTabla("R2",    a ? m.ajuste.r2()   : NAN, m.unPaso.r2(),   4);
  display.printf("%-5s%8lu%8lu\n", "n", a ? (unsigned long)m.ajuste.n : 0UL, (unsigned long)m.unPaso.n);
  display.print("Ajuste=ventana N");
}

void dibujarGrafica(const ModeloAR1& m) {
  const int Y0 = 10, Y1 = SCREEN_HEIGHT - 1;
  const int nh = m.histCount;

  float fut[GRAF_FUT + 1];
  for (int h = 1; h <= GRAF_FUT; h++) fut[h] = m.pronostico(h);

  // Escala vertical automática
  float mn = INFINITY, mx = -INFINITY;
  auto incluir = [&](float y) { if (!isnan(y)) { if (y < mn) mn = y; if (y > mx) mx = y; } };
  for (int i = 0; i < nh; i++) { incluir(m.histRealAt(i)); incluir(m.histPredAt(i)); }
  for (int h = 1; h <= GRAF_FUT; h++) incluir(fut[h]);

  display.setTextSize(1);
  display.setCursor(0, 0);
  if (isinf(mn)) { display.printf("AR1 %s sin datos", m.letra); return; }
  display.printf("AR1 %s %.1f-%.1f%s", m.letra, mn, mx, m.unidad);
  if (mx - mn < 0.5f) { float c = 0.5f * (mx + mn); mn = c - 0.25f; mx = c + 0.25f; }

  auto mapY = [&](float y) -> int { return Y1 - (int)lroundf((y - mn) / (mx - mn) * (Y1 - Y0)); };

  // Línea vertical punteada = "ahora"
  for (int y = Y0; y <= Y1; y += 3) display.drawPixel(GRAF_HIST - 1, y, SSD1306_WHITE);

  // Historia: real (línea) y predicción a 1 paso (puntos)
  int x0 = GRAF_HIST - nh;
  for (int i = 0; i < nh; i++) {
    if (i > 0)
      display.drawLine(x0 + i - 1, mapY(m.histRealAt(i - 1)), x0 + i, mapY(m.histRealAt(i)), SSD1306_WHITE);
    float p = m.histPredAt(i);
    if (!isnan(p) && (i % 2 == 0)) display.drawPixel(x0 + i, mapY(p), SSD1306_WHITE);
  }

  // Pronóstico futuro (línea discontinua)
  float prev = (nh > 0) ? m.histRealAt(nh - 1) : NAN;
  for (int h = 1; h <= GRAF_FUT; h++) {
    if (!isnan(prev) && !isnan(fut[h]) && ((h / 2) % 2 == 0))
      display.drawLine(GRAF_HIST - 2 + h, mapY(prev), GRAF_HIST - 1 + h, mapY(fut[h]), SSD1306_WHITE);
    prev = fut[h];
  }
}

// Gráfica "Identificación del modelo AR(1)" del cuaderno:
// dispersión X(t) vs X(t-1) y recta de regresión Y = c + phi X
void dibujarDispersion(const ModeloAR1& m) {
  const int X0 = 0, X1 = SCREEN_WIDTH - 1, Y0 = 10, Y1 = SCREEN_HEIGHT - 1;
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.printf("%s(t) vs %s(t-1)", m.letra, m.letra);
  if (m.serie.count < 2) return;

  float mn = INFINITY, mx = -INFINITY;
  for (int i = 0; i < m.serie.count; i++) {
    float v = m.serie.get(i);
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  if (mx - mn < 0.5f) { float c = 0.5f * (mx + mn); mn = c - 0.25f; mx = c + 0.25f; }
  display.printf(" %.0f-%.0f", mn, mx);

  auto mapX = [&](float v) -> int { return X0 + (int)lroundf((v - mn) / (mx - mn) * (X1 - X0)); };
  auto mapY = [&](float v) -> int { return Y1 - (int)lroundf((v - mn) / (mx - mn) * (Y1 - Y0)); };

  // Pares observados
  for (int i = 0; i + 1 < m.serie.count; i++)
    display.drawPixel(mapX(m.serie.get(i)), mapY(m.serie.get(i + 1)), SSD1306_WHITE);

  // Recta de regresión (punteada)
  if (m.ok) {
    for (int px = X0; px <= X1; px += 2) {
      float x = mn + (mx - mn) * (px - X0) / (float)(X1 - X0);
      int py = mapY((float)(m.c + m.phi * x));
      if (py >= Y0 && py <= Y1) display.drawPixel(px, py, SSD1306_WHITE);
    }
  }
}

void dibujarPantalla(const RtcDateTime& now, float temperatura, float humedad) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  if (pantalla == 0) {
    dibujarPrincipal(now, temperatura, humedad);
  } else {
    const ModeloAR1& m = ((pantalla - 1) / SUBPANTALLAS == 0) ? arT : arH;
    switch ((pantalla - 1) % SUBPANTALLAS) {
      case 0: dibujarModelo(m);     break;
      case 1: dibujarErrores(m);    break;
      case 2: dibujarGrafica(m);    break;
      case 3: dibujarDispersion(m); break;
    }
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
      pantalla = (pantalla + 1) % NUM_PANTALLAS; // tras la última vuelve a 0
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
void imprimirBloque(const ModeloAR1& m) {
  Serial.printf("--- %s (real = %.2f %s) ---\n", m.nombre, m.serie.ultimo(), m.unidad);
  if (!m.ok) { Serial.printf("Modelo no estimado (hist %d/%d)\n", m.serie.count, N_MIN_AJUSTE); return; }
  Serial.printf("Numero de pares  = %d\n", m.nPares);
  Serial.printf("Media X          = %.8f\n", m.xMedia);
  Serial.printf("Media Y          = %.8f\n", m.yMedia);
  Serial.printf("Sxx              = %.8f\n", m.sxx);
  Serial.printf("Sxy              = %.8f\n", m.sxy);
  Serial.printf("Phi              = %.8f  (%s)\n", m.phi, m.estacionario() ? "|phi| < 1" : "|phi| >= 1");
  Serial.printf("Intercepto       = %.8f\n", m.c);
  Serial.printf("Modelo: %s(t) = %.8f + (%.8f)*%s(t-1)\n", m.letra, m.c, m.phi, m.letra);
  Serial.printf("Ajuste   RMSE=%.6f MAE=%.6f R2=%.6f\n", m.ajuste.rmse(), m.ajuste.mae(), m.ajuste.r2());
  Serial.printf("1 paso   pred usada=%.4f  RMSE=%.6f MAE=%.6f MAPE=%.4f%% R2=%.6f n=%lu\n",
                m.predEval, m.unPaso.rmse(), m.unPaso.mae(), m.unPaso.mape(), m.unPaso.r2(),
                (unsigned long)m.unPaso.n);
  Serial.printf("Pronostico t+1   = %.4f %s  (+%d min)\n", m.predSig, m.unidad, INTERVALO_MIN);
}

void imprimirMetricas(const RtcDateTime& t) {
  Serial.println("=======================================================");
  Serial.printf("AR(1) | %02d/%02d/%04d %02d:%02d | cada %d min | hist %d/%d\n",
                t.Day(), t.Month(), t.Year(), t.Hour(), t.Minute(), INTERVALO_MIN, arT.serie.count, N_VALORES);
  imprimirBloque(arT);
  imprimirBloque(arH);
  Serial.println("=======================================================");
}

void guardarSD(const RtcDateTime& now, float refT, float refH, float t, float h) {
  if (!sdOK) return;

  // Mismo formato que los otros sketches; el pronóstico guardado es el que
  // el AR(1) hizo para ESTA muestra.
  File f = SD.open("/datos_clima.csv", FILE_APPEND);
  if (f) {
    f.printf("%02d/%02d/%04d,%02d:%02d:%02d,%.2f,%.2f,%.2f,%.2f,",
             now.Day(), now.Month(), now.Year(), now.Hour(), now.Minute(), now.Second(),
             refT, refH, t, h);
    if (!isnan(arT.predEval) && !isnan(arH.predEval)) f.printf("%.2f,%.2f\n", arT.predEval, arH.predEval);
    else f.print(",\n");
    f.close();
  } else {
    Serial.println("-> Error abriendo datos_clima.csv");
  }

  // Parámetros del modelo en cada muestra
  f = SD.open("/modelo_ar1.csv", FILE_APPEND);
  if (f) {
    if (f.size() == 0) f.println("Fecha,Hora,T,H,T_pred,H_pred,phi_T,c_T,phi_H,c_H");
    f.printf("%02d/%02d/%04d,%02d:%02d:%02d,%.2f,%.2f,",
             now.Day(), now.Month(), now.Year(), now.Hour(), now.Minute(), now.Second(), t, h);
    if (isnan(arT.predEval)) f.print(","); else f.printf("%.4f,", arT.predEval);
    if (isnan(arH.predEval)) f.print(","); else f.printf("%.4f,", arH.predEval);
    if (arT.ok) f.printf("%.8f,%.8f,", arT.phi, arT.c); else f.print(",,");
    if (arH.ok) f.printf("%.8f,%.8f\n", arH.phi, arH.c); else f.print(",\n");
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
      if (arT.ok && arH.ok) Serial.printf("AR1 T: %.1f C  H: %.0f %% | ", arT.predSig, arH.predSig);
      Serial.println(sdOK ? "SD: OK" : "SD: ERROR");
    }

    // ==========================================
    // UNA MUESTRA CADA INTERVALO_MIN MINUTOS: AR(1), ESP-NOW y SD
    // ==========================================
    if (now.IsValid() && !isnan(temperatura) && !isnan(humedad)) {
      uint32_t muestraAbs = now.TotalSeconds() / (60UL * INTERVALO_MIN);
      // 0. Evalúa la predicción hecha para esta muestra, agrega la lectura,
      //    re-estima phi y c y calcula el pronóstico siguiente. Devuelve
      //    false si el intervalo ya estaba registrado (p. ej. por la precarga).
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
