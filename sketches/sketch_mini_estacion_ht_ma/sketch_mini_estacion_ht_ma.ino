/*
 * MINI ESTACIÓN T/H - MODELO DE MEDIAS MÓVILES MA(q)
 * ==================================================
 * Implementación en ESP32 DevKit del cuaderno modelo_MA:
 *
 *    X(t) = mu + e(t) + theta1 e(t-1) + ... + thetaq e(t-q)
 *
 *   1. Se toma una muestra cada INTERVALO_MIN = 5 minutos y se guardan en
 *      memoria los últimos N valores (288 = 24 h).
 *   2. Se ajustan los modelos MA(0) ... MA(Q_MAX) y se elige el orden q con
 *      menor AIC (igual que el cuaderno, Q_MAX = 5).
 *   3. Estimación: el cuaderno usa ARIMA de statsmodels (máxima verosimilitud).
 *      En el ESP32 se usa el método equivalente de Suma Condicional de
 *      Cuadrados (CSS), que es como statsmodels arranca su optimización:
 *        a) Valores iniciales por Hannan-Rissanen: se ajusta un AR(m) largo
 *           por mínimos cuadrados, sus residuales estiman las innovaciones
 *           e(t) y luego se regresa X(t) sobre [1, e(t-1) ... e(t-q)].
 *        b) Refinamiento con Gauss-Newton / Levenberg-Marquardt minimizando
 *              SSE = sum e(t)^2,  e(t) = X(t) - mu - sum theta_i e(t-i)
 *      AIC = -2 logL + 2k,  BIC = -2 logL + k ln(n),  k = q + 2 (mu, thetas, sigma2)
 *   4. Predicción (cuaderno, sección 8):
 *        X(t+h) = mu + sum_{i=h..q} theta_i e(t+h-i)     para h <= q
 *        X(t+h) = mu                                     para h >  q
 *      -> la memoria del MA(q) es de solo q pasos; luego vuelve a la media.
 *   5. Métricas de AJUSTE (residuales CSS en la ventana) y FUERA DE MUESTRA a
 *      1 paso (en cada muestra se compara la lectura nueva con la predicción).
 *
 * El modelo se re-estima cada REAJUSTE_MUESTRAS muestras = 1 h (tarda < 1 s; mientras
 * tanto la pantalla muestra "Estimando...").
 *
 * PANTALLAS (pulsador en PIN_BOTON contra GND):
 *   Principal -> [Temperatura: modelo, errores, gráfica, tabla AIC]
 *             -> [Humedad: modelo, errores, gráfica, tabla AIC] -> Principal
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
// N_MIN_AJUSTE  : valores mínimos para el primer ajuste (24 = 2 h).
// REAJUSTE_MUESTRAS: cada cuántas muestras se re-estima el modelo (12 = 1 h).
// Q_MAX         : orden máximo evaluado (cuaderno: 5).
// M_AR_HR       : orden del AR largo de Hannan-Rissanen (valores iniciales).
// MAX_HUECO_MUESTRAS: huecos <= este número de muestras se rellenan con el
//                 último valor; si son mayores se reinicia la serie.
#define N_VALORES          MUESTRAS_DIA
#define N_MIN_AJUSTE       24
#define REAJUSTE_MUESTRAS  12
#define Q_MAX              5
#define M_AR_HR            10
#define MAX_ITER_LM        25
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
// ESTIMACIÓN MA(q)
// ============================================================

// Declaración adelantada: el IDE de Arduino genera los prototipos de las
// funciones justo antes de la primera función, que está antes de ModeloMA.
struct ModeloMA;

// Resuelve A x = b (k x k, fila mayor) por eliminación gaussiana con pivoteo
// parcial. La solución queda en b. Devuelve false si A es singular.
bool resolverSistema(double* A, double* b, int k) {
  for (int col = 0; col < k; col++) {
    int piv = col;
    for (int r = col + 1; r < k; r++)
      if (fabs(A[r * k + col]) > fabs(A[piv * k + col])) piv = r;
    if (fabs(A[piv * k + col]) < 1e-12) return false;
    if (piv != col) {
      for (int c = 0; c < k; c++) { double t = A[col * k + c]; A[col * k + c] = A[piv * k + c]; A[piv * k + c] = t; }
      double t = b[col]; b[col] = b[piv]; b[piv] = t;
    }
    for (int r = col + 1; r < k; r++) {
      double f = A[r * k + col] / A[col * k + col];
      for (int c = col; c < k; c++) A[r * k + c] -= f * A[col * k + c];
      b[r] -= f * b[col];
    }
  }
  for (int r = k - 1; r >= 0; r--) {
    double s = b[r];
    for (int c = r + 1; c < k; c++) s -= A[r * k + c] * b[c];
    b[r] = s / A[r * k + r];
  }
  return true;
}

/*
 * residualesCSS
 * -------------
 * e(t) = x(t) - mu - sum theta_i e(t-i), con e(t) = 0 antes del inicio.
 * Devuelve SSE = sum e(t)^2 (INFINITY si la recursión explota: modelo no
 * invertible). Opcionalmente acumula métricas de ajuste y devuelve las
 * últimas q innovaciones (epsFinal[0] = e(t) más reciente).
 */
double residualesCSS(const float* x, int n, int q, double mu, const double* th,
                     Metricas* met = nullptr, float* epsFinal = nullptr) {
  float e[Q_MAX] = {0};          // e[0] = e(t-1), e[1] = e(t-2), ...
  float thf[Q_MAX];
  for (int i = 0; i < q; i++) thf[i] = (float)th[i];
  const float muf = (float)mu;
  double sse = 0;
  for (int t = 0; t < n; t++) {
    float pred = muf;
    for (int i = 0; i < q; i++) pred += thf[i] * e[i];
    float et = x[t] - pred;
    if (!isfinite(et) || fabsf(et) > 1e6f) return INFINITY;
    sse += (double)et * et;
    if (met) met->agregar(x[t], pred);
    for (int i = q - 1; i > 0; i--) e[i] = e[i - 1];
    if (q > 0) e[0] = et;
  }
  if (epsFinal) for (int i = 0; i < Q_MAX; i++) epsFinal[i] = (i < q) ? e[i] : 0.0f;
  return sse;
}

/*
 * residualesARLargo (Hannan-Rissanen, paso 1)
 * Ajusta X(t) = a0 + a1 X(t-1) + ... + am X(t-m) por mínimos cuadrados y
 * guarda los residuales en ehat (ehat[t] = 0 para t < m).
 */
bool residualesARLargo(const float* x, int n, int m, float* ehat) {
  const int k = m + 1;
  double A[(M_AR_HR + 1) * (M_AR_HR + 1)] = {0}, b[M_AR_HR + 1] = {0};
  double fila[M_AR_HR + 1];
  for (int t = m; t < n; t++) {
    fila[0] = 1.0;
    for (int j = 1; j <= m; j++) fila[j] = x[t - j];
    for (int r = 0; r < k; r++) {
      b[r] += fila[r] * x[t];
      for (int c = 0; c < k; c++) A[r * k + c] += fila[r] * fila[c];
    }
  }
  if (!resolverSistema(A, b, k)) return false;
  for (int t = 0; t < n; t++) {
    if (t < m) { ehat[t] = 0; continue; }
    double p = b[0];
    for (int j = 1; j <= m; j++) p += b[j] * x[t - j];
    ehat[t] = (float)(x[t] - p);
  }
  return true;
}

/*
 * valoresInicialesHR (Hannan-Rissanen, paso 2)
 * Regresa X(t) sobre [1, ehat(t-1) ... ehat(t-q)].
 */
bool valoresInicialesHR(const float* x, const float* ehat, int n, int q, int m,
                        double& mu, double* th) {
  const int k = q + 1;
  double A[(Q_MAX + 1) * (Q_MAX + 1)] = {0}, b[Q_MAX + 1] = {0};
  double fila[Q_MAX + 1];
  for (int t = m + q; t < n; t++) {
    fila[0] = 1.0;
    for (int j = 1; j <= q; j++) fila[j] = ehat[t - j];
    for (int r = 0; r < k; r++) {
      b[r] += fila[r] * x[t];
      for (int c = 0; c < k; c++) A[r * k + c] += fila[r] * fila[c];
    }
  }
  if (!resolverSistema(A, b, k)) return false;
  mu = b[0];
  for (int i = 0; i < q; i++) th[i] = b[i + 1];
  return true;
}

/*
 * ajustarCSS
 * ----------
 * Levenberg-Marquardt sobre beta = (mu, theta1..thetaq). Las derivadas de
 * e(t) se obtienen con la misma recursión del modelo:
 *   de(t)/dmu      = -1      - sum theta_i de(t-i)/dmu
 *   de(t)/dtheta_j = -e(t-j) - sum theta_i de(t-i)/dtheta_j
 */
double ajustarCSS(const float* x, int n, int q, double& mu, double* th) {
  const int k = q + 1;
  double sse = residualesCSS(x, n, q, mu, th);
  if (!isfinite(sse)) return sse;
  double lambda = 1e-3;

  for (int it = 0; it < MAX_ITER_LM; it++) {
    double JtJ[(Q_MAX + 1) * (Q_MAX + 1)] = {0}, Jte[Q_MAX + 1] = {0};
    float e[Q_MAX] = {0};                      // e(t-1-i)
    float d[Q_MAX][Q_MAX + 1] = {{0}};         // d[i][j] = de(t-1-i)/dbeta_j
    float thf[Q_MAX];
    for (int i = 0; i < q; i++) thf[i] = (float)th[i];

    for (int t = 0; t < n; t++) {
      float pred = (float)mu;
      for (int i = 0; i < q; i++) pred += thf[i] * e[i];
      float et = x[t] - pred;

      float dt[Q_MAX + 1];
      for (int j = 0; j < k; j++) {
        float s = (j == 0) ? -1.0f : -e[j - 1];
        for (int i = 0; i < q; i++) s -= thf[i] * d[i][j];
        dt[j] = s;
      }
      for (int r = 0; r < k; r++) {
        Jte[r] += (double)dt[r] * et;
        for (int c = r; c < k; c++) JtJ[r * k + c] += (double)dt[r] * dt[c];
      }
      for (int i = q - 1; i > 0; i--) {
        e[i] = e[i - 1];
        for (int j = 0; j < k; j++) d[i][j] = d[i - 1][j];
      }
      if (q > 0) {
        e[0] = et;
        for (int j = 0; j < k; j++) d[0][j] = dt[j];
      }
    }
    for (int r = 0; r < k; r++)
      for (int c = 0; c < r; c++) JtJ[r * k + c] = JtJ[c * k + r];

    // Busca un paso que reduzca el SSE aumentando el amortiguamiento
    bool mejoro = false;
    while (lambda < 1e8) {
      double M[(Q_MAX + 1) * (Q_MAX + 1)], g[Q_MAX + 1];
      memcpy(M, JtJ, sizeof(double) * k * k);
      for (int r = 0; r < k; r++) { M[r * k + r] *= (1.0 + lambda); g[r] = -Jte[r]; }
      if (resolverSistema(M, g, k)) {
        double muN = mu + g[0], thN[Q_MAX];
        for (int i = 0; i < q; i++) thN[i] = th[i] + g[i + 1];
        double sseN = residualesCSS(x, n, q, muN, thN);
        if (sseN < sse) {
          double mejora = (sse - sseN) / sse;
          mu = muN;
          memcpy(th, thN, sizeof(double) * q);
          sse = sseN;
          lambda = fmax(lambda * 0.1, 1e-9);
          mejoro = mejora > 1e-9;
          break;
        }
      }
      lambda *= 10.0;
    }
    if (!mejoro) break;
    yield();
  }
  return sse;
}

// ============================================================
// MODELO MA(q) DE UNA VARIABLE (temperatura o humedad)
// ============================================================
float trabajoX[N_VALORES];     // copia lineal de la ventana (memoria de trabajo)
float trabajoE[N_VALORES];     // residuales del AR largo (Hannan-Rissanen)

struct ModeloMA {
  const char* nombre;
  const char* letra;     // "T" / "H"
  const char* unidad;
  float lo, hi;          // límites físicos del pronóstico

  Serie serie;

  // Modelo seleccionado
  bool   ok = false;
  int    q = 0;
  double mu = NAN, theta[Q_MAX] = {0}, sigma2 = NAN;
  int    nAjuste = 0;
  Metricas ajuste;                       // residuales CSS en la ventana
  double aic[Q_MAX + 1], bic[Q_MAX + 1]; // tabla de selección del orden

  // Innovaciones recientes: eps[0] = e(t), eps[1] = e(t-1), ...
  float eps[Q_MAX] = {0};

  // Evaluación fuera de muestra a 1 paso
  float    predSig  = NAN;
  float    predEval = NAN;
  Metricas unPaso;

  // Historia corta para la gráfica
  float histReal[GRAF_HIST];
  float histPred[GRAF_HIST];
  int   histHead = 0, histCount = 0;

  ModeloMA(const char* n, const char* l, const char* u, float lo_, float hi_)
    : nombre(n), letra(l), unidad(u), lo(lo_), hi(hi_) {
    for (int i = 0; i <= Q_MAX; i++) aic[i] = bic[i] = NAN;
  }

  // X(t+h) = mu + sum_{i=h..q} theta_i e(t+h-i)   (h > q -> mu)
  float pronosticoCrudo(int h) const {
    if (!ok) return NAN;
    double p = mu;
    for (int i = h; i <= q; i++) p += theta[i - 1] * eps[i - h];
    return (float)p;
  }
  float pronostico(int h) const {
    float p = pronosticoCrudo(h);
    if (!isnan(p)) { if (p < lo) p = lo; if (p > hi) p = hi; }
    return p;
  }

  float histRealAt(int i) const { return histReal[(histHead - histCount + i + 2 * GRAF_HIST) % GRAF_HIST]; }
  float histPredAt(int i) const { return histPred[(histHead - histCount + i + 2 * GRAF_HIST) % GRAF_HIST]; }

  void actualizar(float x, bool evaluar) {
    predEval = evaluar ? predSig : NAN;
    if (!isnan(predEval)) unPaso.agregar(x, predEval);

    histReal[histHead] = x;
    histPred[histHead] = predEval;
    histHead = (histHead + 1) % GRAF_HIST;
    if (histCount < GRAF_HIST) histCount++;

    // Nueva innovación e(t) = x(t) - X_hat(t|t-1)
    if (ok) {
      float e = x - pronosticoCrudo(1);
      if (!isfinite(e) || fabsf(e) > 1e3f) {     // recursión inestable: se reinicia
        memset(eps, 0, sizeof(eps));
      } else {
        for (int i = Q_MAX - 1; i > 0; i--) eps[i] = eps[i - 1];
        eps[0] = e;
      }
    }

    serie.push(x);
    predSig = pronostico(1);
  }

  void reiniciar() {           // hueco largo: se conservan los parámetros
    serie.limpiar();
    histHead = histCount = 0;
    memset(eps, 0, sizeof(eps));
    predSig = predEval = NAN;
  }

  // Ajusta MA(0..Q_MAX) sobre la ventana y elige q por AIC
  void ajustar() {
    const int n = serie.count;
    if (n < N_MIN_AJUSTE) return;
    for (int i = 0; i < n; i++) trabajoX[i] = serie.get(i);

    const int m = (n > 10 * M_AR_HR) ? M_AR_HR : 2;
    bool hrOk = residualesARLargo(trabajoX, n, m, trabajoE);

    double sumX = 0;
    for (int i = 0; i < n; i++) sumX += trabajoX[i];
    const double media = sumX / n;

    int    mejorQ = -1;
    double mejorMu = 0, mejorTh[Q_MAX] = {0}, mejorSSE = INFINITY;

    for (int qq = 0; qq <= Q_MAX; qq++) {
      double muI = media, thI[Q_MAX] = {0};
      // Valores iniciales de Hannan-Rissanen (si fallan: mu = media, theta = 0)
      if (qq > 0 && hrOk && !valoresInicialesHR(trabajoX, trabajoE, n, qq, m, muI, thI)) {
        muI = media;
        memset(thI, 0, sizeof(thI));
      }
      if (!isfinite(residualesCSS(trabajoX, n, qq, muI, thI))) {   // inicio no invertible
        muI = media;
        memset(thI, 0, sizeof(thI));
      }
      double sse = ajustarCSS(trabajoX, n, qq, muI, thI);

      if (isfinite(sse) && sse > 0) {
        double s2 = sse / n;
        double logL = -0.5 * n * (log(2.0 * M_PI * s2) + 1.0);
        int    kPar = qq + 2;                      // mu, thetas y sigma2
        aic[qq] = -2.0 * logL + 2.0 * kPar;
        bic[qq] = -2.0 * logL + kPar * log((double)n);
        if (mejorQ < 0 || aic[qq] < aic[mejorQ]) {
          mejorQ = qq; mejorMu = muI; mejorSSE = sse;
          memcpy(mejorTh, thI, sizeof(mejorTh));
        }
      } else {
        aic[qq] = bic[qq] = NAN;
      }
      yield();
    }
    if (mejorQ < 0) return;

    ok = true;
    q = mejorQ;
    mu = mejorMu;
    memcpy(theta, mejorTh, sizeof(theta));
    nAjuste = n;
    sigma2 = mejorSSE / n;

    // Residuales del modelo elegido e innovaciones actuales
    ajuste.limpiar();
    residualesCSS(trabajoX, n, q, mu, theta, &ajuste, eps);
    predSig = pronostico(1);
  }
};

ModeloMA maT("Temperatura", "T", "C", -INFINITY, INFINITY);
ModeloMA maH("Humedad", "H", "%", 0.0f, 100.0f);

uint32_t ultimaMuestraAbs = 0;   // nº de intervalo (desde 2000-01-01) de la última muestra
bool     hayMuestra = false;
int      muestrasDesdeAjuste = 0;
unsigned long duracionAjusteMs = 0;

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
      maT.reiniciar();
      maH.reiniciar();
    } else {
      // Huecos cortos: se repite el último valor (sin evaluar errores)
      for (int64_t i = 1; i < salto; i++) {
        maT.actualizar(maT.serie.ultimo(), false);
        maH.actualizar(maH.serie.ultimo(), false);
        muestrasDesdeAjuste++;
      }
    }
  }
  maT.actualizar(t, true);
  maH.actualizar(h, true);
  muestrasDesdeAjuste++;
  ultimaMuestraAbs = muestraAbs;
  hayMuestra = true;
  return true;
}

bool necesitaAjuste() {
  if (maT.serie.count < N_MIN_AJUSTE) return false;
  return !maT.ok || !maH.ok || muestrasDesdeAjuste >= REAJUSTE_MUESTRAS;
}

void ajustarModelos() {
  if (ENABLE_SCREEN) {
    display.fillRect(0, 54, SCREEN_WIDTH, 10, SSD1306_BLACK);
    display.setTextSize(1);
    display.setCursor(0, 54);
    display.print("Estimando MA(q)...");
    display.display();
  }
  unsigned long t0 = millis();
  maT.ajustar();
  maH.ajustar();
  duracionAjusteMs = millis() - t0;
  muestrasDesdeAjuste = 0;
  Serial.printf("-> MA re-estimado en %lu ms: T MA(%d), H MA(%d)\n",
                duracionAjusteMs, maT.q, maH.q);
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
                (unsigned long)leidas, maT.serie.count, N_VALORES);
}

// ============================================================
// PANTALLAS
// ============================================================
// 0 = principal; 1..4 = temperatura; 5..8 = humedad
#define SUBPANTALLAS   4       // modelo, errores, gráfica, tabla AIC
#define NUM_PANTALLAS  (1 + 2 * SUBPANTALLAS)

int pantalla = 0;
bool redibujar = true;
unsigned long ultimoBotonMs = 0;

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
  if (maT.ok && maH.ok) display.printf("MA   T:%.1fC H:%.0f%%", maT.predSig, maH.predSig);
  else display.printf("MA: faltan %d min", (N_MIN_AJUSTE - maT.serie.count) * INTERVALO_MIN);

  display.setCursor(0, 54);
  display.print(sdOK ? "SD:OK" : "SD:ERR");
  display.print("  [BTN] MA(q)");
}

void dibujarModelo(const ModeloMA& m) {
  char s[16];
  display.setTextSize(1);
  display.setCursor(0, 0);
  if (!m.ok) {
    display.printf("MA(q) %s\n\n", m.nombre);
    display.printf("Faltan %d min\n", (N_MIN_AJUSTE - m.serie.count) * INTERVALO_MIN);
    display.printf("\nHist: %d/%d", m.serie.count, N_VALORES);
    return;
  }
  display.printf("MA(%d) %s\n", m.q, m.nombre);
  display.printf("mu = %.4f\n", m.mu);
  // thetas en dos columnas (3 renglones para q <= 5)
  for (int i = 0; i < Q_MAX; i += 2) {
    if (i < m.q) display.printf("t%d=%7.3f", i + 1, m.theta[i]);
    if (i + 1 < m.q) display.printf(" t%d=%7.3f", i + 2, m.theta[i + 1]);
    display.print("\n");
  }
  if (m.q == 0) {
    display.setCursor(0, 16);
    display.print("(sin thetas: X=mu+e)");
  }
  display.setCursor(0, 40);
  display.printf("s2=%.4f AIC=%.0f\n", m.sigma2, m.aic[m.q]);
  fmt(s, sizeof(s), m.predSig, 2);
  display.printf("Pred +%dm: %s %s\n", INTERVALO_MIN, s, m.unidad);
  display.printf("Reajuste en %d min", max(0, REAJUSTE_MUESTRAS - muestrasDesdeAjuste) * INTERVALO_MIN);
}

void dibujarErrores(const ModeloMA& m) {
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.printf("MA(%d) %s  errores\n", m.q, m.letra);
  display.println("       Ajuste  1 paso");
  bool a = m.ok;
  filaTabla("MAE",   a ? m.ajuste.mae()  : NAN, m.unPaso.mae(),  4);
  filaTabla("RMSE",  a ? m.ajuste.rmse() : NAN, m.unPaso.rmse(), 4);
  filaTabla("MAPE%", a ? m.ajuste.mape() : NAN, m.unPaso.mape(), 3);
  filaTabla("R2",    a ? m.ajuste.r2()   : NAN, m.unPaso.r2(),   4);
  display.printf("%-5s%8lu%8lu\n", "n", a ? (unsigned long)m.ajuste.n : 0UL, (unsigned long)m.unPaso.n);
  display.print("Ajuste=ventana N");
}

void dibujarGrafica(const ModeloMA& m) {
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
  if (isinf(mn)) { display.printf("MA %s sin datos", m.letra); return; }
  display.printf("MA%d %s %.1f-%.1f%s", m.q, m.letra, mn, mx, m.unidad);
  if (mx - mn < 0.5f) { float c = 0.5f * (mx + mn); mn = c - 0.25f; mx = c + 0.25f; }

  auto mapY = [&](float y) -> int { return Y1 - (int)lroundf((y - mn) / (mx - mn) * (Y1 - Y0)); };

  // Línea vertical punteada = "ahora"
  for (int y = Y0; y <= Y1; y += 3) display.drawPixel(GRAF_HIST - 1, y, SSD1306_WHITE);

  // Media del proceso (línea punteada horizontal a la derecha)
  if (m.ok)
    for (int x = GRAF_HIST; x < SCREEN_WIDTH; x += 4) display.drawPixel(x, mapY((float)m.mu), SSD1306_WHITE);

  // Historia: real (línea) y predicción a 1 paso (puntos)
  int x0 = GRAF_HIST - nh;
  for (int i = 0; i < nh; i++) {
    if (i > 0)
      display.drawLine(x0 + i - 1, mapY(m.histRealAt(i - 1)), x0 + i, mapY(m.histRealAt(i)), SSD1306_WHITE);
    float p = m.histPredAt(i);
    if (!isnan(p) && (i % 2 == 0)) display.drawPixel(x0 + i, mapY(p), SSD1306_WHITE);
  }

  // Pronóstico futuro: llega a mu después de q pasos
  float prev = (nh > 0) ? m.histRealAt(nh - 1) : NAN;
  for (int h = 1; h <= GRAF_FUT; h++) {
    if (!isnan(prev) && !isnan(fut[h]))
      display.drawLine(GRAF_HIST - 2 + h, mapY(prev), GRAF_HIST - 1 + h, mapY(fut[h]), SSD1306_WHITE);
    prev = fut[h];
  }
}

// Tabla de selección del orden (cuaderno: "COMPARACIÓN DE MODELOS MA(q)")
void dibujarTablaAIC(const ModeloMA& m) {
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.printf("MA(q) %s  orden AIC\n", m.letra);
  display.println("q        AIC      BIC");
  for (int qq = 0; qq <= Q_MAX; qq++) {
    char a[12], b[12];
    fmt(a, sizeof(a), m.aic[qq], 1);
    fmt(b, sizeof(b), m.bic[qq], 1);
    display.printf("%d%c%9s%9s\n", qq, (m.ok && qq == m.q) ? '*' : ' ', a, b);
  }
}

void dibujarPantalla(const RtcDateTime& now, float temperatura, float humedad) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  if (pantalla == 0) {
    dibujarPrincipal(now, temperatura, humedad);
  } else {
    const ModeloMA& m = ((pantalla - 1) / SUBPANTALLAS == 0) ? maT : maH;
    switch ((pantalla - 1) % SUBPANTALLAS) {
      case 0: dibujarModelo(m);   break;
      case 1: dibujarErrores(m);  break;
      case 2: dibujarGrafica(m);  break;
      case 3: dibujarTablaAIC(m); break;
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
void imprimirBloque(const ModeloMA& m) {
  Serial.printf("--- %s (real = %.2f %s) ---\n", m.nombre, m.serie.ultimo(), m.unidad);
  if (!m.ok) { Serial.printf("Modelo no estimado (hist %d/%d)\n", m.serie.count, N_MIN_AJUSTE); return; }
  Serial.printf("Modelo MA(%d) ajustado con n=%d  (reajuste en %d min)\n",
                m.q, m.nAjuste, max(0, REAJUSTE_MUESTRAS - muestrasDesdeAjuste) * INTERVALO_MIN);
  Serial.printf("const (mu)       = %.6f\n", m.mu);
  for (int i = 0; i < m.q; i++) Serial.printf("ma.L%d            = %.6f\n", i + 1, m.theta[i]);
  Serial.printf("sigma2           = %.6f\n", m.sigma2);
  Serial.println("  q          AIC          BIC");
  for (int qq = 0; qq <= Q_MAX; qq++)
    Serial.printf("  %d %12.3f %12.3f%s\n", qq, m.aic[qq], m.bic[qq], qq == m.q ? "  <-" : "");
  Serial.printf("Ajuste   RMSE=%.6f MAE=%.6f R2=%.6f\n", m.ajuste.rmse(), m.ajuste.mae(), m.ajuste.r2());
  Serial.printf("1 paso   pred usada=%.4f  RMSE=%.6f MAE=%.6f MAPE=%.4f%% R2=%.6f n=%lu\n",
                m.predEval, m.unPaso.rmse(), m.unPaso.mae(), m.unPaso.mape(), m.unPaso.r2(),
                (unsigned long)m.unPaso.n);
  Serial.print("Pronostico h=1..q+1:");
  for (int h = 1; h <= m.q + 1; h++) Serial.printf(" %.3f", m.pronostico(h));
  Serial.println();
}

void imprimirMetricas(const RtcDateTime& t) {
  Serial.println("=======================================================");
  Serial.printf("MA(q) | %02d/%02d/%04d %02d:%02d | cada %d min | hist %d/%d\n",
                t.Day(), t.Month(), t.Year(), t.Hour(), t.Minute(), INTERVALO_MIN, maT.serie.count, N_VALORES);
  imprimirBloque(maT);
  imprimirBloque(maH);
  Serial.println("=======================================================");
}

void guardarSD(const RtcDateTime& now, float refT, float refH, float t, float h) {
  if (!sdOK) return;

  // Mismo formato que los otros sketches; el pronóstico guardado es el que
  // el MA(q) hizo para ESTA muestra.
  File f = SD.open("/datos_clima.csv", FILE_APPEND);
  if (f) {
    f.printf("%02d/%02d/%04d,%02d:%02d:%02d,%.2f,%.2f,%.2f,%.2f,",
             now.Day(), now.Month(), now.Year(), now.Hour(), now.Minute(), now.Second(),
             refT, refH, t, h);
    if (!isnan(maT.predEval) && !isnan(maH.predEval)) f.printf("%.2f,%.2f\n", maT.predEval, maH.predEval);
    else f.print(",\n");
    f.close();
  } else {
    Serial.println("-> Error abriendo datos_clima.csv");
  }

  // Pronósticos y orden del modelo en cada muestra
  f = SD.open("/modelo_ma.csv", FILE_APPEND);
  if (f) {
    if (f.size() == 0) f.println("Fecha,Hora,T,H,T_pred,H_pred,q_T,mu_T,q_H,mu_H");
    f.printf("%02d/%02d/%04d,%02d:%02d:%02d,%.2f,%.2f,",
             now.Day(), now.Month(), now.Year(), now.Hour(), now.Minute(), now.Second(), t, h);
    if (isnan(maT.predEval)) f.print(","); else f.printf("%.4f,", maT.predEval);
    if (isnan(maH.predEval)) f.print(","); else f.printf("%.4f,", maH.predEval);
    if (maT.ok) f.printf("%d,%.6f,", maT.q, maT.mu); else f.print(",,");
    if (maH.ok) f.printf("%d,%.6f\n", maH.q, maH.mu); else f.print(",\n");
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

  // -- PRECARGA DE LOS ÚLTIMOS N VALORES DESDE LA SD Y PRIMER AJUSTE --
  if (PRECARGAR_SD && sdOK) {
    if (ENABLE_SCREEN) { display.println("Precargando SD..."); display.display(); }
    precargarDesdeSD();
    if (necesitaAjuste()) ajustarModelos();
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
      if (maT.ok && maH.ok) Serial.printf("MA T: %.1f C  H: %.0f %% | ", maT.predSig, maH.predSig);
      Serial.println(sdOK ? "SD: OK" : "SD: ERROR");
    }

    // ==========================================
    // UNA MUESTRA CADA INTERVALO_MIN MINUTOS: MA(q), ESP-NOW y SD
    // ==========================================
    if (now.IsValid() && !isnan(temperatura) && !isnan(humedad)) {
      uint32_t muestraAbs = now.TotalSeconds() / (60UL * INTERVALO_MIN);
      // 0. Evalúa la predicción hecha para esta muestra, actualiza la
      //    innovación e(t) y calcula el pronóstico siguiente. Devuelve false
      //    si el intervalo ya estaba registrado (p. ej. por la precarga).
      if (registrarMuestra(muestraAbs, temperatura, humedad)) {

        // 1. ESP-NOW y SD primero (no esperan a la re-estimación)
        datosTelemetria.contador++;
        datosTelemetria.uptime_ms = millis();
        datosTelemetria.temperatura = temperatura;
        datosTelemetria.humedad = humedad;
        datosTelemetria.heap_libre = ESP.getFreeHeap();
        datosTelemetria.estado = 0;
        esp_err_t result = esp_now_send(macReceptor, (uint8_t*)&datosTelemetria, sizeof(datosTelemetria));
        Serial.println(result == ESP_OK ? "-> ESP-NOW: Datos enviados con exito"
                                        : "-> ESP-NOW: Error en el envío");

        Serial.printf(">>> TX: %02d/%02d/%04d,%02d:%02d:%02d,%.1f,%.0f\n",
                      now.Day(), now.Month(), now.Year(),
                      now.Hour(), now.Minute(), now.Second(), temperatura, humedad);

        guardarSD(now, ref_temperatura, ref_humedad, temperatura, humedad);

        // 2. Re-estimación periódica del modelo
        if (necesitaAjuste()) ajustarModelos();
        imprimirMetricas(now);
      }
    }
  }

  if (ENABLE_SCREEN && redibujar) {
    redibujar = false;
    dibujarPantalla(now, temperatura, humedad);
  }

  delay(10);
}
