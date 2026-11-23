/*
 * MINI ESTACIÓN T/H - MODELO AUTORREGRESIVO AR(p)
 * ===============================================
 * Implementación en ESP32 DevKit del cuaderno 08b_modelo_ar_p (generaliza
 * el sketch AR(1); con P_FIJO = 1 se reproduce el AR(1)):
 *
 *     X(t) = c + phi_1 X(t-1) + phi_2 X(t-2) + ... + phi_p X(t-p) + e(t)
 *
 * El álgebra lineal se hace con Eigen (biblioteca "ArduinoEigen" de
 * hideakitai en el Gestor de bibliotecas del IDE de Arduino).
 *
 *   1. Se toma una muestra cada INTERVALO_MIN = 5 minutos y se guardan en
 *      memoria los últimos N valores (288 = 24 h).
 *   2. SELECCIÓN DEL ORDEN p (cuaderno CRITERIOS_AIC_BIC):
 *        - Se ajustan AR(1) ... AR(P_MAX) sobre LA MISMA muestra (se descartan
 *          los primeros P_MAX datos) y se calculan, con verosimilitud gaussiana,
 *              -2 ln L = n [ln(2 pi SSE/n) + 1]
 *              AIC = -2 ln L + 2k      BIC = -2 ln L + k ln(n),   k = p + 2
 *          (k cuenta c, phi_1..phi_p y sigma2, igual que statsmodels).
 *        - PACF muestral (ecuaciones de Yule-Walker) con bandas +-1.96/sqrt(N).
 *        - Se usa el p que minimiza el criterio elegido (CRITERIO_BIC), o un
 *          orden fijo si P_FIJO > 0.
 *   3. Estimación por mínimos cuadrados ordinarios con el p elegido
 *      (n = N - p filas, igual que el cuaderno):
 *          X = [1, x(t-1), ..., x(t-p)],  y = x(t)
 *          beta = (c, phi_1..phi_p) = argmin ||y - X beta||^2   (QR de Eigen)
 *   4. Estacionariedad: las raíces de  lambda^p - phi_1 lambda^(p-1) - ... - phi_p = 0
 *      son los valores propios de la matriz compañera (Eigen::EigenSolver);
 *      estacionario si max|lambda| < 1. Media del proceso  mu = c / (1 - sum phi_i).
 *   5. Métricas de AJUSTE sobre la ventana (RMSE, MAE, MAPE, R2) y comparación
 *      contra el modelo de persistencia X(t) ~ X(t-1).
 *   6. En cada muestra se compara la lectura nueva contra la predicción a
 *      1 paso hecha en la muestra anterior: métricas FUERA DE MUESTRA.
 *   7. Pronóstico recursivo a h pasos:
 *          X^(t+h) = c + sum phi_i X^(t+h-i),   X^(t+j) = X(t+j) si j <= 0
 *
 * El primer ajuste se hace al completar un ciclo de 24 h (288 muestras); después
 * el modelo (orden y coeficientes) se re-estima en cada muestra (ventana móvil
 * de las últimas 24 h). Al arrancar se recargan los últimos N registros de
 * /datos_clima.csv.
 *
 * PANTALLAS (pulsador en PIN_BOTON contra GND):
 *   Principal -> [Temperatura: modelo, diagnóstico, errores, gráfica, orden AIC, PACF]
 *             -> [Humedad:     modelo, diagnóstico, errores, gráfica, orden AIC, PACF]
 *             -> Principal
 *   - Pulsación corta : siguiente pantalla (después de la última regresa a la principal).
 *   - Pulsación larga : regresa directo a la pantalla principal.
 *   - Sin pulsar durante TIMEOUT_PANTALLA_MS: regresa a la principal.
 */

#include <ArduinoEigenDense.h>   // Eigen: Core, QR, Cholesky, Eigenvalues
#include <WiFi.h>
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

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXcd;

// --- 1. DATOS WI-FI (solo para sincronizar el RTC por NTP) ---
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

// --- 5. PULSADOR ---
// Pulsador entre PIN_BOTON y GND (se usa la resistencia pull-up interna).
// Alternativa sin cableado: usar el botón BOOT de la DevKit (GPIO 0).
#define PIN_BOTON              32
#define DEBOUNCE_MS            40
#define PULSACION_LARGA_MS     800
#define TIMEOUT_PANTALLA_MS    60000UL

// --- 6. PARÁMETROS DEL MODELO ---
// INTERVALO_MIN : minutos entre muestras del modelo (el sensor se sigue leyendo
//                 cada 500 ms para la pantalla, pero al modelo y a la SD solo
//                 entra una muestra cada INTERVALO_MIN minutos, alineada al
//                 reloj: hh:00, hh:05, hh:10, ...).
#define INTERVALO_MIN    5
#define MUESTRAS_DIA     (24 * 60 / INTERVALO_MIN)     // 288 muestras = 24 h
// N_VALORES     : valores en memoria por variable = ventana de ajuste (24 h).
// N_MIN_AJUSTE  : valores mínimos para estimar el AR(p). Se espera un ciclo
//                 completo de 24 h (288 muestras) antes de estimar el modelo y
//                 empezar a pronosticar el siguiente ciclo.
// MAX_HUECO_MUESTRAS: huecos <= este número de muestras se rellenan con el
//                 último valor; si son mayores se reinicia la serie.
#define N_VALORES          MUESTRAS_DIA
#define N_MIN_AJUSTE       MUESTRAS_DIA                // 288 muestras = 24 h
#define MAX_HUECO_MUESTRAS 3                           // 15 min
#define PRECARGAR_SD       true

// ORDEN DEL MODELO
// P_MAX        : orden máximo evaluado en la selección de p (cuaderno: 10).
//                La tabla de la pantalla "Orden" admite hasta 10.
// P_FIJO       : 0 -> el orden se elige automáticamente en cada ajuste.
//                1..P_MAX -> se usa siempre ese orden (1 = AR(1) del sketch anterior).
// CRITERIO_BIC : false -> se elige el p de menor AIC; true -> el de menor BIC
//                (más parsimonioso). Solo aplica si P_FIJO = 0.
#define P_MAX         10
#define P_FIJO        0
#define CRITERIO_BIC  false

static_assert(P_MAX >= 1 && P_MAX <= 10, "P_MAX debe estar entre 1 y 10");
static_assert(P_FIJO >= 0 && P_FIJO <= P_MAX, "P_FIJO debe estar entre 0 y P_MAX");
static_assert(N_MIN_AJUSTE > 3 * P_MAX, "Serie demasiado corta para evaluar P_MAX");
static_assert(N_MIN_AJUSTE <= N_VALORES, "N_MIN_AJUSTE no puede superar N_VALORES");

// Gráficas: GRAF_HIST muestras de historia (8 h) + GRAF_FUT de pronóstico (~2.7 h) = 128 px
#define GRAF_HIST  96
#define GRAF_FUT   (SCREEN_WIDTH - GRAF_HIST)

// Declaración adelantada: el IDE de Arduino inserta los prototipos de las
// funciones antes de la primera función, que está antes de ModeloARp.
struct ModeloARp;

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

  // La serie en orden cronológico como vector de Eigen
  VectorXd vector() const {
    VectorXd v(count);
    for (int i = 0; i < count; i++) v(i) = get(i);
    return v;
  }
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
// ÁLGEBRA LINEAL (Eigen)
// ============================================================

/*
 * matrizRezagos
 * -------------
 * Matriz de diseño de la regresión  x(t) = c + phi_1 x(t-1) + ... + phi_k x(t-k)
 * con renglones t = inicio .. N-1:
 *     X = [ 1, x(t-1), x(t-2), ..., x(t-k) ]       (n x (k+1)),  n = N - inicio
 * La columna i es la serie desplazada i lugares: x.segment(inicio - i, n).
 */
MatrixXd matrizRezagos(const VectorXd& x, int k, int inicio) {
  const int n = x.size() - inicio;
  MatrixXd X(n, k + 1);
  X.col(0).setOnes();
  for (int i = 1; i <= k; i++) X.col(i) = x.segment(inicio - i, n);
  return X;
}

/*
 * minimosCuadrados
 * ----------------
 * beta = argmin ||y - X beta||^2 por QR con pivoteo de columnas.
 * Devuelve false si X no tiene rango completo (rezagos casi colineales o
 * serie constante), equivalente al chequeo cond(Sxx) del cuaderno.
 */
bool minimosCuadrados(const MatrixXd& X, const VectorXd& y, VectorXd& beta) {
  Eigen::ColPivHouseholderQR<MatrixXd> qr;
  qr.setThreshold(1e-10);
  qr.compute(X);
  if (qr.rank() < X.cols()) return false;
  beta = qr.solve(y);
  return true;
}

/*
 * criteriosInformacion
 * --------------------
 * Verosimilitud gaussiana con sigma2 = SSE/n (cuaderno CRITERIOS_AIC_BIC):
 *   -2 ln L = n [ln(2 pi SSE/n) + 1]
 *   AIC = -2 ln L + 2k        BIC = -2 ln L + k ln(n)
 */
void criteriosInformacion(double sse, int n, int k, double& aic, double& bic) {
  const double menos2LogL = n * (log(2.0 * M_PI * sse / n) + 1.0);
  aic = menos2LogL + 2.0 * k;
  bic = menos2LogL + k * log((double)n);
}

/*
 * raicesCaracteristicas
 * ---------------------
 * Raíces de lambda^p - phi_1 lambda^(p-1) - ... - phi_p = 0 como valores
 * propios de la matriz compañera:
 *         | phi_1 phi_2 ... phi_p |
 *     C = |   1     0   ...   0   |
 *         |   0     1   ...   0   |
 *         |   0     0   ... 1 0   |
 */
VectorXcd raicesCaracteristicas(const VectorXd& phi) {
  const int p = phi.size();
  MatrixXd C = MatrixXd::Zero(p, p);
  C.row(0) = phi.transpose();
  if (p > 1) C.diagonal(-1).setOnes();
  return Eigen::EigenSolver<MatrixXd>(C, false).eigenvalues();
}

/*
 * pacfMuestral
 * ------------
 * ACF muestral r(k) (dividida entre N, igual que el cuaderno) y PACF:
 * phi_kk es el último coeficiente de las ecuaciones de Yule-Walker de orden k
 *     R_k phi = (r(1), ..., r(k)),   R_k(i,j) = r(|i-j|)   (Toeplitz)
 * Devuelve false si la serie es constante.
 */
bool pacfMuestral(const VectorXd& x, int kMax, VectorXd& pacf) {
  const int N = x.size();
  const VectorXd xc = x.array() - x.mean();
  const double c0 = xc.squaredNorm() / N;
  if (c0 <= 1e-12) return false;

  VectorXd r(kMax + 1);                                   // autocorrelaciones
  for (int k = 0; k <= kMax; k++) r(k) = xc.head(N - k).dot(xc.tail(N - k)) / N / c0;

  pacf(0) = 1.0;
  for (int k = 1; k <= kMax; k++) {
    MatrixXd R(k, k);
    for (int i = 0; i < k; i++)
      for (int j = 0; j < k; j++) R(i, j) = r(i > j ? i - j : j - i);
    VectorXd phiK = R.ldlt().solve(r.segment(1, k));
    pacf(k) = phiK(k - 1);
  }
  return true;
}

// ============================================================
// MODELO AR(p) DE UNA VARIABLE (temperatura o humedad)
// ============================================================
struct ModeloARp {
  const char* nombre;
  const char* letra;     // "T" / "H"
  const char* unidad;
  float lo, hi;          // límites físicos del pronóstico

  Serie serie;

  // --- Selección del orden (AR(1)..AR(P_MAX) sobre la misma muestra) ---
  bool     selOk = false;
  int      nSel = 0;                // filas comunes = N - P_MAX
  VectorXd aic, bic, rmseSel;       // índice = p (0..P_MAX)
  int      pAic = 0, pBic = 0;
  VectorXd pacf;                    // PACF muestral, pacf(0) = 1
  float    banda = NAN;             // 1.96 / sqrt(N)
  int      pPacf = 0;               // último retardo significativo (<= P_MAX)

  // --- Modelo estimado con el orden elegido ---
  bool      ok = false;
  int       p = 0;                  // orden en uso
  int       nFilas = 0;             // n = N - p
  double    c = NAN;
  VectorXd  phi;                    // phi(0) = phi_1, ..., phi(p-1) = phi_p
  double    sigma2 = NAN;           // SSE / (n - p - 1)
  double    sumaPhi = NAN, radioMax = NAN;
  VectorXcd raices;
  double    rmsePers = NAN;         // RMSE del modelo de persistencia X(t) ~ X(t-1)
  Metricas  ajuste;                 // residuales dentro de la ventana (in-sample)

  // --- Evaluación fuera de muestra a 1 paso ---
  float    predSig  = NAN;   // pronóstico para la próxima muestra
  float    predEval = NAN;   // pronóstico que se evaluó en la última muestra
  int      pEval    = 0;     // orden del modelo que hizo predEval
  Metricas unPaso;

  // --- Historia corta para la gráfica ---
  float histReal[GRAF_HIST];
  float histPred[GRAF_HIST];
  int   histHead = 0, histCount = 0;

  ModeloARp(const char* n, const char* l, const char* u, float lo_, float hi_)
    : nombre(n), letra(l), unidad(u), lo(lo_), hi(hi_) {
    aic = bic = rmseSel = pacf = VectorXd::Constant(P_MAX + 1, NAN);
  }

  float media() const {                       // mu = c / (1 - sum phi)
    if (!estacionario() || fabs(1.0 - sumaPhi) < 1e-12) return NAN;
    return (float)(c / (1.0 - sumaPhi));
  }
  bool estacionario() const { return ok && radioMax < 1.0; }
  float mejoraPers() const {                  // % de mejora del RMSE vs persistencia
    if (!ok || !(rmsePers > 0)) return NAN;
    return (float)(100.0 * (1.0 - ajuste.rmse() / rmsePers));
  }

  // ---------- Selección de p: AIC, BIC y PACF ----------
  void seleccionar(const VectorXd& x) {
    selOk = false;
    pAic = pBic = pPacf = 0;
    aic.setConstant(NAN);
    bic.setConstant(NAN);
    rmseSel.setConstant(NAN);
    pacf.setConstant(NAN);

    // Todos los órdenes usan las mismas filas t = P_MAX .. N-1
    const MatrixXd X = matrizRezagos(x, P_MAX, P_MAX);
    const VectorXd y = x.tail(X.rows());
    nSel = y.size();

    for (int q = 1; q <= P_MAX; q++) {
      const MatrixXd Xq = X.leftCols(q + 1);       // [1, x(t-1) .. x(t-q)]
      VectorXd beta;
      if (!minimosCuadrados(Xq, y, beta)) continue;
      double sse = fmax((y - Xq * beta).squaredNorm(), 1e-12);
      criteriosInformacion(sse, nSel, q + 2, aic(q), bic(q));
      rmseSel(q) = sqrt(sse / nSel);
      if (pAic == 0 || aic(q) < aic(pAic)) pAic = q;
      if (pBic == 0 || bic(q) < bic(pBic)) pBic = q;
    }
    selOk = (pAic > 0);

    // PACF con bandas de significancia del 95 %
    banda = 1.96f / sqrtf((float)x.size());
    if (!pacfMuestral(x, P_MAX, pacf)) return;
    for (int k = 1; k <= P_MAX; k++)
      if (fabs(pacf(k)) > banda) pPacf = k;
  }

  // ---------- Estimación por MCO con el orden elegido ----------
  void estimar() {
    ok = false;
    nFilas = 0;
    if (serie.count < N_MIN_AJUSTE) return;

    const VectorXd x = serie.vector();
    seleccionar(x);
    int q = (P_FIJO > 0) ? P_FIJO : (CRITERIO_BIC ? pBic : pAic);
    if (q < 1) return;

    // Regresión con n = N - q filas:  y = X beta,  beta = (c, phi_1..phi_q)
    const MatrixXd X = matrizRezagos(x, q, q);
    const VectorXd y = x.tail(X.rows());
    VectorXd beta;
    if (!minimosCuadrados(X, y, beta)) return;

    p       = q;
    nFilas  = y.size();
    c       = beta(0);
    phi     = beta.tail(p);
    sumaPhi = phi.sum();

    // Residuales del ajuste: RMSE, MAE, MAPE, R2
    const VectorXd yAjustada = X * beta;
    ajuste.limpiar();
    for (int i = 0; i < nFilas; i++) ajuste.agregar(y(i), yAjustada(i));
    sigma2 = (nFilas > p + 1) ? (y - yAjustada).squaredNorm() / (nFilas - p - 1) : NAN;

    // Persistencia: X(t) ~ X(t-1), que es la columna 1 de la matriz de diseño
    rmsePers = sqrt((y - X.col(1)).squaredNorm() / nFilas);

    // Estacionariedad: |lambda| < 1 para todas las raíces
    raices   = raicesCaracteristicas(phi);
    radioMax = raices.cwiseAbs().maxCoeff();
    ok = true;
  }

  // Pronóstico recursivo: out[h-1] = X^(t+h), h = 1..H
  void trayectoria(float* out, int H) const {
    if (!ok || serie.count < p) { for (int h = 0; h < H; h++) out[h] = NAN; return; }
    VectorXd w(p);                                    // w(i) = X(t-i)
    for (int i = 0; i < p; i++) w(i) = serie.get(serie.count - 1 - i);
    for (int h = 0; h < H; h++) {
      double x = c + phi.dot(w);
      if (p > 1) w.tail(p - 1) = w.head(p - 1).eval();   // desplaza los rezagos
      w(0) = x;
      out[h] = fminf(fmaxf((float)x, lo), hi);     // límites físicos
    }
  }

  float pronostico1() const { float f; trayectoria(&f, 1); return f; }

  float histRealAt(int i) const { return histReal[(histHead - histCount + i + 2 * GRAF_HIST) % GRAF_HIST]; }
  float histPredAt(int i) const { return histPred[(histHead - histCount + i + 2 * GRAF_HIST) % GRAF_HIST]; }

  // Registra una lectura real. Si evaluar = true se acumula el error de la
  // predicción hecha para esta muestra.
  void actualizar(float x, bool evaluar) {
    predEval = evaluar ? predSig : NAN;
    pEval    = ok ? p : 0;
    if (!isnan(predEval)) unPaso.agregar(x, predEval);

    histReal[histHead] = x;
    histPred[histHead] = predEval;
    histHead = (histHead + 1) % GRAF_HIST;
    if (histCount < GRAF_HIST) histCount++;

    serie.push(x);
    estimar();
    predSig = pronostico1();
  }

  // Hubo un hueco largo: la serie deja de ser equiespaciada y se reinicia.
  // Las métricas a 1 paso acumuladas se conservan.
  void reiniciar() {
    serie.limpiar();
    histHead = histCount = 0;
    ok = selOk = false;
    p = 0;
    predSig = predEval = NAN;
  }
};

ModeloARp arT("Temperatura", "T", "C", -INFINITY, INFINITY);
ModeloARp arH("Humedad", "H", "%", 0.0f, 100.0f);

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
    if (salto < 0 || salto > MAX_HUECO_MUESTRAS) {      // reloj hacia atrás o hueco largo
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
 *   Fecha,Hora,T_orig,H_orig,T_corr,H_corr[,pred...]   (sketches GM / AR / este)
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
// 0 = principal; 1..6 = temperatura; 7..12 = humedad
#define SUBPANTALLAS   6       // modelo, diagnóstico, errores, gráfica, orden, PACF
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

const char* nombreCriterio() {
  if (P_FIJO > 0) return "fijo";
  return CRITERIO_BIC ? "BIC" : "AIC";
}

// Mensaje común cuando el modelo aún no está estimado
bool avisoSinModelo(const ModeloARp& m) {
  if (m.ok) return false;
  if (m.serie.count < N_MIN_AJUSTE) {
    char f[12];
    fmtFaltan(f, sizeof(f), m.serie.count);
    display.printf("\nCiclo de 24h: faltan\n%s\n", f);
  }
  else display.println("\nSxx singular (rezagos\ncolineales o serie\nconstante)");
  display.printf("\nHist: %d/%d", m.serie.count, N_VALORES);
  return true;
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
  if (arT.ok && arH.ok) display.printf("ARp  T:%.1fC H:%.0f%%", arT.predSig, arH.predSig);
  else {
    char f[12];
    fmtFaltan(f, sizeof(f), arT.serie.count);
    display.printf("ARp: faltan %s", f);
  }

  display.setCursor(0, 54);
  display.print(sdOK ? "SD:OK" : "SD:ERR");
  if (arT.ok && arH.ok) display.printf("  p: T=%d H=%d", arT.p, arH.p);
  else                  display.print("  [BTN] AR(p)");
}

// Coeficientes del modelo
void dibujarModelo(const ModeloARp& m) {
  char s[16];
  display.setTextSize(1);
  display.setCursor(0, 0);
  if (!m.ok) { display.printf("AR(p) %s\n", m.nombre); avisoSinModelo(m); return; }

  display.printf("AR(%d) %s\n", m.p, m.nombre);
  display.printf("c = %.6f\n", m.c);
  for (int i = 0; i < m.p; i++) {
    display.printf("%d:%+.4f", i + 1, m.phi(i));
    display.print((i % 2 == 0 && i + 1 < m.p) ? " " : "\n");
  }
  fmt(s, sizeof(s), m.predSig, 2);
  display.printf("Pred +%dm: %s %s\n", INTERVALO_MIN, s, m.unidad);
  if (display.getCursorY() <= SCREEN_HEIGHT - 8)
    display.printf("Orden: %s (max %d)", nombreCriterio(), P_MAX);
}

// Estacionariedad, media, varianza y comparación con persistencia
void dibujarDiagnostico(const ModeloARp& m) {
  char s[16];
  display.setTextSize(1);
  display.setCursor(0, 0);
  if (m.ok) display.printf("AR(%d) %s diagnostico\n", m.p, m.letra);
  else      display.printf("AR(p) %s diagnostico\n", m.letra);
  if (avisoSinModelo(m)) return;
  display.printf("Suma phi = %.5f\n", m.sumaPhi);
  display.printf("max|lam| = %.5f\n", m.radioMax);
  display.printf("Estacion.: %s\n", m.estacionario() ? "SI |lam|<1" : "NO");
  fmt(s, sizeof(s), m.media(), 2);
  display.printf("mu = %s %s\n", s, m.unidad);
  display.printf("sigma2 = %.6f\n", m.sigma2);
  display.printf("Filas n = %d\n", m.nFilas);
  fmt(s, sizeof(s), m.mejoraPers(), 1);
  display.printf("vs persist: %s%%", s);
}

void dibujarErrores(const ModeloARp& m) {
  display.setTextSize(1);
  display.setCursor(0, 0);
  if (m.ok) display.printf("AR(%d) %s  errores\n", m.p, m.letra);
  else      display.printf("AR(p) %s  errores\n", m.letra);
  display.println("       Ajuste  1 paso");
  bool a = m.ok;
  filaTabla("MAE",   a ? m.ajuste.mae()  : NAN, m.unPaso.mae(),  4);
  filaTabla("RMSE",  a ? m.ajuste.rmse() : NAN, m.unPaso.rmse(), 4);
  filaTabla("MAPE%", a ? m.ajuste.mape() : NAN, m.unPaso.mape(), 3);
  filaTabla("R2",    a ? m.ajuste.r2()   : NAN, m.unPaso.r2(),   4);
  display.printf("%-5s%8lu%8lu\n", "n", a ? (unsigned long)m.ajuste.n : 0UL, (unsigned long)m.unPaso.n);
  display.print("Ajuste=ventana N");
}

void dibujarGrafica(const ModeloARp& m) {
  const int Y0 = 10, Y1 = SCREEN_HEIGHT - 1;
  const int nh = m.histCount;

  float fut[GRAF_FUT];
  m.trayectoria(fut, GRAF_FUT);

  // Escala vertical automática
  float mn = INFINITY, mx = -INFINITY;
  auto incluir = [&](float y) { if (!isnan(y)) { if (y < mn) mn = y; if (y > mx) mx = y; } };
  for (int i = 0; i < nh; i++) { incluir(m.histRealAt(i)); incluir(m.histPredAt(i)); }
  for (int h = 0; h < GRAF_FUT; h++) incluir(fut[h]);

  display.setTextSize(1);
  display.setCursor(0, 0);
  if (isinf(mn)) { display.printf("AR %s sin datos", m.letra); return; }
  display.printf("AR%d %s %.1f-%.1f%s", m.p, m.letra, mn, mx, m.unidad);
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
    float f = fut[h - 1];
    if (!isnan(prev) && !isnan(f) && ((h / 2) % 2 == 0))
      display.drawLine(GRAF_HIST - 2 + h, mapY(prev), GRAF_HIST - 1 + h, mapY(f), SSD1306_WHITE);
    prev = f;
  }
}

// Tabla de selección del orden: AIC(p) - AIC_min, en dos columnas.
// '*' = mínimo AIC, 'b' = mínimo BIC, '<' = orden en uso (si difiere).
void dibujarOrden(const ModeloARp& m) {
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.printf("Orden %s AIC:%d BIC:%d\n", m.letra, m.pAic, m.pBic);
  if (!m.selOk) { avisoSinModelo(m); return; }
  display.println(" p   dAIC   p   dAIC");

  const double aMin = m.aic(m.pAic);
  const int filas = (P_MAX + 1) / 2;
  for (int r = 0; r < filas; r++) {
    for (int col = 0; col < 2; col++) {
      int q = r + 1 + col * filas;
      if (q > P_MAX) break;
      char s[10];
      double d = m.aic(q) - aMin;
      if (isnan(d)) snprintf(s, sizeof(s), "--");
      else if (d > 9999.0) snprintf(s, sizeof(s), ">9999");
      else snprintf(s, sizeof(s), "%.1f", d);
      char marca = ' ';
      if (q == m.pAic) marca = '*';
      else if (q == m.pBic) marca = 'b';
      else if (m.ok && q == m.p) marca = '<';
      display.printf("%2d%7s%c", q, s, marca);
      if (col == 0) display.print(" ");
    }
    display.print("\n");
  }
  display.setCursor(0, SCREEN_HEIGHT - 8);
  display.printf("n=%d uso:%s p=%d", m.nSel, nombreCriterio(), m.ok ? m.p : 0);
}

// PACF muestral (barras) con bandas +-1.96/sqrt(N): identifica el orden p
void dibujarPACF(const ModeloARp& m) {
  const int Y0 = 10, Y1 = SCREEN_HEIGHT - 1;
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.printf("PACF %s  p_pacf=%d", m.letra, m.pPacf);
  if (!m.selOk) { avisoSinModelo(m); return; }

  auto mapY = [&](float r) -> int {
    if (r > 1) r = 1;
    if (r < -1) r = -1;
    return Y0 + (int)lroundf((1.0f - r) * 0.5f * (Y1 - Y0));
  };
  const int y0 = mapY(0), yb1 = mapY(m.banda), yb2 = mapY(-m.banda);
  display.drawFastHLine(0, y0, SCREEN_WIDTH, SSD1306_WHITE);
  for (int x = 0; x < SCREEN_WIDTH; x += 4) {
    display.drawPixel(x, yb1, SSD1306_WHITE);
    display.drawPixel(x, yb2, SSD1306_WHITE);
  }

  const int ancho = SCREEN_WIDTH / P_MAX;
  for (int k = 1; k <= P_MAX; k++) {
    if (isnan(m.pacf(k))) continue;
    int x = (k - 1) * ancho + 1;
    int w = ancho - 3 > 1 ? ancho - 3 : 1;
    int y = mapY((float)m.pacf(k));
    if (y < y0) display.fillRect(x, y, w, y0 - y + 1, SSD1306_WHITE);
    else        display.drawRect(x, y0, w, y - y0 + 1, SSD1306_WHITE);
  }
}

void dibujarPantalla(const RtcDateTime& now, float temperatura, float humedad) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  if (pantalla == 0) {
    dibujarPrincipal(now, temperatura, humedad);
  } else {
    const ModeloARp& m = ((pantalla - 1) / SUBPANTALLAS == 0) ? arT : arH;
    switch ((pantalla - 1) % SUBPANTALLAS) {
      case 0: dibujarModelo(m);      break;
      case 1: dibujarDiagnostico(m); break;
      case 2: dibujarErrores(m);     break;
      case 3: dibujarGrafica(m);     break;
      case 4: dibujarOrden(m);       break;
      case 5: dibujarPACF(m);        break;
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
void imprimirBloque(const ModeloARp& m) {
  Serial.printf("--- %s (real = %.2f %s) ---\n", m.nombre, m.serie.ultimo(), m.unidad);
  if (!m.ok) { Serial.printf("Modelo no estimado (hist %d/%d)\n", m.serie.count, N_MIN_AJUSTE); return; }

  // Selección del orden
  Serial.printf("Seleccion (n=%d): p_AIC=%d  p_BIC=%d  p_PACF=%d  -> AR(%d) [%s]\n",
                m.nSel, m.pAic, m.pBic, m.pPacf, m.p, nombreCriterio());
  Serial.println("  p        AIC          BIC        RMSE     PACF");
  for (int q = 1; q <= P_MAX; q++)
    Serial.printf("  %-2d %12.4f %12.4f %11.6f %+8.4f%s\n", q, m.aic(q), m.bic(q), m.rmseSel(q), m.pacf(q),
                  q == m.p ? "  <" : "");
  Serial.printf("  Banda PACF 95%% = +-%.5f\n", m.banda);

  // Modelo estimado
  Serial.printf("Numero de filas  = %d\n", m.nFilas);
  Serial.printf("Intercepto c     = %.8f\n", m.c);
  for (int i = 0; i < m.p; i++) Serial.printf("phi_%-2d           = %.8f\n", i + 1, m.phi(i));
  Serial.printf("Modelo: %s(t) = %.8f", m.letra, m.c);
  for (int i = 0; i < m.p; i++) Serial.printf(" + (%.8f)*%s(t-%d)", m.phi(i), m.letra, i + 1);
  Serial.println();

  // Estacionariedad
  Serial.printf("Suma phi = %.8f   max|lambda| = %.8f  (%s)\n", m.sumaPhi, m.radioMax,
                m.estacionario() ? "estacionario" : "NO estacionario");
  Serial.print("Raices lambda:");
  for (int i = 0; i < m.p; i++) {
    double re = m.raices(i).real(), im = m.raices(i).imag();
    if (fabs(im) < 1e-9) Serial.printf(" %.6f", re);
    else Serial.printf(" %.6f%+.6fi", re, im);
  }
  Serial.println();
  Serial.printf("mu = c/(1-sum phi) = %.6f   sigma2 = %.8f\n", m.media(), m.sigma2);

  // Errores
  Serial.printf("Ajuste   RMSE=%.6f MAE=%.6f R2=%.6f  (persistencia RMSE=%.6f, mejora %.2f%%)\n",
                m.ajuste.rmse(), m.ajuste.mae(), m.ajuste.r2(), m.rmsePers, m.mejoraPers());
  Serial.printf("1 paso   pred usada=%.4f (AR%d)  RMSE=%.6f MAE=%.6f MAPE=%.4f%% R2=%.6f n=%lu\n",
                m.predEval, m.pEval, m.unPaso.rmse(), m.unPaso.mae(), m.unPaso.mape(), m.unPaso.r2(),
                (unsigned long)m.unPaso.n);
  Serial.printf("Pronostico t+1   = %.4f %s  (+%d min)\n", m.predSig, m.unidad, INTERVALO_MIN);
}

void imprimirMetricas(const RtcDateTime& t) {
  Serial.println("=======================================================");
  Serial.printf("AR(p) | %02d/%02d/%04d %02d:%02d | cada %d min | hist %d/%d\n",
                t.Day(), t.Month(), t.Year(), t.Hour(), t.Minute(), INTERVALO_MIN, arT.serie.count, N_VALORES);
  imprimirBloque(arT);
  imprimirBloque(arH);
  Serial.println("=======================================================");
}

// Columnas p, c, phi_1..phi_PMAX de una variable (vacías si no aplica)
void escribirParametros(File& f, const ModeloARp& m, bool ultimo) {
  if (m.ok) f.printf("%d,%.8f", m.p, m.c); else f.print(",");
  for (int i = 0; i < P_MAX; i++) {
    if (m.ok && i < m.p) f.printf(",%.8f", m.phi(i));
    else f.print(",");
  }
  f.print(ultimo ? "\n" : ",");
}

void guardarSD(const RtcDateTime& now, float refT, float refH, float t, float h) {
  if (!sdOK) return;

  // Mismo formato que los otros sketches; el pronóstico guardado es el que
  // el AR(p) hizo para ESTA muestra.
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
  f = SD.open("/modelo_ar_p.csv", FILE_APPEND);
  if (f) {
    if (f.size() == 0) {
      f.print("Fecha,Hora,T,H,T_pred,H_pred");
      for (const char* v : {"T", "H"}) {
        f.printf(",p_%s,c_%s", v, v);
        for (int i = 1; i <= P_MAX; i++) f.printf(",phi%d_%s", i, v);
      }
      f.println();
    }
    f.printf("%02d/%02d/%04d,%02d:%02d:%02d,%.2f,%.2f,",
             now.Day(), now.Month(), now.Year(), now.Hour(), now.Minute(), now.Second(), t, h);
    if (isnan(arT.predEval)) f.print(","); else f.printf("%.4f,", arT.predEval);
    if (isnan(arH.predEval)) f.print(","); else f.printf("%.4f,", arH.predEval);
    escribirParametros(f, arT, false);
    escribirParametros(f, arH, true);
    f.close();
  }
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  pinMode(PIN_BOTON, INPUT_PULLUP);

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
  RtcDateTime now = Rtc.GetDateTime();
  RtcDateTime fechaReferencia(2026, 10, 01, 10, 0, 0);

  if (!now.IsValid() || now < fechaReferencia) {
    if (ENABLE_SCREEN) { display.println("Buscando WiFi..."); display.display(); }
    WiFi.mode(WIFI_STA);
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
    }
    WiFi.disconnect(true);   // el radio ya no se usa: se apaga
    WiFi.mode(WIFI_OFF);
  }

  // -- PRECARGA DE LOS ÚLTIMOS N VALORES DESDE LA SD --
  if (PRECARGAR_SD && sdOK) {
    if (ENABLE_SCREEN) { display.println("Precargando SD..."); display.display(); }
    precargarDesdeSD();
  }
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
      if (arT.ok && arH.ok)
        Serial.printf("AR(%d) T: %.1f C  AR(%d) H: %.0f %% | ", arT.p, arT.predSig, arH.p, arH.predSig);
      Serial.println(sdOK ? "SD: OK" : "SD: ERROR");
    }

    // ==========================================
    // UNA MUESTRA CADA INTERVALO_MIN MINUTOS: AR(p) y SD
    // ==========================================
    if (now.IsValid() && !isnan(temperatura) && !isnan(humedad)) {
      uint32_t muestraAbs = now.TotalSeconds() / (60UL * INTERVALO_MIN);
      // 0. Evalúa la predicción hecha para esta muestra, agrega la lectura,
      //    re-selecciona p, re-estima c y phi_i y calcula el pronóstico
      //    siguiente. Devuelve false si el intervalo ya estaba registrado
      //    (p. ej. por la precarga).
      if (registrarMuestra(muestraAbs, temperatura, humedad)) {
        imprimirMetricas(now);

        // 1. TX Serial
        Serial.printf(">>> TX: %02d/%02d/%04d,%02d:%02d:%02d,%.1f,%.0f\n",
                      now.Day(), now.Month(), now.Year(),
                      now.Hour(), now.Minute(), now.Second(), temperatura, humedad);

        // 2. MicroSD
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
