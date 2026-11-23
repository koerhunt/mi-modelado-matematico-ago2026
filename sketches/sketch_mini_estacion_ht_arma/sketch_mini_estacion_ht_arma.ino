/*
 * MINI ESTACIÓN T/H - MODELO ARMA(p,q)
 * ====================================
 * Implementación en ESP32 DevKit del modelo mixto descrito en el cuaderno
 * CRITERIOS_AIC_BIC (sección 9). Combina los sketches AR(p) y MA(q):
 *
 *   X(t) = c + phi_1 X(t-1) + ... + phi_p X(t-p)
 *            + e(t) + theta_1 e(t-1) + ... + theta_q e(t-q)
 *
 * El álgebra lineal se hace con Eigen (biblioteca "ArduinoEigen" de
 * hideakitai en el Gestor de bibliotecas del IDE de Arduino). Se usa float
 * (MatrixXf / VectorXf) porque la FPU del ESP32 es de precisión simple.
 *
 *   1. Se toma una muestra cada INTERVALO_MIN = 5 minutos y se guardan en
 *      memoria los últimos N valores (288 = 24 h).
 *   2. SELECCIÓN DEL ORDEN: se ajustan todos los ARMA(p,q) con p = 0..P_MAX y
 *      q = 0..Q_MAX sobre LA MISMA muestra (se condiciona en los primeros
 *      P_MAX datos, n = N - P_MAX residuales) y se elige el de menor AIC
 *      (o BIC si CRITERIO_BIC). Verosimilitud gaussiana:
 *        -2 ln L = n [ln(2 pi SSE/n) + 1]
 *        AIC = -2 ln L + 2k,  BIC = -2 ln L + k ln(n),  k = p + q + 2
 *      (k cuenta c, phi, theta y sigma2, igual que statsmodels).
 *   3. Estimación por Suma Condicional de Cuadrados (CSS):
 *        e(t) = X(t) - c - sum phi_i X(t-i) - sum theta_j e(t-j)
 *        a) Valores iniciales por Hannan-Rissanen: AR(m) largo por mínimos
 *           cuadrados (QR); sus residuales ehat(t) estiman las innovaciones y
 *           se regresa X(t) sobre [1, X(t-1..t-p), ehat(t-1..t-q)].
 *        b) Refinamiento con Levenberg-Marquardt minimizando SSE = ||e||^2:
 *              (J'J + lambda diag(J'J)) delta = -J'e        (LDLT de Eigen)
 *   4. Diagnóstico (valores propios de matrices compañeras):
 *        - Estacionario si las raíces de lambda^p - phi_1 lambda^(p-1) - ... - phi_p
 *          cumplen |lambda| < 1.   Media  mu = c / (1 - sum phi_i).
 *        - Invertible si las raíces de lambda^q + theta_1 lambda^(q-1) + ... + theta_q
 *          cumplen |lambda| < 1.
 *   5. Pronóstico recursivo (las innovaciones futuras valen 0):
 *        X^(t+h) = c + sum phi_i X^(t+h-i) + sum_{j>=h} theta_j e(t+h-j)
 *      Los primeros q pasos usan las innovaciones recientes; después el
 *      pronóstico sigue la parte AR y tiende a mu.
 *   6. Métricas de AJUSTE (residuales CSS en la ventana, comparadas con la
 *      persistencia X(t) ~ X(t-1)) y FUERA DE MUESTRA a 1 paso.
 *
 * El primer ajuste se hace al completar un ciclo de 24 h (288 muestras); después
 * el modelo se re-estima cada REAJUSTE_MUESTRAS muestras = 1 h (mientras
 * tanto la pantalla muestra "Estimando..."). Entre ajustes solo se actualizan
 * las innovaciones e(t) con cada lectura nueva.
 *
 * PANTALLAS (pulsador en PIN_BOTON contra GND):
 *   Principal -> [Temperatura: modelo, diagnóstico, errores, gráfica, tabla AIC]
 *             -> [Humedad:     modelo, diagnóstico, errores, gráfica, tabla AIC]
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

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::MatrixXd;

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
// N_MIN_AJUSTE  : valores mínimos para el primer ajuste (un ciclo de 24 h).
// REAJUSTE_MUESTRAS: cada cuántas muestras se re-estima el modelo (12 = 1 h).
// M_AR_HR       : orden del AR largo de Hannan-Rissanen (valores iniciales).
// MAX_HUECO_MUESTRAS: huecos <= este número de muestras se rellenan con el
//                 último valor; si son mayores se reinicia la serie.
#define N_VALORES          MUESTRAS_DIA
#define N_MIN_AJUSTE       MUESTRAS_DIA                // 288 muestras = 24 h
#define REAJUSTE_MUESTRAS  12
#define M_AR_HR            10
#define MAX_ITER_LM        25
#define MAX_HUECO_MUESTRAS 3                           // 15 min
#define PRECARGAR_SD       true

// ORDEN DEL MODELO
// P_MAX, Q_MAX : órdenes máximos de la rejilla (p,q). Se ajustan
//                (P_MAX+1)(Q_MAX+1) modelos; la tabla de la pantalla admite 3 x 3.
// CRITERIO_BIC : false -> se elige el (p,q) de menor AIC; true -> el de menor BIC.
#define P_MAX         3
#define Q_MAX         3
#define CRITERIO_BIC  false

static_assert(P_MAX >= 0 && P_MAX <= 3, "P_MAX debe estar entre 0 y 3");
static_assert(Q_MAX >= 0 && Q_MAX <= 3, "Q_MAX debe estar entre 0 y 3");
static_assert(N_MIN_AJUSTE > 10 * M_AR_HR, "Serie demasiado corta para Hannan-Rissanen");
static_assert(N_MIN_AJUSTE <= N_VALORES, "N_MIN_AJUSTE no puede superar N_VALORES");

// Gráficas: GRAF_HIST muestras de historia (8 h) + GRAF_FUT de pronóstico (~2.7 h) = 128 px
#define GRAF_HIST  96
#define GRAF_FUT   (SCREEN_WIDTH - GRAF_HIST)

// Declaración adelantada: el IDE de Arduino inserta los prototipos de las
// funciones antes de la primera función, que está antes de ModeloARMA.
struct ModeloARMA;

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
  VectorXf vector() const {
    VectorXf v(count);
    for (int i = 0; i < count; i++) v(i) = get(i);
    return v;
  }

  // Los k valores más recientes: (x(t), x(t-1), ..., x(t-k+1))
  VectorXf recientes(int k) const {
    VectorXf w(k);
    for (int i = 0; i < k; i++) w(i) = get(count - 1 - i);
    return w;
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
 * Matriz de diseño con renglones t = inicio .. N-1:
 *     X = [ 1, v(t-1), v(t-2), ..., v(t-k) ]       (n x (k+1)),  n = N - inicio
 * La columna i es el vector desplazado i lugares: v.segment(inicio - i, n).
 */
MatrixXf matrizRezagos(const VectorXf& v, int k, int inicio) {
  const int n = v.size() - inicio;
  MatrixXf X(n, k + 1);
  X.col(0).setOnes();
  for (int i = 1; i <= k; i++) X.col(i) = v.segment(inicio - i, n);
  return X;
}

/*
 * matrizARMA
 * ----------
 * Regresores del ARMA(p,q) con renglones t = inicio .. N-1:
 *     X = [ 1, x(t-1) .. x(t-p), e(t-1) .. e(t-q) ]     (n x (1+p+q))
 * x y e deben tener la misma longitud N.
 */
MatrixXf matrizARMA(const VectorXf& x, const VectorXf& e, int p, int q, int inicio) {
  const int n = x.size() - inicio;
  MatrixXf X(n, 1 + p + q);
  X.leftCols(1 + p) = matrizRezagos(x, p, inicio);
  for (int j = 1; j <= q; j++) X.col(p + j) = e.segment(inicio - j, n);
  return X;
}

/*
 * minimosCuadrados
 * ----------------
 * beta = argmin ||y - X beta||^2 por QR con pivoteo de columnas.
 * Devuelve false si X no tiene rango completo.
 */
bool minimosCuadrados(const MatrixXf& X, const VectorXf& y, VectorXf& beta) {
  Eigen::ColPivHouseholderQR<MatrixXf> qr(X);
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
 * radioCompanera
 * --------------
 * max |lambda| de las raíces de  lambda^k - a_1 lambda^(k-1) - ... - a_k = 0,
 * calculadas como valores propios de la matriz compañera:
 *         | a_1 a_2 ... a_k |
 *     C = |  1   0  ...  0  |
 *         |  0   1  ... 0 0 |
 * Parte AR: a = phi.   Parte MA: a = -theta  (raíces de 1 + theta_1 B + ...).
 * Devuelve 0 si k = 0.
 */
float radioCompanera(const VectorXf& a) {
  const int k = a.size();
  if (k == 0) return 0.0f;
  MatrixXf C = MatrixXf::Zero(k, k);
  C.row(0) = a.transpose();
  if (k > 1) C.diagonal(-1).setOnes();
  return Eigen::EigenSolver<MatrixXf>(C, false).eigenvalues().cwiseAbs().maxCoeff();
}

// ============================================================
// ESTIMACIÓN ARMA(p,q) POR SUMA CONDICIONAL DE CUADRADOS
// ============================================================
// Parámetros agrupados en beta = (c, phi_1..phi_p, theta_1..theta_q).
// Todos los modelos se condicionan en los primeros P_MAX datos para que los
// AIC/BIC se calculen sobre las mismas n = N - P_MAX observaciones.

/*
 * residualesCSS
 * -------------
 * Parte AR en forma matricial:  v = x(t) - [1, x(t-1..t-p)] (c, phi)
 * Parte MA recursiva:           e(t) = v(t) - sum theta_j e(t-j),  e = 0 antes del inicio
 * Si J != nullptr calcula también el jacobiano J(t, :) = de(t)/dbeta:
 *   J(t, :) = -[1, x(t-1..t-p), e(t-1..t-q)] - sum theta_j J(t-j, :)
 * e tiene n = N - P_MAX elementos (t = P_MAX .. N-1).
 * Devuelve false si la recursión explota (modelo no invertible).
 */
bool residualesCSS(const VectorXf& x, int p, int q, const VectorXf& beta,
                   VectorXf& e, MatrixXf* J) {
  const int n = x.size() - P_MAX;
  const VectorXf theta = beta.tail(q);
  const VectorXf v = x.tail(n) - matrizRezagos(x, p, P_MAX) * beta.head(1 + p);

  e.resize(n);
  for (int t = 0; t < n; t++) {
    const int r = min(q, t);                       // rezagos MA disponibles
    // e(t-1), ..., e(t-r) son e.segment(t-r, r) en orden inverso
    e(t) = v(t) - theta.head(r).dot(e.segment(t - r, r).reverse());
    if (!isfinite(e(t)) || fabsf(e(t)) > 1e6f) return false;
  }

  if (J) {
    VectorXf eCompleto = VectorXf::Zero(x.size());  // alineado con x
    eCompleto.tail(n) = e;
    *J = -matrizARMA(x, eCompleto, p, q, P_MAX);
    for (int t = 1; t < n; t++) {
      const int r = min(q, t);
      for (int j = 1; j <= r; j++) J->row(t) -= theta(j - 1) * J->row(t - j);
    }
  }
  return true;
}

// SSE = ||e||^2 (INFINITY si el modelo no es invertible)
float sseCSS(const VectorXf& x, int p, int q, const VectorXf& beta) {
  VectorXf e;
  return residualesCSS(x, p, q, beta, e, nullptr) ? e.squaredNorm() : INFINITY;
}

/*
 * valoresInicialesHR (Hannan-Rissanen)
 * ------------------------------------
 * ehat son los residuales de un AR(m) largo (0 para t < m). Se regresa x(t)
 * sobre [1, x(t-1..t-p), ehat(t-1..t-q)] con t = max(m+q, P_MAX) .. N-1.
 */
bool valoresInicialesHR(const VectorXf& x, const VectorXf& ehat, int p, int q, int m,
                        VectorXf& beta) {
  const MatrixXf X = matrizARMA(x, ehat, p, q, max(m + q, P_MAX));
  return minimosCuadrados(X, x.tail(X.rows()), beta);
}

/*
 * ajustarCSS (Levenberg-Marquardt)
 * --------------------------------
 * Minimiza SSE(beta). En cada iteración resuelve
 *     (J'J + lambda diag(J'J)) delta = -J'e
 * y acepta el paso si reduce el SSE; si no, aumenta lambda.
 * Devuelve el SSE final (INFINITY si el punto inicial no es invertible).
 */
float ajustarCSS(const VectorXf& x, int p, int q, VectorXf& beta) {
  VectorXf e;
  MatrixXf J;
  if (!residualesCSS(x, p, q, beta, e, &J)) return INFINITY;
  float sse = e.squaredNorm();
  float lambda = 1e-3f;

  for (int it = 0; it < MAX_ITER_LM; it++) {
    const MatrixXf JtJ = J.transpose() * J;
    const VectorXf Jte = J.transpose() * e;

    bool mejoro = false;
    while (lambda < 1e8f) {
      MatrixXf A = JtJ;
      A.diagonal() *= (1.0f + lambda);
      Eigen::LDLT<MatrixXf> ldlt(A);
      if (ldlt.info() == Eigen::Success) {
        const VectorXf betaN = beta - ldlt.solve(Jte);
        const float sseN = sseCSS(x, p, q, betaN);
        if (sseN < sse) {
          mejoro = (sse - sseN) / sse > 1e-6f;
          beta = betaN;
          sse = sseN;
          lambda = fmaxf(lambda * 0.1f, 1e-9f);
          break;
        }
      }
      lambda *= 10.0f;
    }
    if (!mejoro) break;
    residualesCSS(x, p, q, beta, e, &J);           // e y J en el nuevo punto
    yield();
  }
  return sse;
}

// ============================================================
// MODELO ARMA(p,q) DE UNA VARIABLE (temperatura o humedad)
// ============================================================
struct ModeloARMA {
  const char* nombre;
  const char* letra;     // "T" / "H"
  const char* unidad;
  float lo, hi;          // límites físicos del pronóstico

  Serie serie;

  // --- Selección del orden: rejilla (p, q) ---
  MatrixXd aic, bic;                     // aic(p, q), bic(p, q)
  int      pAic = 0, qAic = 0, pBic = 0, qBic = 0;

  // --- Modelo seleccionado ---
  bool     ok = false;
  int      p = 0, q = 0;
  int      nAjuste = 0;                  // n = N - P_MAX residuales
  float    c = NAN;
  VectorXf phi;                          // phi(0) = phi_1, ..., phi(p-1) = phi_p
  VectorXf theta;                        // theta(0) = theta_1, ..., theta(q-1) = theta_q
  float    sigma2 = NAN;                 // SSE / n
  float    radioAR = NAN, radioMA = NAN; // max|lambda| de cada polinomio
  float    rmsePers = NAN;               // RMSE de la persistencia X(t) ~ X(t-1)
  Metricas ajuste;                       // residuales CSS en la ventana

  // Innovaciones recientes: eps(0) = e(t), eps(1) = e(t-1), ...
  VectorXf eps;

  // --- Evaluación fuera de muestra a 1 paso ---
  float    predSig  = NAN;
  float    predEval = NAN;
  Metricas unPaso;

  // --- Historia corta para la gráfica ---
  float histReal[GRAF_HIST];
  float histPred[GRAF_HIST];
  int   histHead = 0, histCount = 0;

  ModeloARMA(const char* n, const char* l, const char* u, float lo_, float hi_)
    : nombre(n), letra(l), unidad(u), lo(lo_), hi(hi_) {
    aic = bic = MatrixXd::Constant(P_MAX + 1, Q_MAX + 1, NAN);
    eps = VectorXf::Zero(Q_MAX);
  }

  float sumaPhi() const { return phi.sum(); }
  bool  estacionario() const { return ok && radioAR < 1.0f; }
  bool  invertible()   const { return ok && radioMA < 1.0f; }
  float media() const {                       // mu = c / (1 - sum phi)
    if (!estacionario() || fabsf(1.0f - sumaPhi()) < 1e-6f) return NAN;
    return c / (1.0f - sumaPhi());
  }
  float mejoraPers() const {                  // % de mejora del RMSE vs persistencia
    if (!ok || !(rmsePers > 0)) return NAN;
    return 100.0f * (1.0f - ajuste.rmse() / rmsePers);
  }

  // Pronóstico recursivo sin límites: out[h-1] = X^(t+h), h = 1..H
  void trayectoriaCruda(float* out, int H) const {
    if (!ok || serie.count < p) { for (int h = 0; h < H; h++) out[h] = NAN; return; }
    VectorXf w = serie.recientes(p);              // w(i)  = X(t-i)
    VectorXf u = eps.head(q);                     // u(j)  = e(t-j); futuras = 0
    for (int h = 0; h < H; h++) {
      float x = c + phi.dot(w) + theta.dot(u);
      if (p > 1) w.tail(p - 1) = w.head(p - 1).eval();   // desplaza los rezagos
      if (p > 0) w(0) = x;
      if (q > 1) u.tail(q - 1) = u.head(q - 1).eval();
      if (q > 0) u(0) = 0.0f;                     // E[e(t+h)] = 0
      out[h] = x;
    }
  }
  void trayectoria(float* out, int H) const {
    trayectoriaCruda(out, H);
    for (int h = 0; h < H; h++)
      if (!isnan(out[h])) out[h] = fminf(fmaxf(out[h], lo), hi);   // límites físicos
  }
  float pronostico1() const { float f; trayectoria(&f, 1); return f; }

  float histRealAt(int i) const { return histReal[(histHead - histCount + i + 2 * GRAF_HIST) % GRAF_HIST]; }
  float histPredAt(int i) const { return histPred[(histHead - histCount + i + 2 * GRAF_HIST) % GRAF_HIST]; }

  void actualizar(float x, bool evaluar) {
    predEval = evaluar ? predSig : NAN;
    if (!isnan(predEval)) unPaso.agregar(x, predEval);

    histReal[histHead] = x;
    histPred[histHead] = predEval;
    histHead = (histHead + 1) % GRAF_HIST;
    if (histCount < GRAF_HIST) histCount++;

    // Nueva innovación e(t) = x(t) - X_hat(t|t-1)  (antes de agregar x a la serie)
    if (ok) {
      float xHat;
      trayectoriaCruda(&xHat, 1);
      float e = x - xHat;
      if (!isfinite(e) || fabsf(e) > 1e3f) {     // recursión inestable: se reinicia
        eps.setZero();
      } else {
        eps.tail(Q_MAX - 1) = eps.head(Q_MAX - 1).eval();   // desplaza
        eps(0) = e;
      }
    }

    serie.push(x);
    predSig = pronostico1();
  }

  void reiniciar() {           // hueco largo: se conservan los parámetros
    serie.limpiar();
    histHead = histCount = 0;
    eps.setZero();
    predSig = predEval = NAN;
  }

  // Ajusta la rejilla ARMA(0..P_MAX, 0..Q_MAX) y elige (p,q) por AIC o BIC
  void ajustar() {
    const int N = serie.count;
    if (N < N_MIN_AJUSTE) return;
    const VectorXf x = serie.vector();
    const int n = N - P_MAX;                       // residuales por modelo

    // Hannan-Rissanen, paso 1: residuales de un AR(m) largo (0 para t < m)
    const int m = M_AR_HR;
    VectorXf ehat = VectorXf::Zero(N);
    VectorXf a;
    const MatrixXf Xar = matrizRezagos(x, m, m);
    const bool hrOk = minimosCuadrados(Xar, x.tail(N - m), a);
    if (hrOk) ehat.tail(N - m) = x.tail(N - m) - Xar * a;

    aic.setConstant(NAN);
    bic.setConstant(NAN);
    VectorXf betaAic, betaBic;
    float    sseAic = NAN, sseBic = NAN;
    bool     hay = false;

    for (int pp = 0; pp <= P_MAX; pp++) {
      for (int qq = 0; qq <= Q_MAX; qq++) {
        // Inicio por defecto: c = media, phi = theta = 0
        VectorXf beta = VectorXf::Zero(1 + pp + qq);
        beta(0) = x.tail(n).mean();
        // Hannan-Rissanen, paso 2 (si falla o no es invertible se usa el defecto)
        VectorXf betaHR;
        if (pp + qq > 0 && hrOk && valoresInicialesHR(x, ehat, pp, qq, m, betaHR) &&
            isfinite(sseCSS(x, pp, qq, betaHR)))
          beta = betaHR;

        float sse = ajustarCSS(x, pp, qq, beta);
        if (!isfinite(sse) || sse <= 0) continue;

        criteriosInformacion(sse, n, pp + qq + 2, aic(pp, qq), bic(pp, qq));
        if (!hay || aic(pp, qq) < aic(pAic, qAic)) { pAic = pp; qAic = qq; betaAic = beta; sseAic = sse; }
        if (!hay || bic(pp, qq) < bic(pBic, qBic)) { pBic = pp; qBic = qq; betaBic = beta; sseBic = sse; }
        hay = true;
        yield();
      }
    }
    if (!hay) return;

    // Modelo elegido
    ok = true;
    p  = CRITERIO_BIC ? pBic : pAic;
    q  = CRITERIO_BIC ? qBic : qAic;
    const VectorXf beta = CRITERIO_BIC ? betaBic : betaAic;
    const float    sse  = CRITERIO_BIC ? sseBic  : sseAic;
    c       = beta(0);
    phi     = beta.segment(1, p);
    theta   = beta.tail(q);
    nAjuste = n;
    sigma2  = sse / n;
    radioAR = radioCompanera(phi);
    radioMA = radioCompanera(-theta);

    // Residuales del modelo elegido, métricas e innovaciones actuales
    VectorXf e;
    residualesCSS(x, p, q, beta, e, nullptr);
    const VectorXf y = x.tail(n);
    ajuste.limpiar();
    for (int t = 0; t < n; t++) ajuste.agregar(y(t), y(t) - e(t));
    rmsePers = sqrtf((y - x.segment(P_MAX - 1, n)).squaredNorm() / n);
    eps.setZero();
    eps.head(q) = e.tail(q).reverse();          // eps(0) = e(t) más reciente
    predSig = pronostico1();
  }
};

ModeloARMA armaT("Temperatura", "T", "C", -INFINITY, INFINITY);
ModeloARMA armaH("Humedad", "H", "%", 0.0f, 100.0f);

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
    if (salto < 0 || salto > MAX_HUECO_MUESTRAS) {      // reloj hacia atrás o hueco largo
      Serial.printf("-> Hueco de %ld min: se reinicia la serie\n", (long)(salto * INTERVALO_MIN));
      armaT.reiniciar();
      armaH.reiniciar();
    } else {
      // Huecos cortos: se repite el último valor (sin evaluar errores)
      for (int64_t i = 1; i < salto; i++) {
        armaT.actualizar(armaT.serie.ultimo(), false);
        armaH.actualizar(armaH.serie.ultimo(), false);
        muestrasDesdeAjuste++;
      }
    }
  }
  armaT.actualizar(t, true);
  armaH.actualizar(h, true);
  muestrasDesdeAjuste++;
  ultimaMuestraAbs = muestraAbs;
  hayMuestra = true;
  return true;
}

bool necesitaAjuste() {
  if (armaT.serie.count < N_MIN_AJUSTE) return false;
  return !armaT.ok || !armaH.ok || muestrasDesdeAjuste >= REAJUSTE_MUESTRAS;
}

void ajustarModelos() {
  if (ENABLE_SCREEN) {
    display.fillRect(0, 54, SCREEN_WIDTH, 10, SSD1306_BLACK);
    display.setTextSize(1);
    display.setCursor(0, 54);
    display.print("Estimando ARMA...");
    display.display();
  }
  unsigned long t0 = millis();
  armaT.ajustar();
  armaH.ajustar();
  duracionAjusteMs = millis() - t0;
  muestrasDesdeAjuste = 0;
  Serial.printf("-> ARMA re-estimado en %lu ms: T ARMA(%d,%d), H ARMA(%d,%d)\n",
                duracionAjusteMs, armaT.p, armaT.q, armaH.p, armaH.q);
}

/*
 * precargarDesdeSD
 * ----------------
 * Lee el final de /datos_clima.csv y vuelve a pasar los últimos ~N registros
 * por registrarMuestra(). Acepta los dos formatos usados por los sketches:
 *   Fecha,Hora,T,H                                     (sketch base)
 *   Fecha,Hora,T_orig,H_orig,T_corr,H_corr[,pred...]   (sketches GM / AR / MA / este)
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
                (unsigned long)leidas, armaT.serie.count, N_VALORES);
}

// ============================================================
// PANTALLAS
// ============================================================
// 0 = principal; 1..5 = temperatura; 6..10 = humedad
#define SUBPANTALLAS   5       // modelo, diagnóstico, errores, gráfica, tabla AIC
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

const char* nombreCriterio() { return CRITERIO_BIC ? "BIC" : "AIC"; }

int minutosParaReajuste() { return max(0, REAJUSTE_MUESTRAS - muestrasDesdeAjuste) * INTERVALO_MIN; }

// Mensaje común cuando el modelo aún no está estimado
bool avisoSinModelo(const ModeloARMA& m) {
  if (m.ok) return false;
  char f[12];
  fmtFaltan(f, sizeof(f), m.serie.count);
  display.printf("\nCiclo de 24h: faltan\n%s\n", f);
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
  if (armaT.ok && armaH.ok) display.printf("ARMA T:%.1fC H:%.0f%%", armaT.predSig, armaH.predSig);
  else {
    char f[12];
    fmtFaltan(f, sizeof(f), armaT.serie.count);
    display.printf("ARMA: faltan %s", f);
  }

  display.setCursor(0, 54);
  display.print(sdOK ? "SD:OK" : "SD:ERR");
  if (armaT.ok && armaH.ok) display.printf(" T(%d,%d) H(%d,%d)", armaT.p, armaT.q, armaH.p, armaH.q);
  else                      display.print("  [BTN] ARMA");
}

// Coeficientes del modelo: phi en "f", theta en "t"
void dibujarModelo(const ModeloARMA& m) {
  char s[16];
  display.setTextSize(1);
  display.setCursor(0, 0);
  if (!m.ok) { display.printf("ARMA %s\n", m.nombre); avisoSinModelo(m); return; }

  display.printf("ARMA(%d,%d) %s\n", m.p, m.q, m.nombre);
  display.printf("c = %.5f\n", m.c);
  for (int i = 0; i < m.p; i++)
    display.printf("f%d=%+.4f%s", i + 1, m.phi(i), (i % 2 == 0 && i + 1 < m.p) ? " " : "\n");
  for (int j = 0; j < m.q; j++)
    display.printf("t%d=%+.4f%s", j + 1, m.theta(j), (j % 2 == 0 && j + 1 < m.q) ? " " : "\n");
  if (m.p + m.q == 0) display.println("(ruido blanco: X=c+e)");
  fmt(s, sizeof(s), m.predSig, 2);
  display.printf("Pred +%dm: %s %s\n", INTERVALO_MIN, s, m.unidad);
  if (display.getCursorY() <= SCREEN_HEIGHT - 8)
    display.printf("Reajuste en %d min", minutosParaReajuste());
}

// Estacionariedad, invertibilidad, media, varianza y persistencia
void dibujarDiagnostico(const ModeloARMA& m) {
  char s[16];
  display.setTextSize(1);
  display.setCursor(0, 0);
  if (m.ok) display.printf("ARMA(%d,%d) %s diagn.\n", m.p, m.q, m.letra);
  else      display.printf("ARMA %s diagnostico\n", m.letra);
  if (avisoSinModelo(m)) return;
  display.printf("AR max|lam|=%.4f\n", m.radioAR);
  display.printf("Estacion.: %s\n", m.estacionario() ? "SI |lam|<1" : "NO");
  display.printf("MA max|lam|=%.4f\n", m.radioMA);
  display.printf("Invertible: %s\n", m.invertible() ? "SI" : "NO");
  fmt(s, sizeof(s), m.media(), 2);
  display.printf("mu = %s %s\n", s, m.unidad);
  display.printf("sigma2 = %.6f\n", m.sigma2);
  fmt(s, sizeof(s), m.mejoraPers(), 1);
  display.printf("vs persist: %s%%", s);
}

void dibujarErrores(const ModeloARMA& m) {
  display.setTextSize(1);
  display.setCursor(0, 0);
  if (m.ok) display.printf("ARMA(%d,%d) %s errores\n", m.p, m.q, m.letra);
  else      display.printf("ARMA %s  errores\n", m.letra);
  display.println("       Ajuste  1 paso");
  bool a = m.ok;
  filaTabla("MAE",   a ? m.ajuste.mae()  : NAN, m.unPaso.mae(),  4);
  filaTabla("RMSE",  a ? m.ajuste.rmse() : NAN, m.unPaso.rmse(), 4);
  filaTabla("MAPE%", a ? m.ajuste.mape() : NAN, m.unPaso.mape(), 3);
  filaTabla("R2",    a ? m.ajuste.r2()   : NAN, m.unPaso.r2(),   4);
  display.printf("%-5s%8lu%8lu\n", "n", a ? (unsigned long)m.ajuste.n : 0UL, (unsigned long)m.unPaso.n);
  display.print("Ajuste=ventana N");
}

void dibujarGrafica(const ModeloARMA& m) {
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
  if (isinf(mn)) { display.printf("ARMA %s sin datos", m.letra); return; }
  display.printf("ARMA%d%d %s %.1f-%.1f%s", m.p, m.q, m.letra, mn, mx, m.unidad);
  if (mx - mn < 0.5f) { float c = 0.5f * (mx + mn); mn = c - 0.25f; mx = c + 0.25f; }

  auto mapY = [&](float y) -> int { return Y1 - (int)lroundf((y - mn) / (mx - mn) * (Y1 - Y0)); };

  // Línea vertical punteada = "ahora"
  for (int y = Y0; y <= Y1; y += 3) display.drawPixel(GRAF_HIST - 1, y, SSD1306_WHITE);

  // Media del proceso (línea punteada horizontal a la derecha), si es estacionario
  float mu = m.media();
  if (!isnan(mu) && mu >= mn && mu <= mx)
    for (int x = GRAF_HIST; x < SCREEN_WIDTH; x += 4) display.drawPixel(x, mapY(mu), SSD1306_WHITE);

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

// Tabla de selección del orden: AIC(p,q) - AIC_min en una rejilla p x q.
// '*' = mínimo AIC, 'b' = mínimo BIC.
void dibujarTablaAIC(const ModeloARMA& m) {
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.printf("%s *AIC(%d,%d) bBIC(%d,%d)\n", m.letra, m.pAic, m.qAic, m.pBic, m.qBic);
  if (avisoSinModelo(m)) return;

  display.print("q");                          // renglones = p, columnas = q
  for (int qq = 0; qq <= Q_MAX; qq++) display.printf("%4d ", qq);
  display.print("\n");
  const double aMin = m.aic(m.pAic, m.qAic);
  for (int pp = 0; pp <= P_MAX; pp++) {
    display.printf("%d", pp);
    for (int qq = 0; qq <= Q_MAX; qq++) {
      char s[8];
      double d = m.aic(pp, qq) - aMin;
      if (isnan(d))        snprintf(s, sizeof(s), "--");
      else if (d < 10.0)   snprintf(s, sizeof(s), "%.1f", d);
      else if (d < 1000.0) snprintf(s, sizeof(s), "%.0f", d);
      else                 snprintf(s, sizeof(s), ">999");
      char marca = ' ';
      if (pp == m.pAic && qq == m.qAic)      marca = '*';
      else if (pp == m.pBic && qq == m.qBic) marca = 'b';
      display.printf("%4s%c", s, marca);
    }
    display.print("\n");
  }
  display.setCursor(0, SCREEN_HEIGHT - 8);
  display.printf("n=%d uso:%s (%d,%d)", m.nAjuste, nombreCriterio(), m.p, m.q);
}

void dibujarPantalla(const RtcDateTime& now, float temperatura, float humedad) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  if (pantalla == 0) {
    dibujarPrincipal(now, temperatura, humedad);
  } else {
    const ModeloARMA& m = ((pantalla - 1) / SUBPANTALLAS == 0) ? armaT : armaH;
    switch ((pantalla - 1) % SUBPANTALLAS) {
      case 0: dibujarModelo(m);      break;
      case 1: dibujarDiagnostico(m); break;
      case 2: dibujarErrores(m);     break;
      case 3: dibujarGrafica(m);     break;
      case 4: dibujarTablaAIC(m);    break;
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
void imprimirBloque(const ModeloARMA& m) {
  Serial.printf("--- %s (real = %.2f %s) ---\n", m.nombre, m.serie.ultimo(), m.unidad);
  if (!m.ok) { Serial.printf("Modelo no estimado (hist %d/%d)\n", m.serie.count, N_MIN_AJUSTE); return; }

  // Selección del orden
  Serial.printf("Seleccion (n=%d): AIC -> (%d,%d)  BIC -> (%d,%d)  -> ARMA(%d,%d) [%s]\n",
                m.nAjuste, m.pAic, m.qAic, m.pBic, m.qBic, m.p, m.q, nombreCriterio());
  Serial.print("  AIC   p\\q");
  for (int qq = 0; qq <= Q_MAX; qq++) Serial.printf("%12d", qq);
  Serial.println();
  for (int pp = 0; pp <= P_MAX; pp++) {
    Serial.printf("        %d  ", pp);
    for (int qq = 0; qq <= Q_MAX; qq++) Serial.printf("%12.3f", m.aic(pp, qq));
    Serial.println();
  }

  // Modelo estimado
  Serial.printf("Intercepto c     = %.6f   (reajuste en %d min)\n", m.c, minutosParaReajuste());
  for (int i = 0; i < m.p; i++) Serial.printf("phi_%d            = %.6f\n", i + 1, m.phi(i));
  for (int j = 0; j < m.q; j++) Serial.printf("theta_%d          = %.6f\n", j + 1, m.theta(j));
  Serial.printf("sigma2           = %.6f\n", m.sigma2);
  Serial.printf("AR: max|lambda| = %.6f (%s)   MA: max|lambda| = %.6f (%s)\n",
                m.radioAR, m.estacionario() ? "estacionario" : "NO estacionario",
                m.radioMA, m.invertible() ? "invertible" : "NO invertible");
  Serial.printf("mu = c/(1-sum phi) = %.6f\n", m.media());

  // Errores
  Serial.printf("Ajuste   RMSE=%.6f MAE=%.6f R2=%.6f  (persistencia RMSE=%.6f, mejora %.2f%%)\n",
                m.ajuste.rmse(), m.ajuste.mae(), m.ajuste.r2(), m.rmsePers, m.mejoraPers());
  Serial.printf("1 paso   pred usada=%.4f  RMSE=%.6f MAE=%.6f MAPE=%.4f%% R2=%.6f n=%lu\n",
                m.predEval, m.unPaso.rmse(), m.unPaso.mae(), m.unPaso.mape(), m.unPaso.r2(),
                (unsigned long)m.unPaso.n);
  float f[6];
  m.trayectoria(f, 6);
  Serial.print("Pronostico h=1..6:");
  for (int h = 0; h < 6; h++) Serial.printf(" %.3f", f[h]);
  Serial.println();
}

void imprimirMetricas(const RtcDateTime& t) {
  Serial.println("=======================================================");
  Serial.printf("ARMA(p,q) | %02d/%02d/%04d %02d:%02d | cada %d min | hist %d/%d\n",
                t.Day(), t.Month(), t.Year(), t.Hour(), t.Minute(), INTERVALO_MIN, armaT.serie.count, N_VALORES);
  imprimirBloque(armaT);
  imprimirBloque(armaH);
  Serial.println("=======================================================");
}

// Columnas p, q, c, phi_1..phi_PMAX, theta_1..theta_QMAX de una variable
void escribirParametros(File& f, const ModeloARMA& m, bool ultimo) {
  if (m.ok) f.printf("%d,%d,%.6f", m.p, m.q, m.c); else f.print(",,");
  for (int i = 0; i < P_MAX; i++) {
    if (m.ok && i < m.p) f.printf(",%.6f", m.phi(i)); else f.print(",");
  }
  for (int j = 0; j < Q_MAX; j++) {
    if (m.ok && j < m.q) f.printf(",%.6f", m.theta(j)); else f.print(",");
  }
  f.print(ultimo ? "\n" : ",");
}

void guardarSD(const RtcDateTime& now, float refT, float refH, float t, float h) {
  if (!sdOK) return;

  // Mismo formato que los otros sketches; el pronóstico guardado es el que
  // el ARMA hizo para ESTA muestra.
  File f = SD.open("/datos_clima.csv", FILE_APPEND);
  if (f) {
    f.printf("%02d/%02d/%04d,%02d:%02d:%02d,%.2f,%.2f,%.2f,%.2f,",
             now.Day(), now.Month(), now.Year(), now.Hour(), now.Minute(), now.Second(),
             refT, refH, t, h);
    if (!isnan(armaT.predEval) && !isnan(armaH.predEval)) f.printf("%.2f,%.2f\n", armaT.predEval, armaH.predEval);
    else f.print(",\n");
    f.close();
  } else {
    Serial.println("-> Error abriendo datos_clima.csv");
  }

  // Parámetros del modelo en cada muestra
  f = SD.open("/modelo_arma.csv", FILE_APPEND);
  if (f) {
    if (f.size() == 0) {
      f.print("Fecha,Hora,T,H,T_pred,H_pred");
      for (const char* v : {"T", "H"}) {
        f.printf(",p_%s,q_%s,c_%s", v, v, v);
        for (int i = 1; i <= P_MAX; i++) f.printf(",phi%d_%s", i, v);
        for (int j = 1; j <= Q_MAX; j++) f.printf(",theta%d_%s", j, v);
      }
      f.println();
    }
    f.printf("%02d/%02d/%04d,%02d:%02d:%02d,%.2f,%.2f,",
             now.Day(), now.Month(), now.Year(), now.Hour(), now.Minute(), now.Second(), t, h);
    if (isnan(armaT.predEval)) f.print(","); else f.printf("%.4f,", armaT.predEval);
    if (isnan(armaH.predEval)) f.print(","); else f.printf("%.4f,", armaH.predEval);
    escribirParametros(f, armaT, false);
    escribirParametros(f, armaH, true);
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

  // -- PRECARGA DE LOS ÚLTIMOS N VALORES DESDE LA SD Y PRIMER AJUSTE --
  if (PRECARGAR_SD && sdOK) {
    if (ENABLE_SCREEN) { display.println("Precargando SD..."); display.display(); }
    precargarDesdeSD();
    if (necesitaAjuste()) ajustarModelos();
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
      if (armaT.ok && armaH.ok)
        Serial.printf("ARMA T: %.1f C  H: %.0f %% | ", armaT.predSig, armaH.predSig);
      Serial.println(sdOK ? "SD: OK" : "SD: ERROR");
    }

    // ==========================================
    // UNA MUESTRA CADA INTERVALO_MIN MINUTOS: ARMA(p,q) y SD
    // ==========================================
    if (now.IsValid() && !isnan(temperatura) && !isnan(humedad)) {
      uint32_t muestraAbs = now.TotalSeconds() / (60UL * INTERVALO_MIN);
      // 0. Evalúa la predicción hecha para esta muestra, actualiza la
      //    innovación e(t) y calcula el pronóstico siguiente. Devuelve false
      //    si el intervalo ya estaba registrado (p. ej. por la precarga).
      if (registrarMuestra(muestraAbs, temperatura, humedad)) {

        // 1. Serial y SD primero (no esperan a la re-estimación)
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
