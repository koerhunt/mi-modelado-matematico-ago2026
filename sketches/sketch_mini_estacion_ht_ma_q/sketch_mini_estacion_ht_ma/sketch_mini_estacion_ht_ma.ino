/*
 * MINI ESTACIÓN T/H - MODELO DE MEDIAS MÓVILES MA(q)
 * ==================================================
 * Implementación en ESP32 DevKit del cuaderno modelo_MA:
 *
 *    X(t) = mu + e(t) + theta1 e(t-1) + ... + thetaq e(t-q)
 *
 * El álgebra lineal se hace con Eigen (biblioteca "ArduinoEigen" de
 * hideakitai en el Gestor de bibliotecas del IDE de Arduino). Se usa float
 * (MatrixXf / VectorXf) porque la FPU del ESP32 es de precisión simple.
 *
 *   1. Se toma una muestra cada INTERVALO_MIN = 5 minutos y se guardan en
 *      memoria los últimos N valores (288 = 24 h).
 *   2. Se ajustan los modelos MA(0) ... MA(Q_MAX) y se elige el orden q con
 *      menor AIC (igual que el cuaderno, Q_MAX = 5).
 *   3. Estimación: el cuaderno usa ARIMA de statsmodels (máxima verosimilitud).
 *      En el ESP32 se usa el método equivalente de Suma Condicional de
 *      Cuadrados (CSS), que es como statsmodels arranca su optimización:
 *        a) Valores iniciales por Hannan-Rissanen: se ajusta un AR(m) largo
 *           por mínimos cuadrados (QR), sus residuales estiman las
 *           innovaciones e(t) y luego se regresa X(t) sobre [1, e(t-1) ... e(t-q)].
 *        b) Refinamiento con Levenberg-Marquardt minimizando
 *              SSE = ||e||^2,  e(t) = X(t) - mu - sum theta_i e(t-i)
 *           paso:  (J'J + lambda diag(J'J)) delta = -J'e     (LDLT de Eigen)
 *      Criterios (cuaderno CRITERIOS_AIC_BIC), verosimilitud gaussiana:
 *        -2 ln L = n [ln(2 pi SSE/n) + 1]
 *        AIC = -2 ln L + 2k,  BIC = -2 ln L + k ln(n),  k = q + 2 (mu, thetas, sigma2)
 *   4. Predicción (cuaderno, sección 8):
 *        X(t+h) = mu + sum_{i=h..q} theta_i e(t+h-i)     para h <= q
 *        X(t+h) = mu                                     para h >  q
 *      -> la memoria del MA(q) es de solo q pasos; luego vuelve a la media.
 *   5. Métricas de AJUSTE (residuales CSS en la ventana) y FUERA DE MUESTRA a
 *      1 paso (en cada muestra se compara la lectura nueva con la predicción).
 *
 * El primer ajuste se hace al completar un ciclo de 24 h (288 muestras); después
 * el modelo se re-estima cada REAJUSTE_MUESTRAS muestras = 1 h (mientras
 * tanto la pantalla muestra "Estimando...").
 *
 * PANTALLAS (pulsador en PIN_BOTON contra GND):
 *   Principal -> [Temperatura: modelo, errores, gráfica, tabla AIC]
 *             -> [Humedad: modelo, errores, gráfica, tabla AIC] -> Principal
 *   - Pulsación corta : siguiente pantalla (después de la última regresa a la principal).
 *   - Pulsación larga : regresa directo a la pantalla principal.
 *   - Sin pulsar durante TIMEOUT_PANTALLA_MS: regresa a la principal.
 */

#include <ArduinoEigenDense.h>   // Eigen: Core, QR, Cholesky
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
using Eigen::VectorXd;

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
// N_MIN_AJUSTE  : valores mínimos para el primer ajuste. Se espera un ciclo
//                 completo de 24 h (288 muestras) antes de estimar el modelo y
//                 empezar a pronosticar el siguiente ciclo.
// REAJUSTE_MUESTRAS: cada cuántas muestras se re-estima el modelo (12 = 1 h).
// Q_MAX         : orden máximo evaluado (cuaderno: 5).
// M_AR_HR       : orden del AR largo de Hannan-Rissanen (valores iniciales).
// MAX_HUECO_MUESTRAS: huecos <= este número de muestras se rellenan con el
//                 último valor; si son mayores se reinicia la serie.
#define N_VALORES          MUESTRAS_DIA
#define N_MIN_AJUSTE       MUESTRAS_DIA                // 288 muestras = 24 h
#define REAJUSTE_MUESTRAS  12
#define Q_MAX              5
#define M_AR_HR            10
#define MAX_ITER_LM        25
#define MAX_HUECO_MUESTRAS 3                           // 15 min
#define PRECARGAR_SD       true

// Gráficas: GRAF_HIST muestras de historia (8 h) + GRAF_FUT de pronóstico (~2.7 h) = 128 px
#define GRAF_HIST  96
#define GRAF_FUT   (SCREEN_WIDTH - GRAF_HIST)

// Declaración adelantada: el IDE de Arduino inserta los prototipos de las
// funciones antes de la primera función, que está antes de ModeloMA.
struct ModeloMA;

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

// ============================================================
// ESTIMACIÓN MA(q) POR SUMA CONDICIONAL DE CUADRADOS
// ============================================================
// Parámetros agrupados en beta = (mu, theta_1, ..., theta_q), tamaño q + 1.

/*
 * residualesCSS
 * -------------
 * e(t) = x(t) - mu - sum theta_i e(t-i), con e(t) = 0 antes del inicio.
 * Si J != nullptr calcula también el jacobiano J(t, :) = de(t)/dbeta con la
 * misma recursión del modelo:
 *   de(t)/dmu      = -1      - sum theta_i de(t-i)/dmu
 *   de(t)/dtheta_j = -e(t-j) - sum theta_i de(t-i)/dtheta_j
 * Devuelve false si la recursión explota (modelo no invertible).
 */
bool residualesCSS(const VectorXf& x, const VectorXf& beta, VectorXf& e, MatrixXf* J) {
  const int n = x.size(), q = beta.size() - 1;
  e.setZero(n);
  if (J) J->setZero(n, q + 1);

  for (int t = 0; t < n; t++) {
    const int r = min(q, t);                       // rezagos disponibles
    // e(t-1), ..., e(t-r) son e.segment(t-r, r) en orden inverso
    float pred = beta(0) + beta.segment(1, r).dot(e.segment(t - r, r).reverse());
    e(t) = x(t) - pred;
    if (!isfinite(e(t)) || fabsf(e(t)) > 1e6f) return false;

    if (J) {
      J->operator()(t, 0) = -1.0f;
      for (int j = 1; j <= r; j++) J->operator()(t, j) = -e(t - j);
      for (int i = 1; i <= r; i++) J->row(t) -= beta(i) * J->row(t - i);
    }
  }
  return true;
}

// SSE = ||e||^2 (INFINITY si el modelo no es invertible)
float sseCSS(const VectorXf& x, const VectorXf& beta) {
  VectorXf e;
  return residualesCSS(x, beta, e, nullptr) ? e.squaredNorm() : INFINITY;
}

/*
 * valoresInicialesHR (Hannan-Rissanen)
 * ------------------------------------
 * ehat son los residuales de un AR(m) largo. Se regresa x(t) sobre
 * [1, ehat(t-1) ... ehat(t-q)] con t = m+q .. n-1  ->  beta = (mu, theta).
 */
bool valoresInicialesHR(const VectorXf& x, const VectorXf& ehat, int q, int m, VectorXf& beta) {
  const MatrixXf X = matrizRezagos(ehat, q, m + q);
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
float ajustarCSS(const VectorXf& x, VectorXf& beta) {
  VectorXf e;
  MatrixXf J;
  if (!residualesCSS(x, beta, e, &J)) return INFINITY;
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
        const float sseN = sseCSS(x, betaN);
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
    residualesCSS(x, beta, e, &J);                 // e y J en el nuevo punto
    yield();
  }
  return sse;
}

// ============================================================
// MODELO MA(q) DE UNA VARIABLE (temperatura o humedad)
// ============================================================
struct ModeloMA {
  const char* nombre;
  const char* letra;     // "T" / "H"
  const char* unidad;
  float lo, hi;          // límites físicos del pronóstico

  Serie serie;

  // Modelo seleccionado
  bool     ok = false;
  int      q = 0;
  float    mu = NAN;
  VectorXf theta;                        // theta(0) = theta_1, ..., theta(q-1) = theta_q
  double   sigma2 = NAN;
  int      nAjuste = 0;
  Metricas ajuste;                       // residuales CSS en la ventana
  VectorXd aic, bic;                     // tabla de selección del orden (índice = q)

  // Innovaciones recientes: eps(0) = e(t), eps(1) = e(t-1), ...
  VectorXf eps;

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
    aic = bic = VectorXd::Constant(Q_MAX + 1, NAN);
    eps = VectorXf::Zero(Q_MAX);
  }

  // X(t+h) = mu + sum_{i=h..q} theta_i e(t+h-i)   (h > q -> mu)
  float pronosticoCrudo(int h) const {
    if (!ok) return NAN;
    if (h > q) return mu;
    const int r = q - h + 1;                     // términos theta_h .. theta_q
    return mu + theta.tail(r).dot(eps.head(r));
  }
  float pronostico(int h) const {
    float p = pronosticoCrudo(h);
    return isnan(p) ? p : fminf(fmaxf(p, lo), hi);   // límites físicos
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
        eps.setZero();
      } else {
        eps.tail(Q_MAX - 1) = eps.head(Q_MAX - 1).eval();   // desplaza
        eps(0) = e;
      }
    }

    serie.push(x);
    predSig = pronostico(1);
  }

  void reiniciar() {           // hueco largo: se conservan los parámetros
    serie.limpiar();
    histHead = histCount = 0;
    eps.setZero();
    predSig = predEval = NAN;
  }

  // Ajusta MA(0..Q_MAX) sobre la ventana y elige q por AIC
  void ajustar() {
    const int n = serie.count;
    if (n < N_MIN_AJUSTE) return;
    const VectorXf x = serie.vector();
    const float media = x.mean();

    // Hannan-Rissanen, paso 1: residuales de un AR(m) largo (0 para t < m)
    const int m = (n > 10 * M_AR_HR) ? M_AR_HR : 2;
    VectorXf ehat = VectorXf::Zero(n);
    VectorXf a;
    const MatrixXf Xar = matrizRezagos(x, m, m);
    const bool hrOk = minimosCuadrados(Xar, x.tail(n - m), a);
    if (hrOk) ehat.tail(n - m) = x.tail(n - m) - Xar * a;

    int      mejorQ = -1;
    VectorXf mejorBeta;
    float    mejorSSE = INFINITY;

    for (int qq = 0; qq <= Q_MAX; qq++) {
      // Inicio por defecto: mu = media, theta = 0
      VectorXf beta = VectorXf::Zero(qq + 1);
      beta(0) = media;
      // Hannan-Rissanen, paso 2 (si falla o no es invertible se usa el defecto)
      VectorXf betaHR;
      if (qq > 0 && hrOk && valoresInicialesHR(x, ehat, qq, m, betaHR) && isfinite(sseCSS(x, betaHR)))
        beta = betaHR;

      float sse = ajustarCSS(x, beta);

      if (isfinite(sse) && sse > 0) {
        criteriosInformacion(sse, n, qq + 2, aic(qq), bic(qq));   // k = mu, thetas y sigma2
        if (mejorQ < 0 || aic(qq) < aic(mejorQ)) {
          mejorQ = qq; mejorBeta = beta; mejorSSE = sse;
        }
      } else {
        aic(qq) = bic(qq) = NAN;
      }
      yield();
    }
    if (mejorQ < 0) return;

    ok      = true;
    q       = mejorQ;
    mu      = mejorBeta(0);
    theta   = mejorBeta.tail(q);
    nAjuste = n;
    sigma2  = mejorSSE / n;

    // Residuales del modelo elegido e innovaciones actuales
    VectorXf e;
    residualesCSS(x, mejorBeta, e, nullptr);
    ajuste.limpiar();
    for (int t = 0; t < n; t++) ajuste.agregar(x(t), x(t) - e(t));
    eps.setZero();
    eps.head(q) = e.tail(q).reverse();          // eps(0) = e(t) más reciente
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
  if (maT.ok && maH.ok) display.printf("MA   T:%.1fC H:%.0f%%", maT.predSig, maH.predSig);
  else {
    char f[12];
    fmtFaltan(f, sizeof(f), maT.serie.count);
    display.printf("MA: faltan %s", f);
  }

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
    char f[12];
    fmtFaltan(f, sizeof(f), m.serie.count);
    display.printf("Ciclo de 24h: faltan\n%s\n", f);
    display.printf("\nHist: %d/%d", m.serie.count, N_VALORES);
    return;
  }
  display.printf("MA(%d) %s\n", m.q, m.nombre);
  display.printf("mu = %.4f\n", m.mu);
  // thetas en dos columnas (3 renglones para q <= 5)
  for (int i = 0; i < Q_MAX; i += 2) {
    if (i < m.q) display.printf("t%d=%7.3f", i + 1, m.theta(i));
    if (i + 1 < m.q) display.printf(" t%d=%7.3f", i + 2, m.theta(i + 1));
    display.print("\n");
  }
  if (m.q == 0) {
    display.setCursor(0, 16);
    display.print("(sin thetas: X=mu+e)");
  }
  display.setCursor(0, 40);
  display.printf("s2=%.4f AIC=%.0f\n", m.sigma2, m.aic(m.q));
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
    fmt(a, sizeof(a), m.aic(qq), 1);
    fmt(b, sizeof(b), m.bic(qq), 1);
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
  for (int i = 0; i < m.q; i++) Serial.printf("ma.L%d            = %.6f\n", i + 1, m.theta(i));
  Serial.printf("sigma2           = %.6f\n", m.sigma2);
  Serial.println("  q          AIC          BIC");
  for (int qq = 0; qq <= Q_MAX; qq++)
    Serial.printf("  %d %12.3f %12.3f%s\n", qq, m.aic(qq), m.bic(qq), qq == m.q ? "  <-" : "");
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
      if (maT.ok && maH.ok) Serial.printf("MA T: %.1f C  H: %.0f %% | ", maT.predSig, maH.predSig);
      Serial.println(sdOK ? "SD: OK" : "SD: ERROR");
    }

    // ==========================================
    // UNA MUESTRA CADA INTERVALO_MIN MINUTOS: MA(q) y SD
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
