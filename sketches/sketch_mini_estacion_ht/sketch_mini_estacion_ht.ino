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
         file.println("Fecha,Hora,Temperatura(C),Humedad(%)");
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
  RtcDateTime fechaReferencia(2026, 11, 03, 30, 0, 0);

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

  // CORRECCION DE DATOS
  float temperatura = ( 5.301868409 + (0.702422691 * ref_temperatura));
  float humedad = ( 32.60425363 + (0.6384588546 * ref_temperatura));

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
    display.setCursor(15, 18); 
    if (now.IsValid()) {
      display.printf("%02d:%02d:%02d", now.Hour(), now.Minute(), now.Second());
    } else {
      display.print("--:--:--");
    }

    // --- MOSTRAR CLIMA E INDICADOR SD ---
    display.setTextSize(1);
    display.setCursor(0, 42); 
    if (isnan(temperatura) || isnan(humedad)) {
      display.print("Error SHT3X");
    } else {
      display.printf("T:%.1f C H:%.0f %%", temperatura, humedad);
    }

    // Línea inferior de estado
    display.setCursor(0, 55);
    if (sdOK) {
      display.print("SD: Guardando...");
    } else {
      display.print("SD: ERROR");
    }

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
      if (sdOK) {
        File dataFile = SD.open("/datos_clima.csv", FILE_APPEND);
        if (dataFile) {
          dataFile.printf("%02d/%02d/%04d,%02d:%02d:%02d,%.1f,%.0f\n", 
                        now.Day(), now.Month(), now.Year(),
                        now.Hour(), now.Minute(), now.Second(),
                        temperatura, humedad);
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