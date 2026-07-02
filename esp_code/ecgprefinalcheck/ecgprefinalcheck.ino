#include <Arduino.h>
#include <U8g2lib.h>
#include <Wire.h>

// --- Hardware Pins ---
#define ECG_PIN 3
#define BUZZER_PIN 5 // Connect to a passive or active buzzer

// --- Initialize Ultra-Lightweight Page-Buffer Display ---
// The "_1_" in the constructor name specifies a tiny 128-byte page buffer instead of a 1024-byte full buffer ("_F_")
U8G2_SSD1306_128X64_NONAME_1_HW_I2C u8g2(U8G2_R0, /* reset=*/ U8X8_PIN_NONE);

// --- Real-Time Timing Variables ---
unsigned long lastSampleTime = 0;
const int SAMPLE_INTERVAL_MS = 4; // 250 Hz

// --- Non-Blocking Buzzer Timing ---
bool isBeeping = false;
unsigned long buzzerStartTime = 0;
const int BEEP_DURATION_MS = 50; // Short, crisp hospital-style beep

// --- Simple Peak Detection Variables ---
int peakThreshold = 2500; // Adjust based on your typical R-peak amplitude
bool aboveThreshold = false;

void setup() {
  Serial.begin(115200);
  
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  
  // Overclock the I2C Bus to 400kHz Fast Mode before starting the display
  Wire.begin();
  Wire.setClock(400000); 
  u8g2.begin();
  
  analogReadResolution(12);
}

void loop() {
  unsigned long currentMillis = millis();

  // 1. Strict 250Hz Data Sampling Loop
  if (currentMillis - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = currentMillis;
    
    int rawValue = analogRead(ECG_PIN);

    // 2. Inline Peak Detection (Look for the R-Wave)
    if (rawValue > peakThreshold && !aboveThreshold) {
      // The signal just crossed above the peak boundary - Trigger Beep!
      aboveThreshold = true;
      isBeeping = true;
      buzzerStartTime = currentMillis;
      
      // Turn on buzzer (Using simple digital write or tone mapping)
      digitalWrite(BUZZER_PIN, HIGH); 
    }
    
    // Reset threshold flag when the signal falls back down safely
    if (rawValue < (peakThreshold - 300)) {
      aboveThreshold = false;
    }

    // 3. Lightweight Low-Frame-Rate Screen Refresh
    // Only redraw a single line segment every few loops to keep I2C transactions tiny
    static int refreshDivider = 0;
    refreshDivider++;
    if (refreshDivider >= 10) { // Updates the UI loop at roughly 25 Hz instead of 250 Hz
      refreshDivider = 0;
      
      u8g2.firstPage();
      do {
        // Map your raw 0-4095 ADC value down to the 0-64 pixel screen height
        int yPos = map(rawValue, 0, 4095, 63, 0);
        
        // Draw just a small active indicator pixel or a tiny historic dot matrix
        u8g2.drawPixel(64, yPos); 
        
        if (isBeeping) {
          u8g2.drawStr(0, 10, "BEEP"); // Visual indicator synchronous with audio
        }
      } while (u8g2.nextPage());
    }
  }

  // 4. Asynchronous Buzzer Silencing (Completely outside the sampling loop)
  if (isBeeping && (currentMillis - buzzerStartTime >= BEEP_DURATION_MS)) {
    digitalWrite(BUZZER_PIN, LOW);
    isBeeping = false;
  }
}
