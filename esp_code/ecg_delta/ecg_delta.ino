const int OUTPUT_PIN = 3;  
const int LO_PLUS = 10;    
const int LO_MINUS = 7;    

// Switch from milliseconds to microseconds for absolute precision
unsigned long previousMicros = 0;
const long sampleInterval = 4000; // 4000 microseconds = exactly 250 Hz

void setup() {
  Serial.begin(115200); 
  
  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);
  
  // Explicitly lock the ESP32-C3 ADC settings
  analogReadResolution(12);       // Force 12-bit resolution (0-4095 range)
  analogSetAttenuation(ADC_11db); // Allow reading the full 0V to 3.3V sweep
}

void loop() {
  unsigned long currentMicros = micros();

  // Check if 4000 microseconds have passed
  if (currentMicros - previousMicros >= sampleInterval) {
    
    // Add the interval instead of setting it to currentMicros. 
    // This entirely eliminates long-term timing drift.
    previousMicros += sampleInterval;

    if (digitalRead(LO_PLUS) == HIGH || digitalRead(LO_MINUS) == HIGH) {
      Serial.println("FLATLINE");
    } else {
      Serial.println(analogRead(OUTPUT_PIN));
    }
  }
}
