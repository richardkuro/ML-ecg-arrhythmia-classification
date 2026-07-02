const int OUTPUT_PIN = 3;  
const int LO_PLUS = 10;    
const int LO_MINUS = 7;    

unsigned long previousMillis = 0;
const long sampleInterval = 4; // 250 Hz

void setup() {
  // 115200 baud is fast enough to prevent serial bottlenecks at 250Hz
  Serial.begin(115200); 
  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);
}

void loop() {
  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= sampleInterval) {
    previousMillis = currentMillis;

    if (digitalRead(LO_PLUS) == HIGH || digitalRead(LO_MINUS) == HIGH) {
      // Send a text flag if the leads fall off
      Serial.println("FLATLINE");
    } else {
      // Send the raw 12-bit ADC value
      Serial.println(analogRead(OUTPUT_PIN));
    }
  }
}
