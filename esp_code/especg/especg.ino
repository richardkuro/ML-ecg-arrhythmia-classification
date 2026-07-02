#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// Screen dimensions
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1 // Share reset pin
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// AD8232 Pin Connections for ESP32-C3 Super Mini
const int OUTPUT_PIN = 3;  // Analog input
const int LO_PLUS = 10;    // Digital input for Leads Off
const int LO_MINUS = 7;    // Digital input for Leads Off

// Graphing Variables
int xPos = 0;
int lastY = SCREEN_HEIGHT / 2;

// Non-blocking timers
unsigned long previousSampleMillis = 0;
unsigned long previousDisplayMillis = 0;
const long sampleInterval = 4;   // 4ms = 250Hz sampling rate
const long displayInterval = 20; // 20ms = 50 FPS screen refresh

void setup() {
  Serial.begin(115200);
  
  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);

  // Initialize the OLED 
  // (C3 Super Mini uses GPIO 8 for SDA and 9 for SCL by default)
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("SSD1306 allocation failed"));
    for(;;);
  }
  
  display.clearDisplay();
  display.display();
}

void loop() {
  unsigned long currentMillis = millis();

  // TASK 1: Read the AD8232 at exactly 250Hz
  if (currentMillis - previousSampleMillis >= sampleInterval) {
    previousSampleMillis = currentMillis;
    
    int yPos;

    // Check if the sensors are attached to the body
    if (digitalRead(LO_PLUS) == HIGH || digitalRead(LO_MINUS) == HIGH) {
      // Leads off! Draw a flatline in the middle of the screen
      yPos = SCREEN_HEIGHT / 2;
    } else {
      // Read the 12-bit ADC (0 to 4095)
      int ecgValue = analogRead(OUTPUT_PIN);
      
      // Map the 0-4095 signal to the 0-64 pixel height of the OLED. 
      // The parameters are inverted (SCREEN_HEIGHT, 0) so the graph spikes upward.
      yPos = map(ecgValue, 0, 4095, SCREEN_HEIGHT, 0);
    }

    // Calculate the line coordinates in the buffer
    drawLineInBuffer(yPos);
  }

  // TASK 2: Push the buffer to the OLED at 50 FPS
  if (currentMillis - previousDisplayMillis >= displayInterval) {
    previousDisplayMillis = currentMillis;
    display.display(); 
  }
}

// Function to draw the sweeping trace
void drawLineInBuffer(int currentY) {
  // If the trace reaches the right edge, loop back to the left and clear
  if (xPos == 0) {
    display.clearDisplay();
    lastY = currentY; // Prevent a diagonal line jumping from the end to the start
  }

  // Draw a line from the last recorded point to the current point
  display.drawLine(xPos - 1, lastY, xPos, currentY, SSD1306_WHITE);
  
  // Update last coordinates for the next loop
  lastY = currentY;
  xPos++;

  if (xPos >= SCREEN_WIDTH) {
    xPos = 0; // Reset X position
  }
}
