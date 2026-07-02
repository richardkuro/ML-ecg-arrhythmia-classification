#include <WiFi.h>
#include <WebServer.h>
#include <WebSocketsServer.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// --- Wi-Fi Credentials ---
const char* ssid = "Testwifi";
const char* password = "1234abcd";

// --- Network Servers ---
WebServer server(80);          // Standard HTTP server on port 80
WebSocketsServer webSocket(81); // WebSocket server on port 81

// --- OLED Setup ---
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// --- AD8232 Pins (ESP32-C3 Super Mini) ---
const int OUTPUT_PIN = 3;  
const int LO_PLUS = 10;    
const int LO_MINUS = 7;    

// --- Graphing & Timing Variables ---
int oledXPos = 0;
int oledLastY = SCREEN_HEIGHT / 2;

unsigned long previousSampleMillis = 0;
unsigned long previousDisplayMillis = 0;
const long sampleInterval = 4;   // 250Hz sampling
const long displayInterval = 20; // 50 FPS OLED refresh

// ==============================================================================
// HTML & JavaScript Dashboard (Stored in ESP32 Program Memory)
// ==============================================================================
const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <title>Live ECG Monitor</title>
  <style>
    body { background-color: #111; color: #0f0; font-family: monospace; text-align: center; }
    canvas { background-color: #000; border: 1px solid #333; margin-top: 20px; border-radius: 8px;}
  </style>
</head>
<body>
  <h2>ESP32 Live ECG Dashboard</h2>
  <canvas id="ecgCanvas" width="800" height="400"></canvas>
  
  <script>
    var canvas = document.getElementById('ecgCanvas');
    var ctx = canvas.getContext('2d');
    var x = 0;
    var lastY = 200; // Middle of the 400px canvas
    
    // Connect to the ESP32 WebSocket
    var gateway = `ws://${window.location.hostname}:81/`;
    var websocket = new WebSocket(gateway);
    
    websocket.onmessage = function(event) {
      var ecgValue = parseInt(event.data);
      
      // Map the 12-bit ADC value (0-4095) to the Canvas height (400-0)
      // Inverted so spikes go up
      var y = 400 - (ecgValue * 400 / 4095);
      
      // If we reach the right edge, clear the canvas and wrap around
      if (x === 0) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        lastY = y;
      }
      
      // Draw the sweeping trace
      ctx.beginPath();
      ctx.moveTo(x - 1, lastY);
      ctx.lineTo(x, y);
      ctx.strokeStyle = '#00FF00'; // Medical Green
      ctx.lineWidth = 2;
      ctx.stroke();
      
      lastY = y;
      x += 2; // Move 2 pixels per sample to speed up the sweep visually
      
      if (x >= canvas.width) {
        x = 0;
      }
    };
  </script>
</body>
</html>
)rawliteral";

// ==============================================================================
// Setup & Loop
// ==============================================================================

void setup() {
  Serial.begin(115200);
  
  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);

  // Initialize OLED (GPIO 8 SDA, GPIO 9 SCL)
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("SSD1306 allocation failed"));
    for(;;);
  }
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 20);
  display.println("Connecting to WiFi...");
  display.display();

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  // Display the IP Address on the OLED
  display.clearDisplay();
  display.setCursor(0, 10);
  display.println("WiFi Connected!");
  display.setCursor(0, 30);
  display.println("Go to IP Address:");
  display.setCursor(0, 50);
  display.println(WiFi.localIP());
  display.display();
  
  Serial.println("");
  Serial.println("WiFi connected.");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  // Setup Web Server to send the HTML page
  server.on("/", []() {
    server.send(200, "text/html", index_html);
  });
  server.begin();
  
  // Start WebSocket Server
  webSocket.begin();
  
  delay(3000); // Leave IP on screen for 3 seconds before starting the graph
  display.clearDisplay();
}

void loop() {
  // Keep the network services running
  webSocket.loop();
  server.handleClient();

  unsigned long currentMillis = millis();

  // TASK 1: Read and Broadcast the AD8232 at 250Hz
  if (currentMillis - previousSampleMillis >= sampleInterval) {
    previousSampleMillis = currentMillis;
    
    int oledYPos;
    int rawEcgValue;

    if (digitalRead(LO_PLUS) == HIGH || digitalRead(LO_MINUS) == HIGH) {
      // Leads are off: send flatline to both displays
      oledYPos = SCREEN_HEIGHT / 2;
      rawEcgValue = 2048; // Middle of the 0-4095 range
    } else {
      // Leads are on: read real data
      rawEcgValue = analogRead(OUTPUT_PIN);
      oledYPos = map(rawEcgValue, 0, 4095, SCREEN_HEIGHT, 0);
    }

    // Send the raw data over Wi-Fi instantly
    String payload = String(rawEcgValue);
    webSocket.broadcastTXT(payload);

    // Draw the local OLED buffer
    drawLineInOledBuffer(oledYPos);
  }

  // TASK 2: Push the buffer to the physical OLED at 50 FPS
  if (currentMillis - previousDisplayMillis >= displayInterval) {
    previousDisplayMillis = currentMillis;
    display.display(); 
  }
}

// Helper function for the OLED drawing
void drawLineInOledBuffer(int currentY) {
  if (oledXPos == 0) {
    display.clearDisplay();
    oledLastY = currentY; 
  }

  display.drawLine(oledXPos - 1, oledLastY, oledXPos, currentY, SSD1306_WHITE);
  
  oledLastY = currentY;
  oledXPos++;

  if (oledXPos >= SCREEN_WIDTH) {
    oledXPos = 0; 
  }
}
