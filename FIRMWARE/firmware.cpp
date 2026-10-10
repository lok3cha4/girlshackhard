#include <Arduino.h>
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>

#define TFT_CS P2_IO0
#define TFT_RST P2_IO1
#define TFT_DC P2_IO2
SPIClass mySPI(FSPI);
Adafruit_ST7735 tft = Adafruit_ST7735(&mySPI, TFT_CS, TFT_DC, TFT_RST);
GFXcanvas16 canvas(160, 80);
bool displayDirty = true;
unsigned long lastDraw = 0;

// 1 segment, 2 servos: A0 = pitch tendon pair, A1 = yaw tendon pair.
// TODO: set the real Axiometa pin names (servo 0 / A0 was on P1_IO2 before).
// #define PITCH_SERVO_PIN P1_IO2
// #define YAW_SERVO_PIN   <yaw pin>
#if !defined(PITCH_SERVO_PIN) || !defined(YAW_SERVO_PIN)
#error "Set PITCH_SERVO_PIN and YAW_SERVO_PIN to the real Axiometa pin names"
#endif
constexpr uint8_t NUM_SERVOS = 2;
constexpr uint8_t SERVO_PINS[NUM_SERVOS] = { PITCH_SERVO_PIN, YAW_SERVO_PIN };
constexpr int SERVO_MIN_US = 500;
constexpr int SERVO_MAX_US = 2400;

// Software travel limits — keep well inside 0-180 so a bad command
// can't over-pull a tendon. Tune these once the real robot is assembled.
constexpr int SAFE_MIN_DEG = 30;
constexpr int SAFE_MAX_DEG = 150;

int angle[NUM_SERVOS] = {90, 90};
String inputLine;

void drawAngles() {
  canvas.fillScreen(0);
  canvas.setTextWrap(false);
  canvas.setTextColor(0xFFFF);
  canvas.setTextSize(1);
  canvas.setCursor(4, 4);
  canvas.print("COMMANDED SERVO ANGLES");
  canvas.setTextSize(2);
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    canvas.setCursor(4, 20 + i * 15);
    canvas.printf("A%d: %3d deg", i, angle[i]);
  }
  tft.drawRGBBitmap(0, 0, canvas.getBuffer(), 160, 80);
  displayDirty = false;
  lastDraw = millis();
}

void writeServo(uint8_t i) {
  int a = constrain(angle[i], SAFE_MIN_DEG, SAFE_MAX_DEG);
  const unsigned long pulse = SERVO_MIN_US +
      (unsigned long)(SERVO_MAX_US - SERVO_MIN_US) * a / 180;
  analogWrite(SERVO_PINS[i], (pulse * 4095UL + 10000UL) / 20000UL);
}

// Parses "A0:123,A1:87" (fields assigned by position; extra fields ignored,
// missing fields leave that servo unchanged)
void parseLine(const String& line) {
  int idx = 0, pos = 0;
  while (idx < NUM_SERVOS) {
    int colon = line.indexOf(':', pos);
    int comma = line.indexOf(',', pos);
    if (colon < 0) break;
    int endPos = (comma < 0) ? line.length() : comma;
    int val = line.substring(colon + 1, endPos).toInt();
    angle[idx] = constrain(val, 0, 180);
    writeServo(idx);
    idx++;
    if (comma < 0) break;
    pos = comma + 1;
  }
  displayDirty = true;
  Serial.printf("OK %d,%d\n", angle[0], angle[1]);
}

void setup() {
  Serial.begin(115200);
  mySPI.begin(SCK, MISO, MOSI);
  tft.initR(INITR_MINI160x80);
  tft.setRotation(3);
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    pinMode(SERVO_PINS[i], OUTPUT);
    analogWriteFrequency(SERVO_PINS[i], 50);
    analogWriteResolution(SERVO_PINS[i], 12);
    writeServo(i);  // center all servos at startup
  }
  Serial.println("Ready. Send: A0:deg,A1:deg");
  drawAngles();
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      parseLine(inputLine);
      inputLine = "";
    } else if (c != '\r') {
      inputLine += c;
    }
  }
  if (displayDirty && millis() - lastDraw >= 33) {
    drawAngles();
  }
}