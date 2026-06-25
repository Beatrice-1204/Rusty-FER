#include <AccelStepper.h>
#include <SoftwareSerial.h>

// ---- HC-42 pe pinii PCB ----
#define BT_RX A8   // primeste de la HC-42 (TX)
#define BT_TX 38   // trimite la HC-42 (RX)

SoftwareSerial BT(BT_RX, BT_TX);

// ---- DRV8825 ----
AccelStepper LF(AccelStepper::DRIVER, 40, 41);
AccelStepper LB(AccelStepper::DRIVER, 42, 43);
AccelStepper RF(AccelStepper::DRIVER, 46, 47);
AccelStepper RB(AccelStepper::DRIVER, 44, 45);

const int LF_ENABLE_PIN = 22;
const int LB_ENABLE_PIN = 23;
const int RF_ENABLE_PIN = 24;
const int RB_ENABLE_PIN = 25;
const int DRIVER_ENABLE_ON = LOW;
const int DRIVER_ENABLE_OFF = HIGH;

int speedVal = 1200;
unsigned long lastCmd = 0;
const unsigned long TIMEOUT = 600;
bool motorsEnabled = false;

const int SPEED_HAPPY = 2200;
const int SPEED_ANGRY = 1400;
const int SPEED_SAD = 700;
const int SPEED_SURPRISE = 1800;

const unsigned long HAPPY_DURATION = 700;
const unsigned long ANGRY_DURATION = 500;
const unsigned long SAD_DURATION = 900;
const unsigned long SURPRISE_STEP_DURATION = 300;

void setup() {
  BT.begin(9600);   // baud HC-42

  pinMode(LF_ENABLE_PIN, OUTPUT);
  pinMode(LB_ENABLE_PIN, OUTPUT);
  pinMode(RF_ENABLE_PIN, OUTPUT);
  pinMode(RB_ENABLE_PIN, OUTPUT);

  // dezactivam driverele la inceput pentru a preveni miscari nedorite
  digitalWrite(LF_ENABLE_PIN, DRIVER_ENABLE_OFF);
  digitalWrite(LB_ENABLE_PIN, DRIVER_ENABLE_OFF);
  digitalWrite(RF_ENABLE_PIN, DRIVER_ENABLE_OFF);
  digitalWrite(RB_ENABLE_PIN, DRIVER_ENABLE_OFF);
  motorsEnabled = false;

  LF.setMaxSpeed(5000);
  LB.setMaxSpeed(5000);
  RF.setMaxSpeed(5000);
  RB.setMaxSpeed(5000);

  stopAll();
}

void loop() {
  if (BT.available()) {
    char c = BT.read();
    lastCmd = millis();

    if (c >= '1' && c <= '9') {
      speedVal = map(c - '0', 1, 9, 600, 3000);
    } else {
      switch (c) {
        case 'w': forward(speedVal); break;
        case 's': backward(speedVal); break;
        case 'a': left(speedVal); break;
        case 'd': right(speedVal); break;
        case 'q': rotateLeft(speedVal); break;
        case 'e': rotateRight(speedVal); break;
        case 'x': stopAll(); break;
        case 'H': reactionHappy(); break;
        case 'G': reactionAngry(); break;
        case 'V': reactionSad(); break;
        case 'U': reactionSurprise(); break;
      }
    }
  }

  if (millis() - lastCmd > TIMEOUT) stopAll();

  LF.runSpeed();
  LB.runSpeed();
  RF.runSpeed();
  RB.runSpeed();
}

// ---- MISCARE ----
void setAll(int lf, int lb, int rf, int rb) {
  if (lf != 0 || lb != 0 || rf != 0 || rb != 0) {
    enableMotors();
  }

  LF.setSpeed(lf);
  LB.setSpeed(lb);
  RF.setSpeed(rf);
  RB.setSpeed(rb);
}

void forward(int speed)      { setAll( speed,  speed,  speed,  speed); }
void backward(int speed)     { setAll(-speed, -speed, -speed, -speed); }
void left(int speed)         { setAll(-speed,  speed,  speed, -speed); }
void right(int speed)        { setAll( speed, -speed, -speed,  speed); }
void rotateLeft(int speed)   { setAll(-speed, -speed,  speed,  speed); }
void rotateRight(int speed)  { setAll( speed,  speed, -speed, -speed); }
void stopAll() {
  LF.setSpeed(0);
  LB.setSpeed(0);
  RF.setSpeed(0);
  RB.setSpeed(0);
  disableMotors();
}

void enableMotors() {
  if (motorsEnabled) return;

  digitalWrite(LF_ENABLE_PIN, DRIVER_ENABLE_ON);
  digitalWrite(LB_ENABLE_PIN, DRIVER_ENABLE_ON);
  digitalWrite(RF_ENABLE_PIN, DRIVER_ENABLE_ON);
  digitalWrite(RB_ENABLE_PIN, DRIVER_ENABLE_ON);
  motorsEnabled = true;
}

void disableMotors() {
  if (!motorsEnabled) return;

  digitalWrite(LF_ENABLE_PIN, DRIVER_ENABLE_OFF);
  digitalWrite(LB_ENABLE_PIN, DRIVER_ENABLE_OFF);
  digitalWrite(RF_ENABLE_PIN, DRIVER_ENABLE_OFF);
  digitalWrite(RB_ENABLE_PIN, DRIVER_ENABLE_OFF);
  motorsEnabled = false;
}

void runFor(unsigned long duration) {
  unsigned long start = millis();

  while (millis() - start < duration) {
    LF.runSpeed();
    LB.runSpeed();
    RF.runSpeed();
    RB.runSpeed();
  }
}

void reactionHappy() {
  rotateRight(SPEED_HAPPY);
  runFor(HAPPY_DURATION);
  stopAll();
}

void reactionAngry() {
  backward(SPEED_ANGRY);
  runFor(ANGRY_DURATION);
  stopAll();
}

void reactionSad() {
  forward(SPEED_SAD);
  runFor(SAD_DURATION);
  stopAll();
}

void reactionSurprise() {
  left(SPEED_SURPRISE);
  runFor(SURPRISE_STEP_DURATION);

  right(SPEED_SURPRISE);
  runFor(SURPRISE_STEP_DURATION);

  stopAll();
}
