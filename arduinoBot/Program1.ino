#include <Servo.h>
#include <EEPROM.h>
#include <SoftwareSerial.h>
#include "define.h"

SoftwareSerial mySerial(4, 5); // RX, TX
volatile int mode = BEGIN ;

void setup() {
  Serial.begin(9600);
  mySerial.begin(9600);
  //lineFollowBegin();
  pinMode(13, OUTPUT);
  pinMode(buttonPin, INPUT_PULLUP);
}

void test() {

  motorWrite(0,150);
  delay(1000);

 
  motorWrite(0,0);
  delay(1000);
  /*
    //calibrateSpeed();
    motorWrite(-150, 150);
    delay(500);*/
}
/*

  void turnBack() {
  position = readLine(sensors);

  if (allIn == true) {
    motorWrite(-100 * speedFactor, 100);
    robotStatus = 0;
  }

  else if (allOut == true) {
    robotStatus = 1;
    motorWrite(-100 * speedFactor, 100);
  }

  else if (robotStatus == 1 && allOut == false) {
    motorWrite(-100 * speedFactor, 100);
    delay(150);
    robotStatus = 2;
    motorWrite(0, 0);
  }
  else {
    motorWrite(-100 * speedFactor, 100);
  }

  }*/


