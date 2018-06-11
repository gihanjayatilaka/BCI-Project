
/****************************************************************************
  Functions

  motorWrite(left,right)
  turn(direction)
  wait
  calibrateSpeed()

****************************************************************************/


void motorWrite(int leftMotorSpeed, int rightMotorSpeed) {

  //leftMotorSpeed += drift;
  //rightMotorSpeed -= drift;

  //Set motor direction according to the speed value
  if (leftMotorSpeed > 0) {
    digitalWrite(leftMotor1, HIGH);
    digitalWrite(leftMotor2, LOW);

  } else if (leftMotorSpeed < 0) {
    leftMotorSpeed  *= - slowFactor;
    digitalWrite(leftMotor1, LOW);
    digitalWrite(leftMotor2, HIGH);

  } else {
    digitalWrite(leftMotor1, HIGH);
    digitalWrite(leftMotor2, LOW);
  }

  if (rightMotorSpeed > 0) {
    digitalWrite(rightMotor1, HIGH);
    digitalWrite(rightMotor2, LOW);

  } else if (rightMotorSpeed < 0) {
    rightMotorSpeed *= - slowFactor;
    digitalWrite(rightMotor1, LOW);
    digitalWrite(rightMotor2, HIGH);

  } else {
    digitalWrite(rightMotor1, HIGH);
    digitalWrite(rightMotor2, LOW);

  }

  if (rightMotorSpeed > MaxSpeed ) rightMotorSpeed = MaxSpeed;
  if (leftMotorSpeed > MaxSpeed ) leftMotorSpeed = MaxSpeed;

  analogWrite(leftMotorPWM, leftMotorSpeed);
  analogWrite(rightMotorPWM, rightMotorSpeed);

  if (sPrint == true) {
    Serial.print("   "); Serial.print(leftMotorSpeed); Serial.print(" | "); Serial.println(rightMotorSpeed);
  }
}

void motorStop() {

  analogWrite(leftMotorPWM, 0);
  analogWrite(rightMotorPWM, 0);

  digitalWrite(leftMotor1, HIGH);
  digitalWrite(leftMotor2, HIGH);

  digitalWrite(leftMotor1, HIGH);
  digitalWrite(leftMotor2, HIGH);

}

void motorLeft(int spd) {
  motorWrite(0, spd);
}

void motorRight(int spd)
{
  motorWrite(spd, 0);
}

void wait() {
  motorWrite(0, 0);
}

void wait(int d) {
  motorWrite(0, 0);
  delay(d);
}

void calibrateSpeed() {
  Serial.println(">> Calibrating Speed \n");
  Serial.println(">> Increasing Mode");

  for (int i = 0; i <= 255; i += 1) {
    Serial.print("Speed : "); Serial.println(i);
    motorWrite(i, i);
    delay(100);
  }

  delay(5000);

  Serial.println("\n>> Decreasing Mode");

  for (int i = 250; i >= 10 ; i -= 1) {
    Serial.print("Speed : "); Serial.println(i);
    motorWrite(i, i);
    delay(100);
  }
}



