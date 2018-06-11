
/****************************************************************************
  Functions

  lineFollowingBegin()
  int, calculatePID()

****************************************************************************/

void lineFollowBegin() {

  center = (NUM_SENSORS - 1) * 5;
  lineType = BLACK;

  Serial.println(">> Begin...");

  pinMode(leftMotor1, OUTPUT);
  pinMode(leftMotor2, OUTPUT);
  //pinMode(leftMotorPWM, OUTPUT);

  pinMode(rightMotor1, OUTPUT);
  pinMode(rightMotor2, OUTPUT);
  //pinMode(rightMotorPWM, OUTPUT);

  motorWrite(0, 0);
}

void lineFollow() {

  position = readLine(sensors);
  error = (position - center);

  if (sPrint == true) {
    Serial.println(readLineString());
  }

  if (allOut) {
    motorWrite(MaxSpeed, MaxSpeed);
  }
  else {
    if (line != lastLine) {

      //PID Calculating
      int motorSpeed = calculatePID(error);

      //Assigning motor speeds
      int rightMotorSpeed = BaseSpeed + motorSpeed;
      int leftMotorSpeed = BaseSpeed - motorSpeed;

      //Remapping motor speed
      motorWrite(leftMotorSpeed, rightMotorSpeed);

      lastLine = line;
    }
  }
}
int calculatePID(int error) {

  int P = error;
  int I = I + error;
  int D = (error - lastError);

  int value = (P * 20) + (I * Ki) + (D * Kd);

  lastError = error;

  return value;
}




