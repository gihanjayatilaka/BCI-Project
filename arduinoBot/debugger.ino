

void sendLineData(String line) {
  String output = "" + line;
  Serial.println(output);
  //delay(100);
}

void sendMotorData(int motor, int spd) {

  String output = "m:";

  if (motor == LEFT) {
    output += "le:";
  } else if (motor == RIGHT) {
    output += "ri:";
  }
  output += spd;
  //Serial.println(output);
  //delay(100);
}

void sendOtherData(int key, int value){

  //delay(100);
}



