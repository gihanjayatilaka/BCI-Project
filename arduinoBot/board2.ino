

void pick() {
  mySerial.println("pi");
}

void drop() {
  mySerial.println("dr");
}

void stand() {
  mySerial.println("st");
}

boolean isBoxThere(){
  mySerial.println("i");
  while (Serial.available()){
    char boxThere= (char)mySerial.read();
    return (boxThere=='1');
  }
}

int getDist(int sensor) {
  mySerial.println('d'+sensor);
  while (Serial.available()){
    char dist= (char)mySerial.read();
    return dist-((char)0);
  }
 

}





