/****************************************************************************
  Functions

  array,  readLine(*sensorValues)
  String, readLineString
****************************************************************************/

int avg = 0;
int sum = 0;
int value = 0;
int on_line = 0;



const int rightEdge = (NUM_SENSORS - 1) * 10 / 2;

int readLine(unsigned int *sensor_values)
{
  on_line = 0;
  avg = 0;
  sum = 0;
  line = "";

  allIn = false;
  allOut = false;

  for (int i = 0; i < NUM_SENSORS; i++) {
    value = pinRead(i + 1);;
    sensor_values[i] = value;

    avg += (long)(value * (i * 10));
    sum += value;
    line += value;

  }

  if (sum == NUM_SENSORS){
    allIn = true;
    allInCounter++;
  }else{
    allInCounter = 0;
  }
  //if (line == "110011") allIn = true;

  if (sum == 0) allOut = true;

  if (allOut)
  {
    if (last_value < rightEdge) {                                           // If line missed from right end
      last_value = 0;

    } else if ((last_value == (NUM_SENSORS - 1) * 10 / 2) && (sum == 0)) {  // if sum = 0 mathamatically return -1 for output
      last_value = (NUM_SENSORS - 1) * 10 / 2;

    } else if ((last_value == (NUM_SENSORS - 1) * 10 / 2)) {                // Normal output
      last_value = avg / sum;

    } else {                                                                // if line missed from left edge
      last_value =  (NUM_SENSORS - 1) * 10;

    }
  } else {
    last_value = avg / sum;
  }

  return last_value;
}


String readLineString() {

  String output = "";

  for (int i = 0; i < NUM_SENSORS; i++) {
    int val = pinRead(i + 1);
    output += val;
  }

  return output;
}

int pinRead(int analogPin) {

  int reading = analogRead(analogPin);

  // Nomal reading, 1=HIGH, 0=LOW
  if (reading > 512) {
    value = 1;
  } else {
    value = 0;
  }

  //Formatted output for diferrent backgrounds
  if (lineType == BLACK) {
    value = value;
  } else if (lineType == WHITE) {
    value = 1 - value;
  }

  return value;
}


//Ultrasonic sensors
/*
  void ultrasonicBegin() {
  pinMode(trig, OUTPUT);
  pinMode(echo, INPUT);
  digitalWrite(trig, LOW);
  }


  int getDistance() {

  if (side == LEFT) {
    echo = 4;
  }
  else if (side == RIGHT) {
    echo = 3;
  }
  digitalWrite(trig, HIGH);
  delayMicroseconds(20);
  digitalWrite(trig, LOW);
  duration = pulseIn(echo, HIGH);
  distance = duration / 58;

  if (distance > 100)distance = 100;

  return distance;

  }
*/



