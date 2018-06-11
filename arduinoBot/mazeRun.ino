
void mazeRun() {
  //SENSOR NUMBERS SHOULD BE CHANGED!
  int frontDist = getDist(0);
  int leftDist = getDist(1);
  int rightDist = getDist(2);

  // Right Hand Rule
  if (rightDist > maze_THERSOLD_FOR_WALL) maze_turnRight();
  else if (frontDist > maze_THERSOLD_FOR_WALL) maze_goForward();
  else if (leftDist > maze_THERSOLD_FOR_WALL) maze_turnLeft();
  else maze_turnBack();

  /*
    // Left Hand Rule
    if (leftDist > maze_THERSOLD_FOR_WALL) maze_turnLeft();
    else if (frontDist > maze_THERSOLD_FOR_WALL) maze_goForward();
    else if (rightDist > maze_THERSOLD_FOR_WALL) maze_turnRight();
    else maze_turnBack();
  */

}



void maze_goForward() {
  motorWrite(maze_forwardStepSpeed, maze_forwardStepSpeed);
  delay(maze_forwardStepTime);
  motorStop();
}
void maze_turnLeft() {
  motorWrite(maze_turnLeft_LeftMotorSpeed, maze_turnLeft_RightMotorSpeed);
  delay(maze_turnLeft_Time);
  motorStop();
  maze_goForward();
}
void maze_turnRight() {
  motorWrite(maze_turnRight_LeftMotorSpeed, maze_turnRight_RightMotorSpeed);
  delay(maze_turnRight_Time);
  motorStop();
  maze_goForward();
}
void maze_turnBack() {
  motorWrite(maze_turnBack_LeftMotorSpeed, maze_turnBack_RightMotorSpeed);
  delay(maze_turnBack_Time);
  motorStop();
  maze_goForward();
}

