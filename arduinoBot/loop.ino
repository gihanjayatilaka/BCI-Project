
void loop() {

  switch (mode) {
    case TEST:

      test();
      /*if (allIn == true || allInCounter == 50) {
        mode = END;
        motorWrite(0, 0);
        } else {
        lineFollow();
        }*/
      break;

    case BEGIN:

      buttonStatus = digitalRead(buttonPin);
      Serial.println(buttonStatus);

      if (buttonStatus == 0 ) {
        mode = TEST; //MAZE_RUN_ADVANCED;

      }
      //delay(100);

      break;


    case MAZE_RUN:

      mazeRun();
      break;


    case MAZE_RUN_ADVANCED:
      if (isBoxThere()) {
        if (isMazeSolved == 0 ) {

          // solve the maze using matrix



          isMazeSolved = 1;
        }
        mode = PICKING_BOX;
        break;
      }
      mazeRunAdvanced();
      break;

    case END:
      delay(1000);
      break;
  }

}
