
#define NUM_SENSORS 6

#define BLACK 0
#define WHITE 1

#define pickPin 5
#define grabPin 4

#define FRONT 8
#define LEFT 4
#define RIGHT 6
#define BACK 2

#define rightMotor1 7
#define rightMotor2 8
#define rightMotorPWM 9

#define leftMotor1 11
#define leftMotor2 12
#define leftMotorPWM 10

/*#define RIGHT 6
  #define LEFT 4*/

#define buttonPin 4

float Kp = 0, Kd = 0, Ki = 0;
const double slowFactor = 0.2;
const double speedFactor = 5;

#define MaxSpeed 200
#define BaseSpeed 100

int error = 0;
int lastError = 0;
int center = 0;
int drift = 10;

int allInCounter = 0;

unsigned const int sensor_values[NUM_SENSORS];
//unsigned int pins[] = {14, 15, 16, 19, 20, 21};

// Mode eNum
enum {BEGIN, LINE_FOLLOW, STOP, TEST, MAZE_RUN, MAZE_RUN_ADVANCED, PICKING_BOX, END};

// EEPROM eNum
enum {eP, eI, eD, eMax, eBase, ePrint};
int sPrint = true;


int buttonStatus = 1;

//----------------------------------------------------------------------------------------------
// Sensors specified variables

unsigned int sensors[NUM_SENSORS];
int position = 30 ;

boolean allOut = 0;
boolean allIn = 0;
boolean lineType = WHITE;

int last_value = 0;

String line = "";
String lastLine = "";


//----------------------------------------------------------------------------------------------


//------------------------------------------------------------------------
//These variables are defined for the maze traversal

int maze[6][6];
int isMazeSolved = 0;
int mazeCounter = 0;
int posX = 0, posY = 0;
//int dir[][] = { {0,-1}, {0,1}, {1,0}, {-1,0}};

int maze_THERSOLD_FOR_WALL = 10; //in cm

int maze_forwardStepTime = 10;
int maze_forwardStepSpeed = 100;

int maze_turnLeft_RightMotorSpeed;
int maze_turnLeft_LeftMotorSpeed = -1 * maze_turnLeft_RightMotorSpeed;
int maze_turnLeft_Time;

int maze_turnRight_LeftMotorSpeed;
int maze_turnRight_RightMotorSpeed = -1 * maze_turnRight_LeftMotorSpeed;
int maze_turnRight_Time;


int maze_turnBack_LeftMotorSpeed;
int maze_turnBack_RightMotorSpeed = -1 * maze_turnBack_LeftMotorSpeed;
int maze_turnBack_Time;
//-------------------------------------------------------------------------


