
int r,g,b;
boolean even = false;
int[] RGB={255,0,0};
int i=0;
void setup() {
  size(640, 360);
  //colorMode(HSB, height, height, height);  
  noStroke();
  frameRate(20);
}

void draw() 
{
  even=!even;
  if(even){
    r=0;g=0;b=0;
  }
  else{
    r=RGB[0];g=RGB[1];b=RGB[2];
  }
  background(r,g,b);
}

void mouseClicked(){
  RGB[i]=0;
  i=(i+1)%3;
  RGB[i]=255;
}