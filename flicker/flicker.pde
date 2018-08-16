
int r, g, b;
boolean even = false;
int[] RGB={255, 0, 0};
int i=0;

int refreshRate=0;
void setup() {
  size(1200, 700);
  noStroke();
}

void draw() 
{

  even=!even;
  if (even & refreshRate>2) {
    r=0;
    g=0;
    b=0;
  } else {
    r=RGB[0];
    g=RGB[1];
    b=RGB[2];
  }
  background(r, g, b);
}

void mouseClicked() {
  if (mouseButton==LEFT) {
    RGB[i]=0;
    i=(i+1)%3;
    RGB[i]=255;
  }
  if(mouseButton==RIGHT){
    refreshRate=refreshRate+5;
    refreshRate=refreshRate%50;
    if(refreshRate==0) refreshRate=1;
    frameRate(refreshRate);

  }
}
