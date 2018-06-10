import random


print('python ./FDA.py 40 750 300',end=' ')
fileNames=[]

for x in range(250):
    fileNames.append('./threeClasses/ec'+str(x)+'train.txt 0')
    fileNames.append('./threeClasses/eo'+str(x)+'train.txt 1')
    fileNames.append('./threeClasses/rf'+str(x)+'train.txt 2')

while(len(fileNames))>0:
    r=random.randint(0,len(fileNames)-1)
    print(fileNames.pop(r),end=' ')

for x in range(100):
    fileNames.append('./threeClasses/ec'+str(x)+'test.txt 0')
    fileNames.append('./threeClasses/eo'+str(x)+'test.txt 1')
    fileNames.append('./threeClasses/rf'+str(x)+'test.txt 2')

while(len(fileNames))>0:
    r=random.randint(0,len(fileNames)-1)
    print(fileNames.pop(r),end=' ')
