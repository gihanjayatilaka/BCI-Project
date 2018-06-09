import sys
import numpy as np
import FileReading


DIMENSIONS=int(sys.argv[1])
TRAINING_DATA_POINTS=int(sys.argv[2])




X=np.ndarray((TRAINING_DATA_POINTS,DIMENSIONS))
Y=np.ndarray(TRAINING_DATA_POINTS)


for f in range(TRAINING_DATA_POINTS):
    fileName=input('Enter file name')
    X[0]=readFileAndMakeFeatureVector(fileName)
    Y[0]=input('Enter class')

print('Finished importing data')

from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)


correct=0
wrong=0

for x in range(len(X)):
    pre=clf.predict(X[x])
    print('x='+str(X[x])+' y='+str(Y[x])+' prediction='+str(pre))
    if pre == Y[x]:
        correct += 1
    else:
        wrong += 1

accuracy=(correct*100.0)/(correct+wrong)
print('accuracy='+str(accuracy))




