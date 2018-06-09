

X=[]
Y=[]
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




