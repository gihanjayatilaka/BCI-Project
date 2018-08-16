'''
    USAGE

    python Main.py algorithm  verboseORsilent noOFClasses noOfFilesForClass0 file0class0.txt file1class0.txt noOfFilesForClass1 file0class1.txt file1class1.txt noOfFilesToTest testfile1.txt testfile2.txt

    ex:
    python Main.py rnn verbose 2 2 trainA.txt trainAA.txt 3 trainB.txt trainBB.txt trainBBB.txt 2 testA testB
    python Main.py rnn silent 2 2 trainA.txt trainAA.txt 3 trainB.txt trainBB.txt trainBBB.txt 2 testA testB


'''

import sys
import numpy as np
import keras

SEQUENCE_LENGTH = 1000
VERBOSE=False


def fileRead(fileName,linesToRemove=4,leftColToRemove=1,rightColToRemove=3):
    file  = open(fileName, 'r')
    fullText=file.read()
    file.close()
    lines=fullText.split('\n')
    for x in range(linesToRemove):
        lines.pop(0)

    ar=[]
    for x in range(len(lines)):
        temp=[]
        fields=lines[x].split(",")
        for xx in range(len(fields)):
            fields[xx]=fields[xx].strip()

        for y in range(leftColToRemove,len(fields)-rightColToRemove):
            temp.append(float(fields[y]))
        if len(temp)==0:
            continue
        ar.append(temp)
        sys.stdout.flush()

    npar=np.zeros((len(ar), len(ar[0])))
    for i in range(len(ar)):
        for j in range(len(ar[0])):
            npar[i][j] = ar[i][j]
        npar[i][:]=(npar[i][:]-np.mean(npar[i][:]))
    if(VERBOSE): print("Finished reading file "+fileName+" and returned a matrix: "+str(npar.transpose().shape))
    return npar.transpose()


def rnn_model1(NO_CLASSES,NO_CHANNELS):
    from keras.models import Sequential
    from keras.layers import LSTM,Dense,Softmax,BatchNormalization,Input
    model=Sequential()
    model.add(LSTM(100,input_shape=(SEQUENCE_LENGTH,NO_CHANNELS)))
    model.add(Dense(NO_CLASSES))
    model.add(Softmax())
    return model

def cnnFit(Xtrain,YTrain,NO_CLASSES):
    Xnew = np.ndarray((0))
    Ynew = np.ndarray((0))
    for f in range(len(Xtrain)):

        startPoints = []
        startPoint = np.random.randint(0, SEQUENCE_LENGTH)

        while startPoint + SEQUENCE_LENGTH < Xtrain[f].shape[1]:
            startPoints.append(startPoint)
            startPoint += np.random.randint(0, SEQUENCE_LENGTH)

        Xtemp = np.ndarray((len(startPoints), Xtrain[0].shape[0],SEQUENCE_LENGTH))
        Ytemp = np.ndarray(len(startPoints), dtype=np.int32)

        for seqNum in range(len(startPoints)):
            for timeStamp in range(SEQUENCE_LENGTH):
                for channel in range(Xtemp.shape[1]):
                    Xtemp[seqNum, channel, timeStamp] = Xtrain[f][channel,startPoints[seqNum] + timeStamp]
            Ytemp[seqNum] = Ytrain[f]

        if f == 0:
            Xnew = Xtemp
            Ynew = Ytemp
        else:
            if (VERBOSE): print("Concatnating", Xnew.shape, Xtemp.shape)
            Xnew = np.concatenate((Xnew, Xtemp))
            Ynew = np.concatenate((Ynew, Ytemp))

    Ynew = keras.utils.to_categorical(Ynew, np.max(Ynew) + 1)
    Xnnew=np.zeros((Xnew.shape[0],Xnew.shape[1],Xnew.shape[2],1))
    for seqNum in range(Xnew.shape[0]):
        for channel in range(Xnew.shape[1]):
            Xnnew[seqNum,channel,:,0]=np.abs(np.fft.fft(Xnew[seqNum,channel,:]))

    from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
    from keras.models import Sequential

    #BULLSHIT!!
    model=Sequential()
    model.add(Conv2D(input_shape=(Xnnew.shape[1],Xnnew.shape[2],1),filters=50,kernel_size=(1,100)))
    model.add(MaxPooling2D(pool_size=(1,5)))
    model.add(Flatten())

    model.add(Dropout(0.5))
    model.add(Dense(20,activation='relu'))


    model.add(Dense(NO_CLASSES,activation='softmax'))
    model.compile(optimizer="adam",loss="mean_squared_error")
    model.summary()
    if(VERBOSE):
        input("Start fitting? [Press ENTER]")
    model.fit(Xnnew,Ynew,epochs=10)
    return model


def cnnPredict(model,X):
    startPoints = []
    startPoint = np.random.randint(0, SEQUENCE_LENGTH)

    while startPoint + SEQUENCE_LENGTH < X.shape[1]:
        startPoints.append(startPoint)
        startPoint += np.random.randint(0, SEQUENCE_LENGTH)

    Xnew = np.ndarray((len(startPoints), X.shape[0], SEQUENCE_LENGTH,1))

    for seqNum in range(len(startPoints)):
        for timeStamp in range(SEQUENCE_LENGTH):
            for channel in range(Xnew.shape[1]):
                Xnew[seqNum, channel, timeStamp,0] = X[channel, startPoints[seqNum] + timeStamp]

    Xnnew=np.zeros((Xnew.shape[0],Xnew.shape[1],Xnew.shape[2],1))
    for seqNum in range(Xnew.shape[0]):
        for channel in range(Xnew.shape[1]):
            Xnnew[seqNum,channel,:,0]=np.abs(np.fft.fft(Xnew[seqNum,channel,:])).flatten()




    if (VERBOSE):
        print("TEst dataset is", Xnnew)
        a = input('Start predicting? [Press ENTER]')
    Y = model.predict(Xnnew)

    if (VERBOSE): print("The result is", Y)

    print("Answer: ", np.argmax(np.sum(Y, axis=0)))


def rnnFit(Xtrain,YTrain,NO_CLASSES):
    Xnew=np.ndarray((0))
    Ynew=np.ndarray((0))
    for f in range(len(Xtrain)):

        startPoints = []
        startPoint = np.random.randint(0, SEQUENCE_LENGTH)

        while startPoint+SEQUENCE_LENGTH < Xtrain[f].shape[1]:
            startPoints.append(startPoint)
            startPoint+=np.random.randint(0,SEQUENCE_LENGTH)

        Xtemp = np.ndarray((len(startPoints), SEQUENCE_LENGTH, Xtrain[0].shape[0]))
        Ytemp = np.ndarray(len(startPoints),dtype=np.int32)


        for seqNum in range(Xtemp.shape[0]):
            for timeStamp in range(SEQUENCE_LENGTH):
                for channel in range(Xtemp.shape[2]):
                    Xtemp[seqNum,timeStamp,channel]=Xtrain[f][channel,startPoints[seqNum]+timeStamp]
            Ytemp[seqNum]=Ytrain[f]


        if f==0:
            Xnew=Xtemp
            Ynew=Ytemp
        else:
            if(VERBOSE):print("Concatnating",Xnew.shape,Xtemp.shape)
            Xnew=np.concatenate((Xnew,Xtemp))
            Ynew = np.concatenate((Ynew, Ytemp))


    model=rnn_model1(NO_CLASSES,Xnew.shape[2])
    model.compile(optimizer="adam",loss="mean_squared_error")
    model.summary()

    if (VERBOSE):
        print(Xnew,Ynew)
        a=input('Start fitting? [Press ENTER]')
        print("Fitting for shapes: ",Xnew.shape,Ynew.shape)
        model.fit(Xnew,Ynew,epochs=100,verbose=1)
    else:
        model.fit(Xnew, Ynew, epochs=100, verbose=0)

    return model

def rnnPredict(model,X):


    startPoints = []
    startPoint = np.random.randint(0, SEQUENCE_LENGTH)

    while startPoint + SEQUENCE_LENGTH < X.shape[1]:
        startPoints.append(startPoint)
        startPoint += np.random.randint(0, SEQUENCE_LENGTH)

    Xtemp = np.ndarray((len(startPoints), SEQUENCE_LENGTH, X.shape[0]))

    for seqNum in range(Xtemp.shape[0]):
        for timeStamp in range(SEQUENCE_LENGTH):
            for channel in range(Xtemp.shape[2]):
                Xtemp[seqNum, timeStamp, channel] = X[channel, startPoints[seqNum] + timeStamp]

    if(VERBOSE):
        print("TEst dataset is",Xtemp)
        a=input('Start predicting? [Press ENTER]')
    Y=model.predict(Xtemp)

    if(VERBOSE):print("The result is",Y)

    print("Answer: ",np.argmax(np.sum(Y,axis=0)))




if __name__== "__main__":
    Xtrain=[]
    Ytrain=[]
    ALGORITHM=(sys.argv[1])
    if(sys.argv[2] in ['v','verbose','V']):
        VERBOSE=True

    NO_CLASSES=int(sys.argv[3])
    argvIndex=4
    FILES_PER_CLASS=np.zeros(NO_CLASSES,dtype=np.int32)
    for classNo in range(NO_CLASSES):
        FILES_PER_CLASS[classNo]=int(sys.argv[argvIndex])
        argvIndex+=1

        for trainFileIndex in range(FILES_PER_CLASS[classNo]):
            fileNameToRead=sys.argv[argvIndex]
            argvIndex+=1

            Xtrain.append(fileRead(fileNameToRead))
            Ytrain.append(classNo)


    if ALGORITHM=="rnn":
        rnnModel=rnnFit(Xtrain,Ytrain,NO_CLASSES)

        FILES_TO_PREDICT=int(sys.argv[argvIndex])
        argvIndex+=1

        for predictFileIndex in range(FILES_TO_PREDICT):
            fileNameToRead = sys.argv[argvIndex]
            argvIndex += 1

            Xtest=fileRead(fileNameToRead)
            rnnPredict(rnnModel,Xtest)

    if ALGORITHM=="cnn":
        cnnModel=cnnFit(Xtrain,Ytrain,NO_CLASSES)
        FILES_TO_PREDICT = int(sys.argv[argvIndex])
        argvIndex += 1

        for predictFileIndex in range(FILES_TO_PREDICT):
            fileNameToRead = sys.argv[argvIndex]
            argvIndex += 1

            Xtest = fileRead(fileNameToRead)
            cnnPredict(cnnModel, Xtest)



 #python Main.py rnn v 3 2 ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_R1.txt ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_R2.txt 2 ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_G1.txt ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_G2.txt 2 ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_B1.txt ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_B2.txt 3 ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_R3.txt ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_G3.txt ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_B3.txt