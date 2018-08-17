'''
    USAGE

    python Main.py algorithm  verboseORsilent noOFClasses noOfFilesForClass0 file0class0.txt file1class0.txt noOfFilesForClass1 file0class1.txt file1class1.txt noOfFilesToTest testfile1.txt testfile2.txt

    ex:
    python Main.py rnn verbose 2 2 trainA.txt trainAA.txt 3 trainB.txt trainBB.txt trainBBB.txt 2 testA testB
    python Main.py rnn silent 2 2 trainA.txt trainAA.txt 3 trainB.txt trainBB.txt trainBBB.txt 2 testA testB
    python Main.py fft-dnn silent 2 2 trainA.txt trainAA.txt 3 trainB.txt trainBB.txt trainBBB.txt 2 testA testB
    python Main.py fft-svm silent 2 2 trainA.txt trainAA.txt 3 trainB.txt trainBB.txt trainBBB.txt 2 testA testB


COMMAND in this repo:
    python3 Main.py fft-svm v 3 2 ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_R1.txt ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_R2.txt 2 ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_G1.txt ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_G2.txt 2 ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_B1.txt ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_B2.txt 3 ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_R3.txt ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_G3.txt ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_B3.txt


'''

import sys
import numpy as np
SEQUENCE_LENGTH = 6000
VERBOSE=False


def fileRead(fileName,linesToRemove=2000,leftColToRemove=3,rightColToRemove=6):
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


def makeFeatureVectors(X):
    PASS_BAND_LOW = 3.0
    PASS_BAND_HIGH = 100.0
    NO_OF_BANDS = 20
    TIME_STEP = 1.0 / 250

    CHANNELS=X.shape[0]

    singleBandWidth = (PASS_BAND_HIGH - PASS_BAND_LOW) / NO_OF_BANDS

    interestingBands = [x for x in range(NO_OF_BANDS)]
    allChannelBandResults = np.zeros((CHANNELS, len(interestingBands)))

    startPoints = []

    for x in range(100):
        startPoint = np.random.randint(0, SEQUENCE_LENGTH)
        while startPoint + SEQUENCE_LENGTH < X.shape[1]:
            startPoints.append(startPoint)
            startPoint += np.random.randint(0, SEQUENCE_LENGTH)

    featureVectors=np.zeros((len(startPoints),CHANNELS,NO_OF_BANDS))

    for s in range(len(startPoints)):
        for channel in range(CHANNELS):
            XX=X[channel,startPoints[s]:startPoints[s]+SEQUENCE_LENGTH]
            XX=XX-np.mean(XX)
            XX=XX/np.sqrt(np.var(XX))

            freqSpectrum = np.abs(np.fft.rfft(XX))
            freq=np.fft.rfftfreq(SEQUENCE_LENGTH,TIME_STEP)


            #print("Spetrum",freqSpectrum)
            #print("Frequencies",freq)

            endIndex=0
            startIndex=0
            while(freq[startIndex]<PASS_BAND_LOW):
                startIndex+=1
            while(freq[endIndex]<PASS_BAND_HIGH):
                endIndex+=1
            endIndex-=1

            freqSpectrum=np.abs((freqSpectrum[startIndex:endIndex]))
            freq=(freq[startIndex:endIndex])

            for f in range(len(freq)):
                if(freq[f]>0):
                    band=int((freq[f]-PASS_BAND_LOW)/singleBandWidth)
                    #print("band=",band)
                    featureVectors[s,channel,band]+=freqSpectrum[f]


    if(VERBOSE):print("Created feature vector of shape ",featureVectors.shape)
    return  featureVectors


def dnnForFFT(SampleFeatureVector):
    from keras.models import Sequential
    from keras.layers import Dense,Flatten
    model=Sequential()
    model.add(Flatten())
    model.add(Dense(20, input_shape=SampleFeatureVector.shape, activation="relu"))
    model.add(Dense(15,input_shape=SampleFeatureVector.shape,activation="relu"))
    model.add(Dense(10, input_shape=SampleFeatureVector.shape, activation="relu"))
    model.add(Dense(NO_CLASSES,activation="softmax"))
    return model

if __name__== "__main__":
    np.random.seed(0)
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

    if ALGORITHM=="fft-dnn":


        #model=dnnForFFT()

        for predictFileIndex in range(FILES_TO_PREDICT):
            fileNameToRead = sys.argv[argvIndex]
            argvIndex += 1

            Xtest = fileRead(fileNameToRead)
            Xtest=makeFeatureVectors(Xtest)

            Y=model.predict(Xtest)

            if (VERBOSE): print("The result is", Y)
            print("Answer: ", np.argmax(np.sum(Y, axis=0)))


    if ALGORITHM=="fft-svm":
        XtrainNew=[]
        YtrainNew=[]
        for i in range(len(Xtrain)):
            xNew=makeFeatureVectors(Xtrain[i])

            for x in range(xNew.shape[0]):
                XtrainNew.append(xNew[x,:,:])
                YtrainNew.append(Ytrain[i])
        Xtrain=XtrainNew[0].reshape((1,XtrainNew[0].shape[0],XtrainNew[0].shape[1]))

        for x in range(1,len(XtrainNew)):
            Xtrain=np.concatenate((Xtrain,XtrainNew[x].reshape((1,XtrainNew[x].shape[0],XtrainNew[x].shape[1]))))

        Ytrain = np.array(YtrainNew)

        Xold=Xtrain
        Yold=Ytrain

        Xtrain=np.ndarray(shape = (Xtrain.shape[0],Xtrain.shape[1] * Xtrain.shape[2]))
        Ytrain=np.ndarray(Ytrain.shape[0])

        permutation=np.arange(0,Xold.shape[0])
        np.random.shuffle(permutation)

        for i in range(permutation.shape[0]):
            Xtrain[i,:]=np.ndarray.flatten(Xold[permutation[i],:,:])
            Ytrain[i]=Yold[permutation[i]]


        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        clf = LDA(solver='lsqr')

        if(VERBOSE):
            print("Shapes of training data X,Y",Xtrain.shape,Ytrain.shape)
            a=input("Start training? [Press ENTER]")




        clf.fit(Xtrain, Ytrain)




        FILES_TO_PREDICT = int(sys.argv[argvIndex])
        argvIndex += 1

        for predictFileIndex in range(FILES_TO_PREDICT):
            fileNameToRead = sys.argv[argvIndex]
            argvIndex += 1

            Xtest = fileRead(fileNameToRead)
            Xtest=makeFeatureVectors(Xtest)
            Xtest=Xtest.reshape((Xtest.shape[0],Xtest.shape[1]*Xtest.shape[2]))
            Y=clf.predict(Xtest)

            if (VERBOSE): print("The result is", Y)
            print("Answer: ", np.mean(Y))






 #python Main.py rnn v 3 2 ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_R1.txt ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_R2.txt 2 ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_G1.txt ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_G2.txt 2 ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_B1.txt ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_B2.txt 3 ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_R3.txt ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_G3.txt ./SavedData/Jali_16_8_18/OpenBCI-RAW-S01_B3.txt