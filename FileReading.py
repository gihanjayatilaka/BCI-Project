import sys
from scipy.fftpack import rfft
import numpy as np
#import matplotlib.pyplot as plt


def fileRead(fileName,lineToRemove,leftColToRemove,rightColToRemove):
    file  = open(fileName, 'r')
    fullText=file.read()
    lines=fullText.split('\n')

    print('Read '+str(len(lines))+' lines')

    for x in range(lineToRemove):
        lines.pop(0)


    ar=[]
    for x in range(len(lines)):
        temp=[]
        fields=lines[x].split(",")
        for xx in range(len(fields)):
            fields[xx]=fields[xx].strip()



        for y in range(leftColToRemove,len(fields)-rightColToRemove):
            #print('converting to float >>>'+fields[y])
            temp.append(float(fields[y]))
        ar.append(temp)

    npar=np.array(ar).transpose()

    return npar


def readFiles():
    NO_OF_CHANNELS=8
    interestingBands=[0,20,30,40,50,60,70,80,90,100]
    allChannelBandResults=np.ndarray((NO_OF_CHANNELS,len(interestingBands)))

    fileNames=['TrainingData/OpenBCI-RAW-2018-06-03_13-06-27.txt']
    Y=[1]
    for file in range(len(fileNames)):
        ar=fileRead(fileNames[file],4,1,4)#complete



        for chan in range(len(ar[0])):
            bandResults = np.ndarray(len(interestingBands))
            bandCount = np.ndarray(len(interestingBands))

            freqSpectrum=np.fft.fft(ar[chan])
            timeStep=1.0/250
            n=len(ar[chan])
            freq=np.fft.fftfreq(n,d=timeStep)

            for f in range(len(freq)):
                if(freq[f]>0):
                    band=0
                    for bb in range(len(interestingBands)):
                        if interestingBands[bb]>freq[f]:
                            band=bb-1
                            break

                    bandResults[band]+=np.abs(freqSpectrum[f])
                    bandCount[band]+=1

            for x in range(len(bandResults)):
                if bandCount[x]<1:
                    bandResults[x]=0
                else:
                    bandResults[x]=bandResults[x]/(1.0*bandCount[x])

            allChannelBandResults[chan]=bandResults

        print(allChannelBandResults)
        '''#put the plot code here
        fig = plt.figure()

        for i in range(NO_OF_CHANNELS):
            plt.plot(interestingBands, bandResults[i,:])
        plt.show()
        '''









readFiles()


#fileRead(sys.argv[1])
