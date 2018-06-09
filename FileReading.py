import sys
from scipy.fftpack import rfft,fftfreq
import numpy as np
import matplotlib.pyplot as plt


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
        if len(temp)==0:
            continue
        ar.append(temp)
        print('Reading line '+str(x)+' \r')
        sys.stdout.flush()

    npar=np.ndarray((len(ar), len(ar[0])))
    for i in range(len(ar)):
        for j in range(len(ar[0])):
            npar[i][j] = ar[i][j]
    return npar.transpose()

def nextpow2(i):
    """
    Find the next power of 2 for number i
    """
    n = 1
    while n < i:
        n *= 2
    return n


def readFiles(fileNameList):
    NO_OF_CHANNELS=8

    PASS_BAND_LOW=3.0
    PASS_BAND_HIGH=50.0
    NO_OF_BANDS=5

    singleBandWidth=(PASS_BAND_HIGH-PASS_BAND_LOW)/NO_OF_BANDS

    interestingBands=[x for x in range(NO_OF_BANDS)]
    allChannelBandResults=np.ndarray((NO_OF_CHANNELS,len(interestingBands)))

    fileNames=fileNameList#["openBCI_2013-12-24_meditation.txt"]
    Y=[1]
    for file in range(len(fileNames)):
        ar=fileRead(fileNames[file],4,1,3)#complete

        for chan in range(len(ar)):
            bandResults = np.ndarray(len(interestingBands))
            bandCount = np.ndarray(len(interestingBands))

            freqSpectrum=rfft(ar[chan,:])
            timeStep=1.0/250
            n=len(ar[chan])
            freq=fftfreq(n,d=timeStep)

            endIndex=0
            startIndex=0
            while(freq[startIndex]<PASS_BAND_LOW):
                startIndex+=1
            while(freq[endIndex]<PASS_BAND_HIGH):
                endIndex+=1
            endIndex-=1

            freqSpectrum=np.abs((freqSpectrum[startIndex:endIndex]))
            freq=(freq[startIndex:endIndex])
            '''plt.figure()
            plt.plot(freq, freqSpectrum)'''
            #plt.plot(range(len(freq)),freq)
            for f in range(len(freq)):
                if(freq[f]>0):
                    bandResults[int((freq[f]-PASS_BAND_LOW)/singleBandWidth)]+=freqSpectrum[f]
                    '''band=0
                    for bb in range(len(interestingBands)):
                        if interestingBands[bb]>freq[f]:
                            band=bb-1
                            break

                    bandResults[band]+=np.abs(freqSpectrum[f])
                    bandCount[band]+=1'''
            '''
            for x in range(len(bandResults)):
                if bandCount[x]<1:
                    bandResults[x]=0
                else:
                    bandResults[x]=bandResults[x]/(1.0*bandCount[x])'''

            allChannelBandResults[chan]=bandResults
            print('channel ',chan+1,'of ',len(allChannelBandResults),' channels completed')

        #print(allChannelBandResults)
        #put the plot code here


        for i in range(NO_OF_CHANNELS):
            plt.figure()
            plt.plot(interestingBands, allChannelBandResults[i,:])
        plt.show()

        plt.show()










readFiles(sys.argv[1:])


#fileRead(sys.argv[1])
