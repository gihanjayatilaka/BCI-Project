import matplotlib.pyplot as plt
import keras,numpy as np
import pyedflib
from keras.models import Sequential
from keras.layers import Dense, Conv3D, MaxPooling2D, Dropout, Flatten
from keras.datasets import mnist


def createModel(input_shape, nClasses):
    model = Sequential()

    model.add(Dense(20, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))

    return model

def convertTo2Ddataset(sequencialData, targetShape):
    features = len(sequencialData[0])
    samples = len(sequencialData)
    if (len(targetShape)!=2 or targetShape[0]*targetShape[1] != features):
        print("Bad parameters",features,targetShape)
        exit(-1)

    #fft

    #transform
    transformed = np.ndarray((samples,targetShape[0],targetShape[1]))
    for idx_sample in range(samples):
        transformed[idx_sample,:,:] = sequencialData[idx_sample,:].reshape(targetShape)
    return transformed

num_classes=2
x_train = np.random.random((1000, 100))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 100))
y_test = np.random.randint(2, size=(100, 1))


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model1 = createModel(x_train[0].shape, num_classes)
batch_size = 16
epochs = 100
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model1.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                     validation_data=(x_test, y_test))

model1.evaluate(x_test, y_test)

# Loss Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'], 'r', linewidth=3.0)
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)

# Accuracy Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['acc'], 'r', linewidth=3.0)
plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)

plt.show()