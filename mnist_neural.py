import numpy as np
from numpy import arange
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
import time
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

print("Fetching MNIST data. May take time for the first time")
mnist = fetch_mldata("MNIST Original")

print("Shape of the total dataset is ", mnist.data.shape)
#The dataset is a collection of 70000 images.
#Each image is represented by a 28x28 matrix of grey pixels

#Lets manually split the data in 60:40 for training/testing
n_train = 60000
n_test = 10000
train_idx = arange(0,n_train)
test_idx = arange(n_train,n_train+n_test)

X_train, y_train = mnist.data[train_idx], mnist.target[train_idx]
X_test, y_test = mnist.data[test_idx], mnist.target[test_idx]

print("Shape of X_train is ", X_train.shape)
print("It means it has %d samples and %d features/sample" % (X_train.shape[0], X_train.shape[1]))
print("Shape of X_test=", X_test.shape)

print("Shape of y_train=", y_train.shape, "Shape of y_test=", y_test.shape)
# one-hot encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

# building a linear stack of layers with the sequential model
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('sigmoid'))                            
#model.add(Dropout(0.2))

#model.add(Dense(512))
#model.add(Activation('sigmoid'))
#model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()
# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(X_train, Y_train, batch_size=128, epochs=15, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, batch_size=128)
print("Score=",score)
