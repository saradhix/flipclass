import numpy as np
from numpy import arange
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
import time

print("Fetching MNIST data. May take time for the first time")
mnist = fetch_mldata("MNIST Original")

print("Shape of the total dataset is ", mnist.data.shape)
#The dataset is a collection of 70000 images.
#Each image is represented by a 28x28 matrix of grey pixels

#Lets manually split the data in 60:40 for training/testing
n_train = 42000
n_test = 28000
train_idx = arange(0,n_train)
test_idx = arange(n_train,n_train+n_test)

X_train, y_train = mnist.data[train_idx], mnist.target[train_idx]
X_test, y_test = mnist.data[test_idx], mnist.target[test_idx]

print("Shape of X_train is ", X_train.shape)
print("It means it has %d samples and %d features/sample" % (X_train.shape[0], X_train.shape[1]))
print("Shape of X_test=", X_test.shape)

print("Shape of y_train=", y_train.shape, "Shape of y_test=", y_test.shape)
#Try all values of k between 1 and 7 and see which k value gives the
#"best" result
for n in range(1,8):
  print("Applying KNN algorithm with neighbours:", n)
  start_time = time.time()
  clf = KNeighborsClassifier(n_neighbors=n)
  clf.fit(X_train, y_train)
  print("Making predictions...May take quite some time..Please be patient")
  y_pred = clf.predict(X_test)
# Creating confusion matrix
  conf_matrix = confusion_matrix(y_test, y_pred)
  print(conf_matrix)
# Evaluate the prediction
  print("Evaluating results...")
#Print classification report
  print(classification_report(y_test, y_pred))
  end_time = time.time()
  print("Running time for KNN:", end_time - start_time, "seconds")
