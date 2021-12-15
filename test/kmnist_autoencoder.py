from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict,cross_val_score, RandomizedSearchCV, GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import normalize
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic, WhiteKernel, Matern
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
import joblib
import matplotlib.pylab as pylab
import matplotlib as mpl
from sklearn.linear_model import RandomizedLasso
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

#pandas
import pandas as pd
import math
from math import sqrt
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import sys
from enum import Enum

import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras import datasets, layers, models
import tensorflow as tf
from tensorflow import keras

#https://www.kaggle.com/aakashnain/kmnist-mnist-replacement
# Let us define some paths first
input_path = "../datasets/data/kmnist/"

# Path to training images and corresponding labels provided as numpy arrays
kmnist_train_images_path = input_path+"kmnist-train-imgs.npz"
kmnist_train_labels_path = input_path+"kmnist-train-labels.npz"

# Path to the test images and corresponding labels
kmnist_test_images_path = input_path+"kmnist-test-imgs.npz"
kmnist_test_labels_path = input_path+"kmnist-test-labels.npz"

# Load the training data from the corresponding npz files
kmnist_train_images = np.load(kmnist_train_images_path)['arr_0']
kmnist_train_labels = np.load(kmnist_train_labels_path)['arr_0']

# Load the test data from the corresponding npz files
kmnist_test_images = np.load(kmnist_test_images_path)['arr_0']
kmnist_test_labels = np.load(kmnist_test_labels_path)['arr_0']

print("Number of training samples: {} where each sample is of size: {}".format(
    len(kmnist_train_images), kmnist_train_images.shape[1:] ))
print("Number of test samples: {} where each sample is of size: {}".format(
    len(kmnist_test_images), kmnist_test_images.shape[1:]))

X_train = kmnist_train_images.reshape(60000, 1, 28, 28)
X_test = kmnist_test_images.reshape(10000, 1, 28, 28)
Y_train = kmnist_train_labels
Y_test = kmnist_test_labels
print("x_train shape:", X_train.shape, "y_train shape:", Y_train.shape)


# read training and test data from local disk
data_dir = "/home/peng/k_mnist/"
# read training data
number_of_samples = 60000

y_train_data = np.loadtxt(data_dir + "y_train.txt")
y_train = y_train_data.reshape(number_of_samples, 1)
y_train = pd.DataFrame(data=y_train, index=None, columns=None)

# read testing data
number_of_samples = 10000
#x_test_all = np.loadtxt(data_dir + "x_data_test.txt").reshape(number_of_samples, 1, 28, 28)
#x_data = [x.flatten() for x in x_data ]
#x_test_all = pd.DataFrame(data=x_data, index=None, columns=None)   

y_test_data = np.loadtxt(data_dir + "y_test.txt")
y_test = y_test_data.reshape(number_of_samples, 1)
y_test = pd.DataFrame(data=y_test, index=None, columns=None)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

x_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:]))) /255.0
x_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:]))) /255.0

x_train_all = []
y_train_all = []

x_test_all = []
y_test_all = []


for i in range(10):
    x_train_all.append(x_train[Y_train == i])
    y_train_all.append(y_train[Y_train == i])
    x_test_all.append(x_test[Y_test == i])
    y_test_all.append(y_test[Y_test == i])
    
simplest_index = [579, 274, 547, 824, 273, 760, 569, 86, 505, 610]
simplest_index = [396, 909, 471, 937, 799, 437, 645, 847, 772, 227]

# training data for autoencoder..
x_train_easy = []
x_train_hard = []
x_test_easy = []
x_test_hard = []
x_tmptest_easy = []


num_classes = 10

# use only half of the output classes to train one autoencoder. 
# first half : range(int(num_classes/2))
# second half : range(int(num_classes/2), num_classes)

y_trndata_easy = []
y_trndata_hard = []
y_testdata_easy = []
y_testdata_hard = []

#for i in range(int(num_classes/2), num_classes):
for i in range(int(num_classes)):

    trn_easy = x_train_all[i][y_train_all[i][0]==0]
    trn_hard = x_train_all[i][y_train_all[i][0]>0]
    
    print(i, trn_easy.shape, trn_hard.shape)

    test_easy = x_test_all[i][y_test_all[i][0]==0]
    test_hard = x_test_all[i][y_test_all[i][0]>0]
    test_easy_tmp = x_test_all[i][y_test_all[i][0]==0]
    
    # there are more easy examples than hard examples
    n1 = trn_hard.shape[0]
    n2 = trn_easy.shape[0]
    #trn_easy = trn_easy[0:n1]
    #trn_hard = trn_hard[0:n1]
    
    while 2*n1 > n2:
        trn_easy = np.concatenate((trn_easy, trn_easy), axis=0)
        n2 = trn_easy.shape[0]
        
    trn_hard = np.concatenate((trn_hard[0:n1],  trn_easy[n1:n2]), axis=0)
    
    
    n3 = test_hard.shape[0]
    n4 = test_easy.shape[0]
    test_hard = np.concatenate((test_hard[0:n3], test_easy[n3:n4]), axis=0)
    #test_easy = test_easy[0:n2]
    #test_hard = test_hard[0:n2]
    #trn_easy = np.concatenate((trn_hard[0:n1], trn_easy), axis=0)
    
    for j in range(0,n2):
        #trn_easy[j] = trn_easy[-1] 
        trn_easy[j] = x_train_all[i][simplest_index[i]]
    #test_easy = np.concatenate((test_hard[0:n3], test_easy), axis=0)  
    
    for j in range(0,n4):
        #test_easy[j] = trn_easy[-1]
        test_easy[j] = x_train_all[i][simplest_index[i]]
    x_train_easy.append(trn_easy)
    x_train_hard.append(trn_hard)
    
    
    # testing data
    x_test_easy.append(test_easy)
    x_test_hard.append(test_hard)
    y_trndata_hard += [i for _ in (y_train_all[i][0]>0) if _ is True]
    y_testdata_easy += [i for _ in (y_test_all[i][0]==0) if _ is True]
    #y_testdata_easy = [i for _ in (y_test_all[i][0]>0) if _ is True] + y_testdata_easy

    #y_testdata_hard += [i for _ in (y_test_all[i][0]>0) if _ is True]

x_trndata_easy = np.concatenate(x_train_easy)
x_trndata_hard = np.concatenate(x_train_hard)
x_testdata_easy = np.concatenate(x_test_easy)
x_testdata_hard = np.concatenate(x_test_hard)

print (x_trndata_easy.shape, x_trndata_hard.shape, x_testdata_easy.shape, x_testdata_hard.shape, len(y_testdata_easy))

for i in range(5):
    print (i, x_train_easy[i].shape[0])
    print (i, x_train_hard[i].shape[0])


train_X, train_ground, valid_X, valid_ground = x_trndata_hard, x_trndata_easy, x_testdata_hard,  x_testdata_easy
print(train_X.shape, train_ground.shape, valid_X.shape, valid_ground.shape)



from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import datasets, layers, models, regularizers
import tensorflow as tf
from tensorflow import keras

def autoencoder_model(activation1, activation2, optimizer, unit1, unit2, unit3):
    input_img = keras.Input(shape=(784,))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(unit1, activation=activation1, activity_regularizer=regularizers.l1(10e-8))(input_img)
    encoded = layers.Dense(unit2, activation=activation1, activity_regularizer=regularizers.l1(10e-8))(encoded)
    encoded = layers.Dense(unit3, activation=activation2, activity_regularizer=regularizers.l1(10e-8))(encoded)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(784, activation='sigmoid')(encoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)

    autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])
    return autoencoder

data_shape = (X_test.shape[0], 784)

model = KerasClassifier(build_fn=autoencoder_model, batch_size=256, epochs=20)
#now write out all the parameters you want to try out for the grid search
activation1 = ['relu', 'linear']
activation2 = ['relu', 'linear']

#units = [784, 512, 384, 256, 128, 64, 32,]
unit1 = [784, 512]
unit2 = [384, 256]
unit3 = [128, 64, 32]
optimizer = ['Adam']
param_grid = dict(activation1=activation1,activation2=activation2,
                  optimizer=optimizer, unit1=unit1, unit2=unit2, unit3=unit3)

grid = GridSearchCV(estimator=model, param_grid=param_grid)
result = grid.fit(train_X, train_ground, validation_data=(valid_X,valid_ground))

# summarize results
print("Best: %f using %s" % (result.best_score_, result.best_params_))
means = result.cv_results_['mean_test_score']
stds = result.cv_results_['std_test_score']
params = result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
