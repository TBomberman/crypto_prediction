
from __future__ import print_function
import numpy as np
import pandas as pd
import data_driven as dd
import mach_learn as ml
from optimizer import ga_optimizer
from scipy.stats import linregress
import matplotlib.pyplot as plt

import logging
logging.basicConfig(filename='crypto_predict.log', level=logging.INFO)
np.random.seed(1234)  # for reproducibility

import time
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, InputLayer
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import losses, callbacks, regularizers
from random import sample
from scipy.ndimage import imread
import glob
import os
import sys
import csv


from tensorflow.contrib import learn
from tensorflow.contrib.learn.python import SKCompat
from sklearn.metrics import mean_squared_error

from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lstm_predictor import load_csvdata, lstm_model
from utilities import all_stats

from random import sample

tbCallBck = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
earlyStopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10000, verbose=1, mode='auto')

def ensure_number(data):
    data[np.where(data == np.NAN), :] = 0

cutoff = 0
look_back = 30
lag = 1
# read data as dataframe
#data_frame = pd.read_csv('Data/Candles15mJan2015-Jul2018.csv')
data_frame = pd.read_csv('Data/Bitcoin_DailyPrice_02-01-2018.csv',sep=";")

# data set up
#dataTable = data_frame[[' Open', ' Close', ' Vol']].as_matrix()[1:]
dataTable = data_frame['Close'].as_matrix()[:-1] - data_frame['Close'].as_matrix()[1:]
#dataTable = data_frame[' Close'].as_matrix()[1:] - data_frame[' Close'].as_matrix()[:-1]

#Normilize by volume

labels = (data_frame['Close'].as_matrix()[:-1] - data_frame['Close'].as_matrix()[1:]).reshape(len(data_frame['Open'])-1,1)
#labels = data_frame['Buy Sell'].as_matrix().reshape(len(data_frame['Buy Sell']),1)
labels = labels[0:labels.shape[0]-(look_back + lag),]

data = np.atleast_2d(np.array([dataTable[start:start + look_back] for start in range(lag, dataTable.shape[0] - (look_back))]))
#data = np.array([dataTable[start:start + look_back] for start in range(0, dataTable.shape[0] - look_back)])

sklearn = True
keras = False
lstm = False

# turn labels into binary
y = np.zeros((len(labels), 1)).astype(int)
pos_id = np.where(labels > cutoff)[0]
y[pos_id] = 1

# validate every data is a number
ensure_number(data)
ensure_number(labels)

test_fraction = 0.3
cut = int(len(labels)*test_fraction)
ids = np.arange(len(labels))
np.random.shuffle(ids)
test_id = ids[0:cut]
train_id = ids[cut:]

data = (data - np.min(data,axis=0))/(np.max(data,axis=0) - np.min(data,axis=0) )

if sklearn:
    partition=dd.Partition(trainset=train_id,testset=test_id)
    model = dd.Model(data=np.hstack((data,y)), function=ml.RandF(parameters={'n_estimators':250,'min_samples_split':50}), partition=partition, nfo=3)
    #model = dd.Model(data=np.hstack((data, y)),function=ml.Gproc(parameters=({'theta0':1e-2,'thetaL':1e-4,'thetaU':1e-1})), partition=partition,nfo=3)
    #model = dd.Model(data=np.hstack((data,y)),function=ml.Sksvm(parameters={'regularization':100,'sigma':10}),partition=partition)
    model.training()
    model.crossvalidating()
    model.testing()
    model.performance()
    model.summary()

    #model.optimize(ga_optimizer)

if keras:

    nb_classes = 2
    nb_epoch = 500
    dropout = 0.4  # 0.2
    hidden = 10  # 80
    dense = 10
    batch_size = 128

    X_train = data[train_id,:].astype('float32')
    X_test = data[test_id,:].astype('float32')

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')

    # convert class vectors to binary class matrices

    Y_train = np_utils.to_categorical(y[train_id], nb_classes)
    Y_test = np_utils.to_categorical(y[test_id], nb_classes)

    model = Sequential()
    model.add(Dense(dense, input_shape=(X_train.shape[1],)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    for i in range(4):
        model.add(Dense(dense))
        model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True, callbacks=[tbCallBck, earlyStopping])

    score = model.evaluate(X_test, Y_test, verbose=0)
    y_score_train = model.predict_proba(X_train)
    y_score_test = model.predict_proba(X_test)


    #print (y_score_train,y_train)
    #print (y_score_test,y_test)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    #print(y_score_train[:,1],Y_train[:,1])
    #print(y_score_test[:,1],Y_test[:,1])
    train_stats = all_stats(Y_train[:,1],y_score_train[:,1])
    test_stats = all_stats(Y_test[:,1],y_score_test[:,1],train_stats[-1])

    print('All stats train:',['{:6.2f}'.format(val) for val in train_stats])
    print('All stats test:',['{:6.2f}'.format(val) for val in test_stats])


if lstm:
    TIMESTEPS = 1
    RNN_LAYERS = [{'num_units': 400}]
    DENSE_LAYERS =  [10, 10]
    TRAINING_STEPS = 5000
    PRINT_STEPS = TRAINING_STEPS  # / 10
    BATCH_SIZE = 100

    X={}
    Y={}

    X, Y = load_csvdata(data_frame['Close'] - data_frame['Open'], TIMESTEPS, seperate=False)

    regressor = SKCompat(learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS), ))

    # create a lstm instance and validation monitor
    validation_monitor = learn.monitors.ValidationMonitor(X['test'], Y['test'], )
    # every_n_steps=PRINT_STEPS,)
    # early_stopping_rounds=1000)
    # print(X['train'])
    # print(y['train'])

    SKCompat(regressor.fit(X['train'], Y['train'],
                           monitors=[validation_monitor],
                           batch_size=BATCH_SIZE,
                           steps=TRAINING_STEPS))

    print('X train shape', X['train'].shape)
    print('y train shape', Y['train'].shape)

    print('X test shape', X['test'].shape)
    print('y test shape', Y['test'].shape)
    y_predicted = {}
    predicted = np.asmatrix(regressor.predict(X['train']), dtype=np.float32)  # ,as_iterable=False))
    predicted = np.transpose(predicted)
    y_predicted['train'] = predicted

    predicted = np.asmatrix(regressor.predict(X['test']), dtype=np.float32)  # ,as_iterable=False))
    predicted = np.transpose(predicted)
    y_predicted['test'] = predicted

    print(y_predicted['train'].shape)

    plt.scatter(Y['train'],y_predicted['train'])
    plt.show()
    plt.scatter(Y['test'], y_predicted['test'])
    plt.show()

    print('All stats train:',['{:6.2f}'.format(val) for val in linregress(Y['train'],y_predicted['train'])])
    print('All stats test:',['{:6.2f}'.format(val) for val in linregress(Y['train'], y_predicted['train'])])