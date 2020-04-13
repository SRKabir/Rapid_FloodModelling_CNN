from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, LSTM, Activation, TimeDistributed
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np


def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    import os
    import rasterio as rio
    ###########SORT TRAINING DATA############
    Target = '/home/cvssk/Carlisle/Target/'
    inun_files2 = []

    ##PROCESS TARGET DATA (Y_PARAM)
    inun_files2 += [each for each in os.listdir(Target) if each.endswith('.wd')]
    inun_files2.sort()
    print('No. of all files:',len(inun_files2))
    
    ls = ['Run2-0000.wd', 'Run2-0001.wd', 'Run2-0002.wd', 'Run2-0003.wd', 'Run2-0004.wd', 'Run2-0005.wd', 'Run2-0006.wd', 'Run2-0007.wd',
      'Run3-0000.wd', 'Run3-0001.wd', 'Run3-0002.wd', 'Run3-0003.wd', 'Run3-0004.wd', 'Run3-0005.wd', 'Run3-0006.wd', 'Run3-0007.wd',
      'Run4-0000.wd', 'Run4-0001.wd', 'Run4-0002.wd', 'Run4-0003.wd', 'Run4-0004.wd', 'Run4-0005.wd', 'Run4-0006.wd', 'Run4-0007.wd',
      'Run5-0000.wd', 'Run5-0001.wd', 'Run5-0002.wd', 'Run5-0003.wd', 'Run5-0004.wd', 'Run5-0005.wd', 'Run5-0006.wd', 'Run5-0007.wd',
      'Run6-0000.wd', 'Run6-0001.wd', 'Run6-0002.wd', 'Run6-0003.wd', 'Run6-0004.wd', 'Run6-0005.wd', 'Run6-0006.wd', 'Run6-0007.wd']

    for i in ls:
        inun_files2.remove(i)

    print('No. of files after removing first 2 hours:', len(inun_files2))
    
    
    ###########sort target###############

    target = []

    for i in range(len(inun_files2)):
        data = rio.open(Target+inun_files2[i])
        band = data.read(1)
        value = band.flatten()
        target.append(value)

    Y = np.array(target)
    print('Target data shape:',Y.shape)

    ###################sort X params##################
    
    ####Import Precipitation/Discharge Data
    data_dir = '/home/cvssk/Carlisle/Flows/'

    dt =[]
    dt += [file for file in os.listdir(data_dir) if file.endswith('.csv')]

    dt.sort()
    print(dt)
    
    appended_data = []

    for f in dt:
        df = pd.read_csv(data_dir+f)
        ##Shift the x parameter values back to represent antacedent hydrometeorological values, i.e. t-1, t-2, t-3 etc
        df['Upstream1-1'] = df['Upstream1'].shift(1)
        df['Upstream1-2'] = df['Upstream1'].shift(2)
        df['Upstream1-3'] = df['Upstream1'].shift(3)
        df['Upstream1-4'] = df['Upstream1'].shift(4)
        df['Upstream1-5'] = df['Upstream1'].shift(5)
        df['Upstream1-6'] = df['Upstream1'].shift(6)
        df['Upstream1-7'] = df['Upstream1'].shift(7)
        df['Upstream1-8'] = df['Upstream1'].shift(8)

        df['Upstream2-1'] = df['Upstream2'].shift(1)
        df['Upstream2-2'] = df['Upstream2'].shift(2)
        df['Upstream2-3'] = df['Upstream2'].shift(3)
        df['Upstream2-4'] = df['Upstream2'].shift(4)
        df['Upstream2-5'] = df['Upstream2'].shift(5)
        df['Upstream2-6'] = df['Upstream2'].shift(6)
        df['Upstream2-7'] = df['Upstream2'].shift(7)
        df['Upstream2-8'] = df['Upstream2'].shift(8)    


        df['Upstream3-1'] = df['Upstream3'].shift(1)
        df['Upstream3-2'] = df['Upstream3'].shift(2)
        df['Upstream3-3'] = df['Upstream3'].shift(3)
        df['Upstream3-4'] = df['Upstream3'].shift(4)
        df['Upstream3-5'] = df['Upstream3'].shift(5)
        df['Upstream3-6'] = df['Upstream3'].shift(6)
        df['Upstream3-7'] = df['Upstream3'].shift(7)
        df['Upstream3-8'] = df['Upstream3'].shift(8)
    
        df = df.dropna()
    
        appended_data.append(df)

    appended_data = pd.concat(appended_data,ignore_index=True)
    
    print(len(appended_data))
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(appended_data)
    
    print('X data shape:', X.shape, 'Y data shape:',Y.shape)

    
    ##create train and test data
    ## replace all values less than 0.2m depth by 0
    Y[Y < 0.2] = 0
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

    x_train= X_train.reshape(X_train.shape[0],1, 1, X_train.shape[1])
    x_test= X_test.reshape(X_test.shape[0], 1, 1, X_test.shape[1])
    
    steps = x_train.shape[1]
    features = x_train.shape[3]
    outputs = y_train.shape[1]


    return x_train, y_train, x_test, y_test, steps, features, outputs


def create_model(x_train, y_train, x_test, y_test, steps, features, outputs):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """

    model = Sequential()
  
    model.add(TimeDistributed(Conv1D({{choice([16, 32, 64, 128, 256, 512])}}, kernel_size=1, activation='relu'), 
    input_shape=(None, steps, features)))
    
    model.add(TimeDistributed(Conv1D({{choice([16, 32, 64, 128, 256, 512])}}, kernel_size=1, activation='relu')))

    model.add(TimeDistributed(Flatten()))

    model.add(LSTM({{choice([16, 32, 64, 128, 256, 512])}}, activation='relu',return_sequences=True))

    model.add(LSTM({{choice([16, 32, 64, 128, 256, 512])}}, activation='relu',return_sequences=True))

    model.add(Dense(outputs))

    model.compile(loss='mse', metrics=['mse'], optimizer= {{choice(['rmsprop', 'adam', 'sgd'])}})
    
    result = model.fit(x_train, y_train,
              batch_size={{choice([10,20,30,40])}},
              epochs=10,
              verbose=2,
              validation_split=0.1)


    mse = np.amax(result.history['val_mse'])
    
    print('Best validation acc of epoch:', mse)
    return {'loss': mse, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':

    x_train, y_train, x_test, y_test, steps, features, outputs = data()

    best_run, best_model = optim.minimize(model= create_model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=3,
                                      trials=Trials(),
                                      eval_space=True)
    
    print(best_model.evaluate(x_test, y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

#######################################################################################
