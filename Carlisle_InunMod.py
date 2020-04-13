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
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import timeit
import os
import rasterio as rio

print(tf.__version__)


def data_pre_process():

    print('Running preprocessing script...')
    ###########SORT TRAINING DATA############



    Target = '/home/cvssk/Carlisle/Target/'
    inun_files2 = []

    ##PROCESS TARGET DATA (Y_PARAM)
    inun_files2 += [each for each in os.listdir(Target) if each.endswith('.wd')]
    inun_files2.sort()



    ls = ['Run2-0000.wd', 'Run2-0001.wd', 'Run2-0002.wd', 'Run2-0003.wd', 'Run2-0004.wd', 'Run2-0005.wd', 'Run2-0006.wd', 'Run2-0007.wd',
      'Run3-0000.wd', 'Run3-0001.wd', 'Run3-0002.wd', 'Run3-0003.wd', 'Run3-0004.wd', 'Run3-0005.wd', 'Run3-0006.wd', 'Run3-0007.wd',
      'Run4-0000.wd', 'Run4-0001.wd', 'Run4-0002.wd', 'Run4-0003.wd', 'Run4-0004.wd', 'Run4-0005.wd', 'Run4-0006.wd', 'Run4-0007.wd',
      'Run5-0000.wd', 'Run5-0001.wd', 'Run5-0002.wd', 'Run5-0003.wd', 'Run5-0004.wd', 'Run5-0005.wd', 'Run5-0006.wd', 'Run5-0007.wd',
      'Run6-0000.wd', 'Run6-0001.wd', 'Run6-0002.wd', 'Run6-0003.wd', 'Run6-0004.wd', 'Run6-0005.wd', 'Run6-0006.wd', 'Run6-0007.wd']

    for i in ls:
        inun_files2.remove(i)


    ###########sort target###############

    target = []

    for i in range(len(inun_files2)):
        data = rio.open(Target+inun_files2[i])
        band = data.read(1)
        value = band.flatten()
        target.append(value)

    Y = np.array(target)
    Y[Y<0.2] = 0

    ##PROCESS TEST TARGET DATA
    directory1 = '/home/cvssk/Carlisle/Run1/' #(dIRECTORY OF TEST DATA)
    inun_files = []

    inun_files += [each for each in os.listdir(directory1) if each.endswith('.wd')]
    inun_files.sort()


    l = ['Run1-0000.wd', 'Run1-0001.wd', 'Run1-0002.wd', 'Run1-0003.wd', 'Run1-0004.wd', 'Run1-0005.wd', 'Run1-0006.wd', 'Run1-0007.wd']

    for i in l:
        inun_files.remove(i)


    ###########sort test target###############

    test_target = []

    for i in range(len(inun_files)):
        data = rio.open(directory1+inun_files[i])
        band = data.read(1)
        value = band.flatten()
        test_target.append(value)

    Y_test = np.array(test_target)
    Y_test[Y_test<0.2] = 0
    print(Y_test.shape)

    #########################PREPARE X-PARAM DATA

    ####Import Precipitation/Discharge Data
    data_dir = '/home/cvssk/Carlisle/Flows/'

    data =[]
    data += [file for file in os.listdir(data_dir) if file.endswith('.csv')]

    data.sort()
    print('Flow data files:',data)

    appended_data = []

    for f in data:
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

    appended_data.to_csv('/home/cvssk/Carlisle/Flows/Train/appended.csv')

    ############Prepare test X_Param

    ####Import Precipitation-Discharge Data
    df = pd.read_csv('/home/cvssk/Carlisle/Flows/Test/Upstream_Flows_Run1.csv')


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

    all_data = pd.concat([appended_data, df],ignore_index=True)

    print('Length of the data:',len(all_data))


    scaler = MinMaxScaler(feature_range=(0, 1))
    all_data = scaler.fit_transform(all_data)

    X_Train = all_data[0:1243, :]
    X_Test = all_data[1243:, :]

    x_train= X_Train.reshape(X_Train.shape[0], 1, X_Train.shape[1])
    x_test= X_Test.reshape(X_Test.shape[0], 1, X_Test.shape[1])
    
    
    steps = x_train.shape[1]
    features = x_train.shape[2]
    outputs = Y.shape[1]

    del target
    del test_target
    del inun_files
    del inun_files2
    del appended_data
    del df
    del all_data

    print('X Train shape:', x_train.shape, 'X Test shape:', x_test.shape, 'Train target shape:',Y.shape, 'Test target shape:',Y_test.shape)
    print('Data preprocessing complete!')
    return x_train, Y, x_test, Y_test, steps, features, outputs, X_Test

def CNN_Model_2lr(x_train, Y, x_test, Y_test, steps, features, outputs):
  '''
  Two layered conv network
  '''
  print('Running the CNN model...')
  model = Sequential()
  model.add(Conv1D(32, kernel_size=1, activation='relu', input_shape=(steps, features)))
  model.add(Conv1D(128, kernel_size=1, activation='relu'))
  model.add(Flatten())
  model.add(Dense(32, activation='relu'))
  model.add(Dense(256,activation='relu'))
  model.add(Dense(512,activation='relu'))
  model.add(Dense(outputs))
  model.compile(loss='mse', metrics=['mse'], optimizer= 'adam')
  print(model.summary())
  ##start time
  start = timeit.default_timer()
  monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

  history = model.fit(x_train,Y,validation_data=(x_test,Y_test),batch_size=10,callbacks=[monitor],verbose=0,epochs=100)
  
  #stop time
  stop = timeit.default_timer()

  print('Time: ', stop - start) 

  # plot history
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.legend()
  plt.show()

  return model

def save_model(model, name):
  # Save the weights
  model.save_weights(name+'.h5')

  # Save the model architecture
  with open(name+'.json', 'w') as f:
    f.write(model.to_json())

def load_model(name):
  ##Loading the model weights
  from tensorflow.keras.models import model_from_json

  # Model reconstruction from JSON file
  with open(name+'.json', 'r') as f:
      model = model_from_json(f.read())

  # Load weights into the new model
  model.load_weights(name+'.h5')

  return model

def predict(model,X_Test, index):
  '''
  index (string) = name of the model. e.g., cnn, cnn_1lr, lstm, cnn_lstm etc
  '''
  tar_dir = '/home/cvssk/Carlisle/CNN_Outputs/' # change this directory for targets according to model

  #directory1 = '/home/cvssk/Carlisle/Run1/' #(dIRECTORY OF TEST DATA)
  #inun_files = []

  #inun_files += [each for each in os.listdir(directory1) if each.endswith('.wd')]
  #inun_files.sort()


  #l = ['Run1-0000.wd', 'Run1-0001.wd', 'Run1-0002.wd', 'Run1-0003.wd', 'Run1-0004.wd', 'Run1-0005.wd', 'Run1-0006.wd', 'Run1-0007.wd']

  #for i in l:
  #  inun_files.remove(i)

  ##Make predictions
  for i in range(len(X_Test)):
    x_test = X_Test[i]
    x_test = x_test.reshape(1,1,X_Test.shape[1])
    y_pred = model.predict(x_test)
    
    data = rio.open('/home/cvssk/Carlisle/RapidCNN_Inun/Data/Carlisle_5m.asc') #reference image for fixing raster dimensions
    y_pred.resize(data.height, data.width)
    y_pred[y_pred<0.2] = 0
    
    ##file naming
    #fname = inun_files[i].replace('.wd','')
    # Register GDAL format drivers and configuration options with a
    # context manager.

    src = data
    with rio.Env():
        # Write an array as a raster band to a new 8-bit file. For
        # the new file's profile, we start with the profile of the source
        profile = src.profile

        # And then change the band count to 1, set the
        # dtype to uint8, and specify LZW compression.
        profile.update(dtype=str(y_pred.dtype), count=1,compress='lzw')

        with rio.open(tar_dir+index+'_step_{}'".tif".format(i+8), 'w', **profile) as dst:
        #with rio.open(tar_dir+fname+index+'.tif', 'w', **profile) as dst:
            dst.write(np.absolute(y_pred), 1)

def CNN_Model(x_train, Y, x_test, Y_test, steps, features, outputs):
  '''
  1 layered conv network, 3 dense
  '''
  print('Running the CNN model...')
  model = Sequential()
  model.add(Conv1D(512, kernel_size=1, activation='relu', input_shape=(steps, features)))
  model.add(Flatten())
  model.add(Dense(64,activation='relu'))
  model.add(Dense(256,activation='relu'))
  model.add(Dense(outputs))
  model.compile(loss='mse', metrics=['mse'], optimizer= 'rmsprop')
  print(model.summary())
  ##start time
  start = timeit.default_timer()
  monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

  history = model.fit(x_train,Y,validation_data=(x_test,Y_test),batch_size=20,callbacks=[monitor],verbose=0,epochs=100)
  
  #stop time
  stop = timeit.default_timer()

  print('Time: ', stop - start) 

  # plot history
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.legend()
  plt.show()

  return model

def CNN_LSTM_Model(x_train, Y, x_test, Y_test, outputs):
  x_tr=x_train.reshape(x_train.shape[0],1, x_train.shape[1], x_train.shape[2])
  x_ts=x_test.reshape(x_test.shape[0],1,x_test.shape[1], x_test.shape[2])
  
  # define the CNN-LSTM model
  model = Sequential()
  model.add(TimeDistributed(Conv1D(filters=512, kernel_size=1, activation='relu'), input_shape=(None, x_tr.shape[1] , x_tr.shape[3])))

  model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu')))
                          
  model.add(TimeDistributed(Flatten()))

  model.add(LSTM(256,activation='relu', return_sequences=True))
  
  model.add(LSTM(32,activation='relu', return_sequences=True))

  model.add(Dense(outputs))

  model.compile(loss='mse', optimizer='rmsprop')

  print(model.summary())

  ##start time
  start = timeit.default_timer()

  monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

  history = model.fit(x_tr,Y,validation_data=(x_ts,Y_test),batch_size=30,callbacks=[monitor],verbose=0,epochs=100)

  #stop time
  stop = timeit.default_timer()

  print('Time: ', stop - start)

  # plot history
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.legend()
  plt.show()

  return model

def LSTM_Model(x_train, Y, x_test, Y_test, features, outputs):
  #########LSTM#############

  model = Sequential()

  model.add(LSTM(256, activation='relu',return_sequences=True, input_shape=(None, features)))

  model.add(LSTM(32, return_sequences=True))

  model.add(Dense(outputs))

  model.compile(loss='mse', optimizer='rmsprop')

  print(model.summary())

  ##start time
  start = timeit.default_timer()

  monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

  history = model.fit(x_train,Y,validation_data=(x_test,Y_test),batch_size=30,callbacks=[monitor],verbose=0,epochs=100)

  #stop time
  stop = timeit.default_timer()

  print('Time: ', stop - start)
  
  # plot history
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.legend()
  plt.show()

  return model 

def predict_cnnlstm(model,X_Test, index):
  '''
  index (string) = name of the model. e.g., cnn, cnn_1lr, lstm, cnn_lstm etc
  '''
  tar_dir = '/home/cvssk/Carlisle/CNN_LSTM_Outputs/' # change this directory for targets according to model

  #directory1 = '/home/cvssk/Carlisle/Run1/' #(dIRECTORY OF TEST DATA)
  #inun_files = []

  #inun_files += [each for each in os.listdir(directory1) if each.endswith('.wd')]
  #inun_files.sort()


  #l = ['Run1-0000.wd', 'Run1-0001.wd', 'Run1-0002.wd', 'Run1-0003.wd', 'Run1-0004.wd', 'Run1-0005.wd', 'Run1-0006.wd', 'Run1-0007.wd']

  #for i in l:
  #  inun_files.remove(i)

  ##Make predictions
  for i in range(len(X_Test)):
    x_ts = X_Test[i]
    x_ts = x_ts.reshape(1,1,1, X_Test.shape[1])
    y_pred = model.predict(x_ts)
    
    data = rio.open('/home/cvssk/Carlisle/Carlisle_5m.asc') #reference image for fixing raster dimensions
    y_pred.resize(data.height, data.width)
    y_pred[y_pred<0.2] = 0
    
    ##file naming
    #fname = inun_files[i].replace('.wd','')
    # Register GDAL format drivers and configuration options with a
    # context manager.

    src = data
    with rio.Env():
        # Write an array as a raster band to a new 8-bit file. For
        # the new file's profile, we start with the profile of the source
        profile = src.profile

        # And then change the band count to 1, set the
        # dtype to uint8, and specify LZW compression.
        profile.update(dtype=str(y_pred.dtype), count=1,compress='lzw')

        with rio.open(tar_dir+index+'_step_{}'".tif".format(i+4), 'w', **profile) as dst:
        #with rio.open(tar_dir+fname+index+'.tif', 'w', **profile) as dst:
            dst.write(np.absolute(y_pred), 1)

##### extract values at the validation points

def export_ref_data(locations, ind):
  import geopandas as gpd

  # Read points from shapefile
  pts = gpd.read_file(locations)
  pts = pts[['X', 'Y', 'Descriptio','geometry']]
  pts.index = range(len(pts))
  coords = [(x,y) for x, y in zip(pts.X, pts.Y)]

  directory1 = '/home/cvssk/Carlisle/Run1/' #(dIRECTORY OF TEST DATA)
  inun_files = []

  inun_files += [each for each in os.listdir(directory1) if each.endswith('.wd')]
  inun_files.sort()


  l = ['Run1-0000.wd', 'Run1-0001.wd', 'Run1-0002.wd', 'Run1-0003.wd', 'Run1-0004.wd', 'Run1-0005.wd', 'Run1-0006.wd', 'Run1-0007.wd']

  for i in l:
    inun_files.remove(i)


  for i in range(len(inun_files)):
    src = rio.open(directory1+inun_files[i])
    # Sample the raster at every point location and store values in DataFrame
    pts['Raster Value'+'_step_{}'.format(i+8)] = [x[0] for x in src.sample(coords)]

  df = pd.DataFrame(pts)

    ##output dir

  d = '/home/cvssk/Carlisle/LSTM_Validation/'
  df.to_csv(d+ind+'.csv')

def export_pred_data(locations, ind):
  import geopandas as gpd

  # Read points from shapefile
  pts = gpd.read_file(locations)
  pts = pts[['X', 'Y', 'Descriptio','geometry']]
  pts.index = range(len(pts))
  coords = [(x,y) for x, y in zip(pts.X, pts.Y)]

  directory = '/home/cvssk/Carlisle/CNN_Outputs/' #(dIRECTORY OF predicted DATA)
  inun_files = []

  inun_files += [each for each in os.listdir(directory) if each.endswith('.tif')]
  inun_files.sort()

  for i in range(len(inun_files)):
    src = rio.open(directory+inun_files[i])

    fname = inun_files[i].replace('.tif','')
    # Sample the raster at every point location and store values in DataFrame
    pts[fname] = [x[0] for x in src.sample(coords)]

  df = pd.DataFrame(pts)

    ##output dir

  d = '/home/cvssk/Carlisle/CNN_Validation/'
  df.to_csv(d+ind+'.csv')


