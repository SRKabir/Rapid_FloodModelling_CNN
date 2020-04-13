#%%
import os
import sys
import pandas as pd
import numpy as np 
##############################################################################
scriptsPath = '/home/cvssk/Carlisle/RapidCNN_Inun/' # location of the Carlisle_Inun_Mod.py
sys.path.insert(0,scriptsPath)

os.chdir(scriptsPath)

from Carlisle_InunMod import data_pre_process, CNN_Model_2lr, CNN_LSTM_Model, LSTM_Model,save_model, load_model, predict,predict_cnnlstm, export_ref_data, export_pred_data

#%%
##preprocess data
#x_train, Y, x_test, Y_test, steps, features, outputs, X_Test = data_pre_process()

#Import the training and testing data directly from the disk instead

x_train = pd.read_csv('/home/cvssk/Carlisle/RapidCNN_Inun/Data/X_Train.csv', header=None)
x_test = pd.read_csv('/home/cvssk/Carlisle/RapidCNN_Inun/Data/X_Test.csv', header=None)

Y1 = pd.read_hdf('/home/cvssk/Carlisle/RapidCNN_Inun/Data/Y_Train_Subset1.h5')
Y2 = pd.read_hdf('/home/cvssk/Carlisle/RapidCNN_Inun/Data/Y_Train_Subset2.h5')

dfs = [Y1,Y2]

Y = pd.concat(dfs)

Y_test = pd.read_hdf('/home/cvssk/Carlisle/RapidCNN_Inun/Data/Y_Test.h5')


#%%

x_train = np.array(x_train)
x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1])

x_test = np.array(x_test)
x_test = x_test.reshape(x_test.shape[0],1,x_test.shape[1])

features = x_train.shape[2]
steps = x_train.shape[1]

Y = np.array(Y)
Y_test = np.array(Y_test)

outputs = Y.shape[1]
X_Test = np.array(pd.read_csv('/home/cvssk/Carlisle/RapidCNN_Inun/Data/X_Test.csv', header=None))

#%%
##train model
#model = CNN_Model(x_train, Y, x_test, Y_test, steps, features, outputs)
model = CNN_Model_2lr(x_train, Y, x_test, Y_test, steps, features, outputs)
#model = CNN_LSTM_Model(x_train, Y, x_test, Y_test, outputs)
#model = LSTM_Model(x_train, Y, x_test, Y_test,features, outputs)


#%%
##save model

#name = '/home/cvssk/Carlisle/models/CNN_model' #change names for different models
#save_model(model, name)

#%%
##load model

#model = load_model(name)

#%%
##predict outputs
index = 'CNN_wd'
predict(model, X_Test, index) #use this for cnn and lstm
#predict_cnnlstm(model,X_Test,index) #use this for cnn-lstm


#%%
#Export refence values from LISFLOOD at validation points
locations = '/home/cvssk/Carlisle/RapidCNN_Inun/Validation_locations/validation_locations.shp'
#ind = 'LISFLOOD_wd'

#export_ref_data(locations, ind)

#%%
#Export predicted values at validation points
ind = 'LSTM_wd'
export_pred_data(locations,ind)


# %%
