#%%
from sklearn import metrics
import scipy as sp
import numpy as np
import math
from sklearn import metrics
from IPython.display import display
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
import pandas as pd
import io
import os
import sys
import requests
import numpy as np

##############################################################################
scriptsPath = '/home/cvssk/Carlisle/' # location of the Carlisle_Inun_Mod.py
sys.path.insert(0,scriptsPath)

os.chdir(scriptsPath)

from Carlisle_InunMod import data_pre_process, CNN_Model_2lr

#%%
x_train, Y, x_test, Y_test, steps, features, outputs, X_Test = data_pre_process()

#%%
def perturbation_rank(model, x_train, Y, names):
    errors = []

    for i in range(x_train.shape[2]):
        x = x_train.reshape(x_train.shape[0],x_train.shape[2])
        hold = np.array(x[:, i])
        np.random.shuffle(x[:, i])

        pred = model.predict(x_train)
        error = metrics.mean_squared_error(Y, pred)
        #if regression:
        #    pred = model.predict(x_train)
        #    error = metrics.mean_squared_error(Y, pred)
        #else:
        #   pred = model.predict_proba(x_train)
        #    error = metrics.log_loss(Y, pred)
            
        errors.append(error)
        x[:, i] = hold
        
    max_error = np.max(errors)
    importance = [e/max_error for e in errors]

    data = {'name':names,'error':errors,'importance':importance}
    result = pd.DataFrame(data, columns = ['name','error','importance'])
    result.sort_values(by=['importance'], ascending=[0], inplace=True)
    result.reset_index(inplace=True, drop=True)
    return result

#%%
model = CNN_Model_2lr(x_train, Y, x_test, Y_test, steps, features, outputs)


#%%
#####import dataframe of training data with features
df = pd.read_csv('/home/cvssk/Carlisle/Flows/Train/appended.csv')
print(df.head())
df = df.iloc[:,1:]
print(df.head())

#%%
names = list(df.columns) # x+y column names

rank = perturbation_rank(model, x_train, Y, names)
display(rank)

# %%
rnk = pd.DataFrame(rank)

rnk.to_csv('/home/cvssk/Carlisle/Feature_Rank.csv')
