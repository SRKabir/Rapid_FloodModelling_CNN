#%%

# check scikit-learn version
import sklearn
print(sklearn.__version__)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import timeit

# %%
#prepare target data


def Target_Data_Cal(locations):
    import geopandas as gpd
    import pandas as pd
    import rasterio as rio 
    '''
    This function is to generate target data
    at validation points used for SVM calibration
    purposes.
    '''


    # Read points from shapefile
    pts = gpd.read_file(locations)
    pts = pts[['X', 'Y', 'Descriptio','geometry']]
    pts.index = range(len(pts))
    coords = [(x,y) for x, y in zip(pts.X, pts.Y)]



    ## Train target generation
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

    for i in range(len(inun_files2)):
        src = rio.open(Target+inun_files2[i])
        # Sample the raster at every point location and store values in DataFrame
        pts['Raster Value'+'_step_{}'.format(i+8)] = [x[0] for x in src.sample(coords)]

    df = pd.DataFrame(pts)

    ##output dir

    d = '/home/cvssk/Carlisle/SVM_Model/'
    df.to_csv(d+'Y_Train'+'.csv')

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

    d = '/home/cvssk/Carlisle/SVM_Model/'
    df.to_csv(d+'Y_Test.csv')

def Gen_X_param():
    import pandas as pd
    import numpy as np
    import os
    from sklearn.preprocessing import MinMaxScaler
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

    #appended_data.to_csv('/home/cvssk/Carlisle/Flows/Train/appended.csv')

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

    all_data.to_csv('/home/cvssk/Carlisle/Flows/Train/All_data.csv')

    scaler = MinMaxScaler(feature_range=(0, 1))
    all_data = scaler.fit_transform(all_data)

    X_Train = all_data[0:1243, :]
    X_Test = all_data[1243:, :]

    return X_Train, X_Test

#%%
#generate X and Target variables

#X_Train, X_Test = Gen_X_param()

X_Train = pd.read_csv('/home/cvssk/Carlisle/RapidCNN_Inun/Data/X_Train.csv', header=None)
X_Test = pd.read_csv('/home/cvssk/Carlisle/RapidCNN_Inun/Data/X_Test.csv', header=None)


# %%

# %%

#locations of the ground points

#locations = '/home/cvssk/Carlisle/Validation_locations/validation_locations.shp'

#Target_Data_Cal(locations)


# %%
import pandas as pd 
Target = pd.read_csv('/home/cvssk/Carlisle/RapidCNN_Inun/Data/Y_Train_SVR.csv')

# %%
Target.head()

# %%
import numpy as np 
col_names = np.array(Target['Descriptio'])

# %%

Y = Target.iloc[:,5:]
Y = Y.T
Y.columns = col_names


# %%
Y[Y<0.2] = 0

# %%
##Y test data

Y_Test = pd.read_csv('/home/cvssk/Carlisle/RapidCNN_Inun/Data/Y_Test_SVR.csv')
Y_Test.head()

Y_Test = Y_Test.iloc[:,5:]
Y_Test = Y_Test.T
Y_Test.columns = col_names

Y_Test[Y_Test<0.2] = 0

val_pts = [0,1,2,6,7,8,9,12,13,15,16,19,23,25,26,27,28,29]

# %%
from hyperopt import tpe
from hpsklearn import HyperoptEstimator,svr_rbf

estim = HyperoptEstimator(regressor=svr_rbf('my_rgr'),
                          preprocessing=[],
                          algo=tpe.suggest,
                          max_evals=10,
                          trial_timeout=300)

# Search the hyperparameter space based on the data

models=[]
for i in val_pts:
    estim.fit(X_Train, Y.iloc[:,i])

    # Show the results

    print(estim.score(X_Test, Y_Test.iloc[:,i]))

    print(estim.best_model())
    best_mod = estim.best_model()
    models.append(best_mod)


#%%
scores = []
for i in range(len(val_pts)):
    svr = models[i]['learners']

    acc_score = []

    for j in val_pts:
        model = svr.fit(X_Train, Y.iloc[:,j])
        y_pred = model.predict(X_Test)
        acc = round(y_pred, 3)
        acc_score.append(acc)
    
    scores.append(acc_score)

score = pd.DataFrame(scores)

score.columns = col_names

scores.to_csv('/home/cvssk/Carlisle/SVR_Outputs/svr_valid/18points_validation_acc.csv')


#%%
def save_model(model, name):
    import pickle
    filename= name+'.sav'

    pickle.dump(model, open(filename, 'wb'))

def load_model(name):
    filename= name+'.sav'
    model =pickle.load(open(filename, 'rb'))

    return model


#%%
from sklearn.svm import SVR

predicted = []

##start time
start = timeit.default_timer()

for i in val_pts:
    y = Y.iloc[:,i]

    model = SVR(C=25.296038321648346, cache_size=512, coef0=0.0, degree=1,
    epsilon=0.031196144376997643, gamma=0.016161023499762946, kernel='rbf',
    max_iter=70067892.0, shrinking=False, tol=0.002590932184680306,
    verbose=False)

    model.fit(X_Train, y)

    name = '/home/cvssk/Carlisle/RapidCNN_Inun/Model' #change this location to save the trained models
    name = name+str(i)

    save_model(model,name)
    y_pred = model.predict(X_Test)

    predicted.append(y_pred)

stop = timeit.default_timer()

print('Time: ', stop - start)

#%%
df = pd.DataFrame(predicted).T
loc_pts = [str(i) for i in val_pts]
df.columns = loc_pts
df[df<0] = 0

df.to_csv('/home/cvssk/Carlisle/RapidCNN_Inun//18points_prediction.csv')




# %%
