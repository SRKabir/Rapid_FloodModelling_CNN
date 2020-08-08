#%%

# check scikit-learn version
import sklearn
print(sklearn.__version__)

import pandas as pd
import numpy as np
import geopandas as gpd 
import os
from sklearn.preprocessing import StandardScaler
import timeit

#%%
def Target_Data_Cal(locations):
    import geopandas as gpd
    import pandas as pd
    import rasterio as rio
    import os
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
    Target = '/home/cvssk/Carlisle_Resubmission/2015Event/Target/'
    inun_files2 = []

    ##PROCESS TARGET DATA (Y_PARAM)
    inun_files2 += [each for each in os.listdir(Target) if each.endswith('.wd')]
    inun_files2.sort()

    ls = ['Run2-0000.wd', 'Run2-0001.wd', 'Run2-0002.wd', 'Run2-0003.wd', 'Run2-0004.wd', 'Run2-0005.wd', 'Run2-0006.wd', 'Run2-0007.wd',
      'Run3-0000.wd', 'Run3-0001.wd', 'Run3-0002.wd', 'Run3-0003.wd', 'Run3-0004.wd', 'Run3-0005.wd', 'Run3-0006.wd', 'Run3-0007.wd',
      'Run4-0000.wd', 'Run4-0001.wd', 'Run4-0002.wd', 'Run4-0003.wd', 'Run4-0004.wd', 'Run4-0005.wd', 'Run4-0006.wd', 'Run4-0007.wd',
      'Run5-0000.wd', 'Run5-0001.wd', 'Run5-0002.wd', 'Run5-0003.wd', 'Run5-0004.wd', 'Run5-0005.wd', 'Run5-0006.wd', 'Run5-0007.wd',
      'Run6-0000.wd', 'Run6-0001.wd', 'Run6-0002.wd', 'Run6-0003.wd', 'Run6-0004.wd', 'Run6-0005.wd', 'Run6-0006.wd', 'Run6-0007.wd',
      'Run7-0000.wd', 'Run7-0001.wd', 'Run7-0002.wd', 'Run7-0003.wd', 'Run7-0004.wd', 'Run7-0005.wd', 'Run7-0006.wd', 'Run7-0007.wd',
      'Run8-0000.wd', 'Run8-0001.wd', 'Run8-0002.wd', 'Run8-0003.wd', 'Run8-0004.wd', 'Run8-0005.wd', 'Run8-0006.wd', 'Run8-0007.wd',
      'Run9-0000.wd', 'Run9-0001.wd', 'Run9-0002.wd', 'Run9-0003.wd', 'Run9-0004.wd', 'Run9-0005.wd', 'Run9-0006.wd', 'Run9-0007.wd']

    for i in ls:
        inun_files2.remove(i)

    for i in range(len(inun_files2)):
        src = rio.open(Target+inun_files2[i])
        # Sample the raster at every point location and store values in DataFrame
        pts['Raster Value'+'_SVR_{:03}'.format(i+8)] = [x[0] for x in src.sample(coords)]

    df = pd.DataFrame(pts)

    ##output dir

    d = '/home/cvssk/Carlisle_Resubmission/2015Event/SVR/'
    df.to_csv(d+'Y_Train_18pts'+'.csv')
    y_train = df
    return y_train



# %%
#locations = '/home/cvssk/Carlisle_Resubmission/validation_locations/New_500Samples.shp'
locations = '/home/cvssk/Carlisle_Resubmission/validation_locations/validation_locations_18Points.shp'
Y_train = Target_Data_Cal(locations)

# %%

Y_train = pd.read_csv('/home/cvssk/Carlisle_Resubmission/2015Event/SVR/Y_Train_18pts.csv')

####Import Precipitation/Discharge Data
data_dir = '/home/cvssk/Carlisle_Resubmission/2015Event/Flows/'

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
df = pd.read_csv('/home/cvssk/Carlisle_Resubmission/2015Event/Flows/Test/Upstream_Flows_Test2.csv')


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


scaler = StandardScaler() #MinMaxScaler(feature_range=(0, 1))
all_data = scaler.fit_transform(all_data)

X_Train = all_data[0:2104, :] 
X_Test = all_data[2104:, :]

# %%
Y = Y_train.iloc[:,5:]
cols = list(Y_train.iloc[:,3])
Y = Y.T
Y.columns = cols
Y[Y<0.2] = 0

# %%
from sklearn.svm import SVR

predicted = []
#col_name = []
##start time
start = timeit.default_timer()
for i in Y.columns:
    y = Y[i]

    model = SVR(C=25.296038321648346, cache_size=512, coef0=0.0, degree=1,
    epsilon=0.031196144376997643, gamma=0.016161023499762946, kernel='rbf',
    max_iter=70067892.0, shrinking=False, tol=0.002590932184680306,
    verbose=False)

    model.fit(X_Train, y)

    #name = '/home/cvssk/Carlisle_Resubmission/2015Event/SVR/models/Model'
    #name = name+str(i)

    #save_model(model,name)
    y_pred = model.predict(X_Test)

    #col_name.append(i)
    predicted.append(y_pred)

#stop time
stop = timeit.default_timer()
print('Time: ', stop - start)

#%%
df = pd.DataFrame(predicted)
df[df<0] = 0
df = df.T
df.columns = cols
df.to_csv('/home/cvssk/Carlisle_Resubmission/2015Event/SVR/SVR_Outputs/18pts_prediction.csv')


# %%
pts = gpd.read_file(locations)
df['x'] = pts.x
df['y'] = pts.y
df['Elevation'] = pts.Carlisle_5

# %%
#extract elevation data
#import rasterio as rio
#src = rio.open('/home/cvssk/Carlisle/Carlisle_5m.tif')
#locations = '/home/cvssk/Carlisle/validation_locations/validation_locations.shp'
#pts = gpd.read_file(locations)
#pts = pts[['X_COORD', 'Y_COORD', 'Descriptio','geometry']]
#pts.index = range(len(pts))
#coords = [(x,y) for x, y in zip(pts.X_COORD, pts.Y_COORD)]


#pts['Elevation'] = [x[0] for x in src.sample(coords)]


# %%
#df['Elevation'] = pts['Elevation']
df.to_csv('/home/cvssk/Carlisle_Resubmission/2015Event/SVR/SVR_Outputs/500Sample_prediction.csv')


# %%
############RegressionKriging##############
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from pykrige.rk import RegressionKriging
import rasterio as rio 
from pykrige.compat import train_test_split
from rasterio.plot import show

# %%
#load SVR predicted water depth
df = pd.read_csv('/home/cvssk/Carlisle_Resubmission/2015Event/SVR/SVR_Outputs/500Sample_prediction.csv')
df.head()

# %%
#make subsets i.e. water depth series and contro points with elvation

sub1 = df.iloc[:,1:267] ##depth sequences
sub1[sub1 < 0.3] = 0.0
sub2 = df.iloc[:,267:270] ## elevation

#%%
###RK step

x = sub2.iloc[:,:-1]
x = x.to_numpy() #2D array of coordinates
#convert and reshape the predictor (elevation) variable
p = sub2['Elevation'].values
p = p.reshape(p.shape[0], 1)

src = rio.open('/home/cvssk/Carlisle_Resubmission/2015Event/Carlisle_5m.asc')


#%%
#import gridded xyz values for prediction
#it is possible to extract centroid elevation in Python, however, its much easier
#to this in R. Therefore, we used R to extract centroid pixel values of the src image 
grd = pd.read_csv("/home/cvssk/Carlisle_Resubmission/2015Event/SVR/gridded_elevation.csv")
# %%

x_grd = grd.to_numpy()
x_y = x_grd[:, :-1]
##numpy array of surface elevation
z = x_grd[:, -1:]


#%%
#constract a randomforest model
rf_model = RandomForestRegressor(n_estimators=100)
#fit the regression kriging model
m_rk = RegressionKriging(regression_model=rf_model,
n_closest_points=2, method= 'ordinary',
variogram_model='spherical')

#%%
m_rk.fit(p, x, sub1.iloc[:,39])
pred_sph = m_rk.predict(z, x_y)

#%%
sph = pred_sph

#%%
#filter values
sph[sph < 0.3] = 0
sph.resize(src.height, src.width)
show(sph)

#%%
with rio.Env():
    # Write an array as a raster band to a new 8-bit file. For
    # the new file's profile, we start with the profile of the source
    profile = src.profile

    # And then change the band count to 1, set the
    # dtype to uint8, and specify LZW compression.
    profile.update(dtype=str(sph.dtype), count=1,compress='lzw')

    with rio.open('/home/cvssk/Carlisle_Resubmission/2015Event/SVR/SVR_Outputs/Rasters/svr_048_500Sample.asc', 'w', **profile) as dst:
    #with rio.open(tar_dir+fname+index+'.tif', 'w', **profile) as dst:
        dst.write(np.absolute(sph), 1)


# %%
