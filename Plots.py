#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
from IPython.display import display
from glob import glob0
import numpy as np

#%%
def rmse(sim, obs):
    """
    Root Mean Squared Error
    """
    
    import numpy as np
    return np.sqrt(np.mean((sim-obs)**2))


def nse(sim, obs):
    """
    Nash Sutcliffe efficiency coefficient
  
    """
    import numpy as np
    return 1-sum((sim-obs)**2)/sum((obs-np.mean(obs))**2)


#%%
cnn = pd.read_csv('/home/cvssk/Carlisle_Resubmission/2005Event/CNN_Validation/WithOutBN/CNN_2005_Validation.csv')
svr = pd.read_csv('/home/cvssk/Carlisle_Resubmission/2005Event/SVR/SVR_Outputs/18pts_validation_prediction.csv')
#cnn_bn = pd.read_csv('/home/cvssk/Carlisle_Resubmission/2005Event/CNN_Validation//CNN_2005_Validation.csv')
ref = pd.read_csv('/home/cvssk/Carlisle_Resubmission/2005Event/CNN_Validation/WithOutBN/LF_2005_Validation.csv')

#%%
cnn = cnn.T 
cnn_cols = list(cnn.iloc[3,:])
cnn = cnn.iloc[5:,:]
cnn.columns = cnn_cols

#cnn_bn = cnn_bn.T 
#cnn_bn_cols = list(cnn_bn.iloc[3,:])
#cnn_bn = cnn_bn.iloc[5:,:]
#cnn_bn.columns = cnn_bn_cols

svr = svr.iloc[:,1:]

ref = ref.T 
ref_cols = list(ref.iloc[3,:])
ref = ref.iloc[5:,:]
ref.columns = ref_cols

#%%

th = 0.3

cnn[cnn < th] = 0
#cnn_bn[cnn_bn < th] = 0
svr[svr < th] = 0
ref[ref < th] = 0

ncnn = cnn.iloc[::4, :]
ncnn = ncnn.reset_index(drop=True)

nsvr = svr.iloc[::4, :]
nsvr = nsvr.reset_index(drop=True)

nref = ref.iloc[::4, :]
nref = nref.reset_index(drop=True)

cols = cnn.columns
cols


#%%

fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(8,10), dpi=1200, facecolor='w', edgecolor='k')

ax[0,0].plot(nref.index, ncnn[cols[0]], color='red', ls='--',label="CNN")
ax[0,0].plot(nref.index, nsvr[cols[0]], color='orange',label="SVR")
ax[0,0].plot(nref.index, nref[cols[0]], color='black',label="LISFLOOD")
ax[0,0].tick_params(direction='out', width=2,labelbottom=False)
ax[0,0].set_title(cols[0],fontsize=10,fontweight='bold')
ax[0,0].legend(loc='lower center', fontsize=8)


ax[0,1].plot(nref.index, ncnn[cols[1]], color='red', ls='--')
ax[0,1].plot(nref.index, nref[cols[1]], color='black')
ax[0,1].plot(nref.index, nsvr[cols[1]], color='orange')
ax[0,1].set_title(cols[1],fontsize=10,fontweight='bold')
ax[0,1].tick_params(direction='out', width=2,labelbottom=False)


ax[0,2].plot(nref.index, ncnn[cols[2]], color='red', ls='--')
ax[0,2].plot(nref.index, nref[cols[2]], color='black')
ax[0,2].plot(nref.index, nsvr[cols[2]], color='orange')
ax[0,2].set_title(cols[2],fontsize=10,fontweight='bold')
ax[0,2].tick_params(direction='out', width=2,labelbottom=False)



ax[1,0].plot(nref.index, ncnn[cols[3]], color='red', ls='--')
ax[1,0].plot(nref.index, nref[cols[3]], color='black')
ax[1,0].plot(nref.index, nsvr[cols[3]], color='orange')
ax[1,0].set_title(cols[3],fontsize=10,fontweight='bold')
ax[1,0].tick_params(direction='out', width=2,labelbottom=False)


ax[1,1].plot(nref.index, ncnn[cols[4]], color='red', ls='--')
ax[1,1].plot(nref.index, nref[cols[4]], color='black')
ax[1,1].plot(nref.index, nsvr[cols[4]], color='orange')
ax[1,1].set_title(cols[4],fontsize=10,fontweight='bold')
ax[1,1].tick_params(direction='out', width=2,labelbottom=False)

ax[1,2].plot(nref.index, ncnn[cols[5]], color='red', ls='--')
ax[1,2].plot(nref.index, nref[cols[5]], color='black')
ax[1,2].plot(nref.index, nsvr[cols[5]], color='orange')
ax[1,2].set_title(cols[5],fontsize=10,fontweight='bold')
ax[1,2].tick_params(direction='out', width=2,labelbottom=False)

ax[2,0].plot(nref.index, ncnn[cols[6]], color='red', ls=
'--')
ax[2,0].plot(nref.index, nref[cols[6]], color='black')
ax[2,0].plot(nref.index, nsvr[cols[6]], color='orange')
ax[2,0].set_title(cols[6],fontsize=10,fontweight='bold')
ax[2,0].tick_params(direction='out', width=2,labelbottom=False)


ax[2,1].plot(nref.index, ncnn[cols[7]], color='red', ls='--')
ax[2,1].plot(nref.index, nref[cols[7]], color='black')
ax[2,1].plot(nref.index, nsvr[cols[7]], color='orange')
ax[2,1].set_title(cols[7],fontsize=10,fontweight='bold')
ax[2,1].tick_params(direction='out', width=2,labelbottom=False)


ax[2,2].plot(nref.index, ncnn[cols[8]], color='red', ls='--')
ax[2,2].plot(nref.index, nref[cols[8]], color='black')
ax[2,2].plot(nref.index, nsvr[cols[8]], color='orange')
ax[2,2].set_title(cols[8],fontsize=10,fontweight='bold')
ax[2,2].tick_params(direction='out', width=2,labelbottom=False)

ax[3,0].plot(nref.index, ncnn[cols[9]], color='red', ls='--')
ax[3,0].plot(nref.index, nref[cols[9]], color='black')
ax[3,0].plot(nref.index, nsvr[cols[9]], color='orange')
ax[3,0].set_title(cols[9],fontsize=10,fontweight='bold')
ax[3,0].tick_params(direction='out', width=2,labelbottom=False)

ax[3,1].plot(nref.index, ncnn[cols[10]], color='red', ls='--')
ax[3,1].plot(nref.index, nref[cols[10]], color='black')
ax[3,1].plot(nref.index, nsvr[cols[10]], color='orange')
ax[3,1].set_title(cols[10],fontsize=10,fontweight='bold')
ax[3,1].tick_params(direction='out', width=2,labelbottom=False)


ax[3,2].plot(nref.index, ncnn[cols[11]], color='red', ls='--')
ax[3,2].plot(nref.index, nref[cols[11]], color='black')
ax[3,2].plot(nref.index, nsvr[cols[11]], color='orange')
ax[3,2].set_title(cols[11],fontsize=10,fontweight='bold')
ax[3,2].tick_params(direction='out', width=2,labelbottom=False)


ax[4,0].plot(nref.index, ncnn[cols[12]], color='red', ls='--')
ax[4,0].plot(nref.index, nref[cols[12]], color='black')
ax[4,0].plot(nref.index, nsvr[cols[12]], color='orange')
ax[4,0].set_title(cols[12],fontsize=10,fontweight='bold')
ax[4,0].tick_params(direction='out', width=2,labelbottom=False)


ax[4,1].plot(nref.index, ncnn[cols[13]], color='red', ls='--')
ax[4,1].plot(nref.index, nref[cols[13]], color='black')
ax[4,1].plot(nref.index, nsvr[cols[13]], color='orange')
ax[4,1].set_title(cols[13],fontsize=10,fontweight='bold')
ax[4,1].tick_params(direction='out', width=2,labelbottom=False)

ax[4,2].plot(nref.index, ncnn[cols[14]], color='red', ls='--')
ax[4,2].plot(nref.index, nref[cols[14]], color='black')
ax[4,2].plot(nref.index, nsvr[cols[14]], color='orange')
ax[4,2].set_title(cols[14],fontsize=10,fontweight='bold')
ax[4,2].tick_params(direction='out', width=2,labelbottom=False)


ax[5,0].plot(nref.index, ncnn[cols[15]], color='red', ls='--')
ax[5,0].plot(nref.index, nref[cols[15]], color='black')
ax[5,0].plot(nref.index, nsvr[cols[15]], color='orange')
ax[5,0].set_title(cols[15],fontsize=10,fontweight='bold')
ax[5,0].tick_params(direction='out', width=2,labelbottom=True)


ax[5,1].plot(nref.index, ncnn[cols[16]], color='red', ls='--')
ax[5,1].plot(nref.index, nref[cols[16]], color='black')
ax[5,1].plot(nref.index, nsvr[cols[16]], color='orange')
ax[5,1].set_title(cols[16],fontsize=10,fontweight='bold')
ax[5,1].tick_params(direction='out', width=2,labelbottom=True)

ax[5,2].plot(nref.index, ncnn[cols[17]], color='red', ls='--')
ax[5,2].plot(nref.index, nref[cols[17]], color='black')
ax[5,2].plot(nref.index, nsvr[cols[17]], color='orange')
ax[5,2].set_title(cols[17],fontsize=10,fontweight='bold')
ax[5,2].tick_params(direction='out', width=2,labelbottom=True)

# add a big axis, hide frame

fig.add_subplot(111, frameon=False)

# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

plt.xlabel('Time step (h)',ha='center',fontsize=13,fontweight='bold')
plt.ylabel('Water depth (m)', rotation='vertical',fontsize=13,fontweight='bold')

fig.tight_layout()


# %%
fig.savefig('/home/cvssk/Carlisle_Resubmission/2005Event/CNN_Validation/Depth_cutoff3_plot_CNNvsSVR_2005.png', dpi = 1200)

##save plot
from PIL import Image
from io import BytesIO

# (1) save the image in memory in PNG format
png1 = BytesIO()
fig.savefig(png1, format='png')

# (2) load this image into PIL
png2 = Image.open(png1)

# (3) save as TIFF
png2.save('/home/cvssk/Carlisle_Resubmission/2005Event/CNN_Validation/Depth_cutoff3_plot_CNNvsSVR_2005.tif')
png1.close()

# %%
ns = []
rms = []
col_name = []

for i in range(len(cols)):
    sim = np.array(cnn[cols[i]])
    obs = np.array(ref[cols[i]])
    n = 1-sum((sim-obs)**2)/sum((obs-np.mean(obs))**2)
    rm = np.sqrt(np.mean((sim-obs)**2))
    ns.append(n)
    rms.append(rm)
    col_name.append(cols[i])

error_mat = {'location_name': col_name,
             'nas_sut': ns,
            'rmse':rms}

error = pd.DataFrame(error_mat)



#%%
error.to_csv('/home/cvssk/Carlisle_Resubmission/2005Event/CNN_Validation/st2_error_stats_cutoff_3.csv')


# %%

#########Hydrograph Plots
df1 = pd.read_csv('/home/cvssk/Carlisle_Resubmission/2005Event/Flows/All_Flow_Data/Upstream1_FlowHydrographs.csv')
df1 = df1.iloc[::4, :]
df1 = df1.reset_index(drop=True)

df2 = pd.read_csv('/home/cvssk/Carlisle_Resubmission/2005Event/Flows/All_Flow_Data/Upstream2_FlowHydrographs.csv')
df2 = df2.iloc[::4, :]
df2 = df2.reset_index(drop=True)


df3 = pd.read_csv('/home/cvssk/Carlisle_Resubmission/2005Event/Flows/All_Flow_Data/Upstream3_FlowHydrographs.csv')
df3 = df3.iloc[::4, :]
df3 = df3.reset_index(drop=True)

stream1 = df1.iloc[:,2:]
stream2 = df2.iloc[:,2:]
stream3 = df3.iloc[:,2:]

stream1.columns = ['Upstream 1 hydrograph A', 'Upstream 1 hydrograph B','Upstream 1 hydrograph C','Upstream 1 hydrograph D','Upstream 1 hydrograph E','Upstream 1 hydrograph F','Upstream 1 hydrograph G','Upstream 1 hydrograph H']
stream2.columns = ['Upstream 2 hydrograph A', 'Upstream 2 hydrograph B','Upstream 2 hydrograph C','Upstream 2 hydrograph D','Upstream 2 hydrograph E','Upstream 2 hydrograph F','Upstream 2 hydrograph G','Upstream 2 hydrograph H']
stream3.columns = ['Upstream 3 hydrograph A', 'Upstream 3 hydrograph B','Upstream 3 hydrograph C','Upstream 3 hydrograph D','Upstream 3 hydrograph E','Upstream 3 hydrograph F','Upstream 3 hydrograph G','Upstream 3 hydrograph H']

Event_2005_stream1 = df1.iloc[:,0]
Event_2005_stream2 = df2.iloc[:,0]
Event_2005_stream3 = df3.iloc[:,0]

Event_2015_stream1 = df1.iloc[:,1]
Event_2015_stream2 = df2.iloc[:,1]
Event_2015_stream3 = df3.iloc[:,1]


#%%
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18,10), dpi=1200, facecolor='w', edgecolor='k')

ax[0,0].plot( stream1['Upstream 1 hydrograph A'], color = 'black',ls = '-')
ax[0,0].plot( stream1['Upstream 1 hydrograph B'], color = 'red', ls = '-')
ax[0,0].plot( stream1['Upstream 1 hydrograph C'], color = 'blue',ls = '--')
ax[0,0].plot( stream1['Upstream 1 hydrograph D'], color = 'green',ls = '-.')
ax[0,0].plot( stream1['Upstream 1 hydrograph E'], color = 'magenta',ls = ':')
ax[0,0].plot( stream1['Upstream 1 hydrograph F'], color = 'cyan',ls = '-')
ax[0,0].plot( stream1['Upstream 1 hydrograph G'], color = 'blue',ls = '-')
ax[0,0].plot( stream1['Upstream 1 hydrograph H'], color = 'green',ls = '-')

ax[0,0].set_ylabel(r"Discharge $\mathbf{(m^3 s^{-1})}$",fontsize=14,fontweight='bold')
ax[0,0].set_title('Upstream 1: River Eden',fontsize=14,fontweight='bold')
ax[0,0].tick_params(direction='out', width=2,labelbottom=True,labelleft=True,labelsize ='large')

ax[0,1].plot( stream2['Upstream 2 hydrograph A'], color = 'black',ls = '-', label="hydrograph A")
ax[0,1].plot( stream2['Upstream 2 hydrograph B'], color = 'red', ls = '-',label="hydrograph B")
ax[0,1].plot( stream2['Upstream 2 hydrograph C'], color = 'blue',ls = '--',label="hydrograph C")
ax[0,1].plot( stream2['Upstream 2 hydrograph D'], color = 'green',ls = '-.',label="hydrograph D")
ax[0,1].plot( stream2['Upstream 2 hydrograph E'], color = 'magenta',ls = ':',label="hydrograph E")
ax[0,1].plot( stream2['Upstream 2 hydrograph F'], color = 'cyan',ls = '-',label="hydrograph F")
ax[0,1].plot( stream2['Upstream 2 hydrograph G'], color = 'blue',ls = '-', label="hydrograph G")
ax[0,1].plot( stream2['Upstream 2 hydrograph H'], color = 'green',ls = '-', label="hydrograph H")

ax[0,1].legend(loc='upper right', fontsize=13)
ax[0,1].tick_params(direction='out', width=2,labelbottom=True,labelleft=True,labelsize ='large')
ax[0,1].set_title('Upstream 2: River Petteril',fontsize=14,fontweight='bold')

ax[0,2].plot( stream3['Upstream 3 hydrograph A'], color = 'black',ls = '-')
ax[0,2].plot( stream3['Upstream 3 hydrograph B'], color = 'red', ls = '-')
ax[0,2].plot( stream3['Upstream 3 hydrograph C'], color = 'blue',ls = '--')
ax[0,2].plot( stream3['Upstream 3 hydrograph D'], color = 'green',ls = '-.')
ax[0,2].plot( stream3['Upstream 3 hydrograph E'], color = 'magenta',ls = ':')
ax[0,2].plot( stream3['Upstream 3 hydrograph F'],color = 'cyan',ls = '-')
ax[0,2].plot( stream3['Upstream 3 hydrograph G'], color = 'blue',ls = '-')
ax[0,2].plot( stream3['Upstream 3 hydrograph H'], color = 'green',ls = '-')

ax[0,2].tick_params(direction='out', width=2,labelbottom=True,labelleft=True,labelsize ='large')
#ax[2].set_ylim(ylims)
ax[0,2].set_title('Upstream 3: River Caldew',fontsize=14,fontweight='bold')


#############

ax[1,0].plot( Event_2005_stream1, color = 'black',ls = '-')
ax[1,0].plot( Event_2015_stream1, color = 'blue',ls = '-')

ax[1,0].set_ylabel(r"Discharge $\mathbf{(m^3 s^{-1})}$",fontsize=14,fontweight='bold')
#ax[1,0].set_title('Upstream 1: River Eden',fontsize=14,fontweight='bold')
ax[1,0].tick_params(direction='out', width=2,labelbottom=True,labelleft=True,labelsize ='large')

ax[1,1].plot( Event_2005_stream2, color = 'black',ls = '-', label="2005 event")
ax[1,1].plot( Event_2015_stream2, color = 'blue',ls = '-', label="2015 event")
ax[1,1].legend(loc='upper right', fontsize=11)
ax[1,1].tick_params(direction='out', width=2,labelbottom=True,labelleft=True,labelsize ='large')
#ax[1,1].set_title('Upstream 2: River Petteril',fontsize=14,fontweight='bold')

ax[1,2].plot( Event_2005_stream3, color = 'black',ls = '-')
ax[1,2].plot( Event_2015_stream3, color = 'blue',ls = '-')
ax[1,2].tick_params(direction='out', width=2,labelbottom=True,labelleft=True,labelsize ='large')
#ax[2].set_ylim(ylims)
#ax[1,2].set_title('Upstream 3: River Caldew',fontsize=14,fontweight='bold')

# add a big axis, hide frame

fig.add_subplot(111, frameon=False)

# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

plt.xlabel('Time step (h)',ha='center',fontsize=14,fontweight='bold')

fig.tight_layout()


# %%
fig.savefig('/home/cvssk/Carlisle_Resubmission/2005Event/CNN_Validation/hydrographs.png', dpi = 1200)
# (1) save the image in memory in PNG format
png1 = BytesIO()
fig.savefig(png1, format='png')

# (2) load this image into PIL
png2 = Image.open(png1)

# (3) save as TIFF
png2.save('/home/cvssk/Carlisle_Resubmission/2005Event/CNN_Validation/hydrographs.tif')
png1.close()

# %%

##################Test2 Plots#######
######CNN and LF pre and post defence###############
cnn_pre = pd.read_csv('/home/cvssk/Carlisle_Resubmission/2005Event/CNN_Validation/Test2_WithOutBN/CNN_2005_Validation.csv')
lf_pre = pd.read_csv('/home/cvssk/Carlisle_Resubmission/2005Event/CNN_Validation/Test2_WithOutBN/LF_2005_Validation.csv')
cnn_post = pd.read_csv('/home/cvssk/Carlisle_Resubmission/2015Event/CNN_Validation/WithBN/CNN_2015_Validation.csv')
lf_post = pd.read_csv('/home/cvssk/Carlisle_Resubmission/2015Event/CNN_Validation/WithBN/LF_2015_Validation.csv')


# %%
cnn_pre = cnn_pre.T 
cnn_pre_cols = list(cnn_pre.iloc[3,:])
cnn_pre = cnn_pre.iloc[5:,:]
cnn_pre.columns = cnn_pre_cols

lf_pre = lf_pre.T 
lf_pre_cols = list(lf_pre.iloc[3,:])
lf_pre = lf_pre.iloc[5:,:]
lf_pre.columns = lf_pre_cols

cnn_post = cnn_post.T 
cnn_post_cols = list(cnn_post.iloc[3,:])
cnn_post = cnn_post.iloc[5:,:]
cnn_post.columns = cnn_post_cols

lf_post = lf_post.T 
lf_post_cols = list(lf_post.iloc[3,:])
lf_post = lf_post.iloc[5:,:]
lf_post.columns = lf_post_cols


# %%
th = 0.3

cnn_pre[cnn_pre < th] = 0
cnn_post[cnn_post < th] = 0
lf_pre[lf_pre < th] = 0
lf_post[lf_post < th] = 0

ncnn_pre = cnn_pre.iloc[::4, :]
ncnn_pre = ncnn_pre.reset_index(drop=True)

ncnn_post = cnn_post.iloc[::4, :]
ncnn_post = ncnn_post.reset_index(drop=True)

nlf_pre = lf_pre.iloc[::4, :]
nlf_pre = nlf_pre.reset_index(drop=True)

nlf_post = lf_post.iloc[::4, :]
nlf_post = nlf_post.reset_index(drop=True)

cols = cnn_pre.columns
cols

# %%

fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(8,10), dpi=1200, facecolor='w', edgecolor='k')

ax[0,0].plot(nlf_pre.index, ncnn_pre[cols[0]], color='red', ls='--',label="CNN(pre-def)")
ax[0,0].plot(nlf_pre.index, nlf_pre[cols[0]], color='orange',label="LF(pre-def)")
ax[0,0].plot(nlf_pre.index, ncnn_post[cols[0]], color='blue')
ax[0,0].plot(nlf_pre.index, nlf_post[cols[0]], color='black')
ax[0,0].tick_params(direction='out', width=2,labelbottom=False)
ax[0,0].set_title(cols[0],fontsize=10,fontweight='bold')
ax[0,0].legend(loc='lower center', fontsize=9)


ax[0,1].plot(nlf_pre.index, ncnn_pre[cols[1]], color='red', ls='--')
ax[0,1].plot(nlf_pre.index, nlf_pre[cols[1]], color='orange')
ax[0,1].plot(nlf_pre.index, ncnn_post[cols[1]], color='blue', ls='--',label="CNN(post-def)")
ax[0,1].plot(nlf_pre.index, nlf_post[cols[1]], color='black',label="LF(post-def)")
ax[0,1].tick_params(direction='out', width=2,labelbottom=False)
ax[0,1].set_title(cols[1],fontsize=10,fontweight='bold')
ax[0,1].legend(loc='lower center', fontsize=9)


ax[0,2].plot(nlf_pre.index, ncnn_pre[cols[2]], color='red', ls='--',label="CNN(pre-def)")
ax[0,2].plot(nlf_pre.index, nlf_pre[cols[2]], color='orange',label="LF(pre-def)")
ax[0,2].plot(nlf_pre.index, ncnn_post[cols[2]], color='blue', ls='--',label="CNN(post-def)")
ax[0,2].plot(nlf_pre.index, nlf_post[cols[2]], color='black',label="LF(post-def)")
ax[0,2].tick_params(direction='out', width=2,labelbottom=False)
ax[0,2].set_title(cols[2],fontsize=10,fontweight='bold')



ax[1,0].plot(nlf_pre.index, ncnn_pre[cols[3]], color='red', ls='--',label="CNN(pre-def)")
ax[1,0].plot(nlf_pre.index, nlf_pre[cols[3]], color='orange',label="LF(pre-def)")
ax[1,0].plot(nlf_pre.index, ncnn_post[cols[3]], color='blue', ls='--',label="CNN(post-def)")
ax[1,0].plot(nlf_pre.index, nlf_post[cols[3]], color='black',label="LF(post-def)")
ax[1,0].tick_params(direction='out', width=2,labelbottom=False)
ax[1,0].set_title(cols[3],fontsize=10,fontweight='bold')


ax[1,1].plot(nlf_pre.index, ncnn_pre[cols[4]], color='red', ls='--',label="CNN(pre-def)")
ax[1,1].plot(nlf_pre.index, nlf_pre[cols[4]], color='orange',label="LF(pre-def)")
ax[1,1].plot(nlf_pre.index, ncnn_post[cols[4]], color='blue', ls='--',label="CNN(post-def)")
ax[1,1].plot(nlf_pre.index, nlf_post[cols[4]], color='black',label="LF(post-def)")
ax[1,1].tick_params(direction='out', width=2,labelbottom=False)
ax[1,1].set_title(cols[4],fontsize=10,fontweight='bold')

ax[1,2].plot(nlf_pre.index, ncnn_pre[cols[5]], color='red', ls='--',label="CNN(pre-def)")
ax[1,2].plot(nlf_pre.index, nlf_pre[cols[5]], color='orange',label="LF(pre-def)")
ax[1,2].plot(nlf_pre.index, ncnn_post[cols[5]], color='blue', ls='--',label="CNN(post-def)")
ax[1,2].plot(nlf_pre.index, nlf_post[cols[5]], color='black',label="LF(post-def)")
ax[1,2].tick_params(direction='out', width=2,labelbottom=False)
ax[1,2].set_title(cols[5],fontsize=10,fontweight='bold')

ax[2,0].plot(nlf_pre.index, ncnn_pre[cols[6]], color='red', ls='--',label="CNN(pre-def)")
ax[2,0].plot(nlf_pre.index, nlf_pre[cols[6]], color='orange',label="LF(pre-def)")
ax[2,0].plot(nlf_pre.index, ncnn_post[cols[6]], color='blue', ls='--',label="CNN(post-def)")
ax[2,0].plot(nlf_pre.index, nlf_post[cols[6]], color='black',label="LF(post-def)")
ax[2,0].tick_params(direction='out', width=2,labelbottom=False)
ax[2,0].set_title(cols[6],fontsize=10,fontweight='bold')


ax[2,1].plot(nlf_pre.index, ncnn_pre[cols[7]], color='red', ls='--',label="CNN(pre-def)")
ax[2,1].plot(nlf_pre.index, nlf_pre[cols[7]], color='orange',label="LF(pre-def)")
ax[2,1].plot(nlf_pre.index, ncnn_post[cols[7]], color='blue', ls='--',label="CNN(post-def)")
ax[2,1].plot(nlf_pre.index, nlf_post[cols[7]], color='black',label="LF(post-def)")
ax[2,1].tick_params(direction='out', width=2,labelbottom=False)
ax[2,1].set_title(cols[7],fontsize=10,fontweight='bold')


ax[2,2].plot(nlf_pre.index, ncnn_pre[cols[8]], color='red', ls='--',label="CNN(pre-def)")
ax[2,2].plot(nlf_pre.index, nlf_pre[cols[8]], color='orange',label="LF(pre-def)")
ax[2,2].plot(nlf_pre.index, ncnn_post[cols[8]], color='blue', ls='--',label="CNN(post-def)")
ax[2,2].plot(nlf_pre.index, nlf_post[cols[8]], color='black',label="LF(post-def)")
ax[2,2].tick_params(direction='out', width=2,labelbottom=False)
ax[2,2].set_title(cols[8],fontsize=10,fontweight='bold')

ax[3,0].plot(nlf_pre.index, ncnn_pre[cols[9]], color='red', ls='--',label="CNN(pre-def)")
ax[3,0].plot(nlf_pre.index, nlf_pre[cols[9]], color='orange',label="LF(pre-def)")
ax[3,0].plot(nlf_pre.index, ncnn_post[cols[9]], color='blue', ls='--',label="CNN(post-def)")
ax[3,0].plot(nlf_pre.index, nlf_post[cols[9]], color='black',label="LF(post-def)")
ax[3,0].tick_params(direction='out', width=2,labelbottom=False)
ax[3,0].set_title(cols[9],fontsize=10,fontweight='bold')

ax[3,1].plot(nlf_pre.index, ncnn_pre[cols[10]], color='red', ls='--',label="CNN(pre-def)")
ax[3,1].plot(nlf_pre.index, nlf_pre[cols[10]], color='orange',label="LF(pre-def)")
ax[3,1].plot(nlf_pre.index, ncnn_post[cols[10]], color='blue', ls='--',label="CNN(post-def)")
ax[3,1].plot(nlf_pre.index, nlf_post[cols[10]], color='black',label="LF(post-def)")
ax[3,1].tick_params(direction='out', width=2,labelbottom=False)
ax[3,1].set_title(cols[10],fontsize=10,fontweight='bold')


ax[3,2].plot(nlf_pre.index, ncnn_pre[cols[11]], color='red', ls='--',label="CNN(pre-def)")
ax[3,2].plot(nlf_pre.index, nlf_pre[cols[11]], color='orange',label="LF(pre-def)")
ax[3,2].plot(nlf_pre.index, ncnn_post[cols[11]], color='blue', ls='--',label="CNN(post-def)")
ax[3,2].plot(nlf_pre.index, nlf_post[cols[11]], color='black',label="LF(post-def)")
ax[3,2].tick_params(direction='out', width=2,labelbottom=False)
ax[3,2].set_title(cols[11],fontsize=10,fontweight='bold')


ax[4,0].plot(nlf_pre.index, ncnn_pre[cols[12]], color='red', ls='--',label="CNN(pre-def)")
ax[4,0].plot(nlf_pre.index, nlf_pre[cols[12]], color='orange',label="LF(pre-def)")
ax[4,0].plot(nlf_pre.index, ncnn_post[cols[12]], color='blue', ls='--',label="CNN(post-def)")
ax[4,0].plot(nlf_pre.index, nlf_post[cols[12]], color='black',label="LF(post-def)")
ax[4,0].tick_params(direction='out', width=2,labelbottom=False)
ax[4,0].set_title(cols[12],fontsize=10,fontweight='bold')


ax[4,1].plot(nlf_pre.index, ncnn_pre[cols[13]], color='red', ls='--',label="CNN(pre-def)")
ax[4,1].plot(nlf_pre.index, nlf_pre[cols[13]], color='orange',label="LF(pre-def)")
ax[4,1].plot(nlf_pre.index, ncnn_post[cols[13]], color='blue', ls='--',label="CNN(post-def)")
ax[4,1].plot(nlf_pre.index, nlf_post[cols[13]], color='black',label="LF(post-def)")
ax[4,1].tick_params(direction='out', width=2,labelbottom=False)
ax[4,1].set_title(cols[13],fontsize=10,fontweight='bold')

ax[4,2].plot(nlf_pre.index, ncnn_pre[cols[14]], color='red', ls='--',label="CNN(pre-def)")
ax[4,2].plot(nlf_pre.index, nlf_pre[cols[14]], color='orange',label="LF(pre-def)")
ax[4,2].plot(nlf_pre.index, ncnn_post[cols[14]], color='blue', ls='--',label="CNN(post-def)")
ax[4,2].plot(nlf_pre.index, nlf_post[cols[14]], color='black',label="LF(post-def)")
ax[4,2].tick_params(direction='out', width=2,labelbottom=False)
ax[4,2].set_title(cols[14],fontsize=10,fontweight='bold')


ax[5,0].plot(nlf_pre.index, ncnn_pre[cols[15]], color='red', ls='--',label="CNN(pre-def)")
ax[5,0].plot(nlf_pre.index, nlf_pre[cols[15]], color='orange',label="LF(pre-def)")
ax[5,0].plot(nlf_pre.index, ncnn_post[cols[15]], color='blue', ls='--',label="CNN(post-def)")
ax[5,0].plot(nlf_pre.index, nlf_post[cols[15]], color='black',label="LF(post-def)")
ax[5,0].tick_params(direction='out', width=2,labelbottom=False)
ax[5,0].set_title(cols[15],fontsize=10,fontweight='bold')


ax[5,1].plot(nlf_pre.index, ncnn_pre[cols[16]], color='red', ls='--',label="CNN(pre-def)")
ax[5,1].plot(nlf_pre.index, nlf_pre[cols[16]], color='orange',label="LF(pre-def)")
ax[5,1].plot(nlf_pre.index, ncnn_post[cols[16]], color='blue', ls='--',label="CNN(post-def)")
ax[5,1].plot(nlf_pre.index, nlf_post[cols[16]], color='black',label="LF(post-def)")
ax[5,1].tick_params(direction='out', width=2,labelbottom=False)
ax[5,1].set_title(cols[16],fontsize=10,fontweight='bold')

ax[5,2].plot(nlf_pre.index, ncnn_pre[cols[17]], color='red', ls='--',label="CNN(pre-def)")
ax[5,2].plot(nlf_pre.index, nlf_pre[cols[17]], color='orange',label="LF(pre-def)")
ax[5,2].plot(nlf_pre.index, ncnn_post[cols[17]], color='blue', ls='--',label="CNN(post-def)")
ax[5,2].plot(nlf_pre.index, nlf_post[cols[17]], color='black',label="LF(post-def)")
ax[5,2].tick_params(direction='out', width=2,labelbottom=False)
ax[5,2].set_title(cols[17],fontsize=10,fontweight='bold')

# add a big axis, hide frame

fig.add_subplot(111, frameon=False)

# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

plt.xlabel('Time step (h)',ha='center',fontsize=13,fontweight='bold')
plt.ylabel('Water depth (m)', rotation='vertical',fontsize=13,fontweight='bold')

fig.tight_layout()

# %%
fig.savefig('/home/cvssk/Carlisle_Resubmission/2015Event/CNN_Validation/WaterDepth_Comp_Pre_Post.png', dpi = 1200)

# %%
######NSE and RMSE calculation for Test 2

ns = []
rms = []
col_name = []

for i in range(len(cols)):
    sim = np.array(cnn_post[cols[i]])
    obs = np.array(lf_post[cols[i]])
    n = 1-sum((sim-obs)**2)/sum((obs-np.mean(obs))**2)
    rm = np.sqrt(np.mean((sim-obs)**2))
    ns.append(n)
    rms.append(rm)
    col_name.append(cols[i])

error_mat = {'location_name': col_name,
             'nas_sut': ns,
            'rmse':rms}

error = pd.DataFrame(error_mat)

error.to_csv('/home/cvssk/Carlisle_Resubmission/2015Event/CNN_Validation/CNN_PostDef_error_stats_cutoff_3.csv')


# %%
