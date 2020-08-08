
#%%
from sklearn.metrics import recall_score, precision_score, f1_score
import rasterio as rio

# %%

lf = rio.open('/home/cvssk/Carlisle_Resubmission/2015Event/Classification_results/CNN_PostDef/Run1-0076.wd').read(1)
cnn = rio.open('/home/cvssk/Carlisle_Resubmission/2015Event/Classification_results/CNN_PostDef/CNN_2015_076.asc').read(1)

lf[lf < 0.3] = 0
lf[lf != 0] = 1

cnn[cnn < 0.3] = 0
cnn[cnn != 0] = 1

pred = np.ndarray.flatten(cnn)
pred = pred.reshape(pred.shape[0],1)

###count elements
pred_num_zeros = (pred == 0).sum()
pred_num_ones = (pred == 1).sum()

ref = np.ndarray.flatten(lf)
ref = ref.reshape(ref.shape[0],1)
ref_num_zeros = (ref == 0).sum()
ref_num_ones = (ref == 1).sum()

precision = precision_score(ref, pred)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(ref, pred)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(ref, pred)
print('F1 score: %f' % f1)

classification_acc = pd.DataFrame({'Precision':[precision],
                                'Recall':[recall],
                                'F1': [f1]})

classification_acc.to_csv('/home/cvssk/Carlisle_Resubmission/2015Event/Classification_results/CNN_PostDef_76_acc.csv')



# %%
###Histogram

cnn45 = rio.open('/home/cvssk/Carlisle_Resubmission/2015Event/Classification_results/CNN_PreDef/CNN_2005_045.asc')
cnn76 = rio.open('/home/cvssk/Carlisle_Resubmission/2015Event/Classification_results/CNN_PreDef/CNN_2005_076.asc')
cnn124 = rio.open('/home/cvssk/Carlisle_Resubmission/2015Event/Classification_results/CNN_PreDef/CNN_2005_124.asc')
cnn220 = rio.open('/home/cvssk/Carlisle_Resubmission/2015Event/Classification_results/CNN_PreDef/CNN_2005_220.asc')

lf45 = rio.open('/home/cvssk/Carlisle_Resubmission/2015Event/Classification_results/CNN_PreDef/Run1-0045.wd')
lf76 = rio.open('/home/cvssk/Carlisle_Resubmission/2015Event/Classification_results/CNN_PreDef/Run1-0076.wd')
lf124 = rio.open('/home/cvssk/Carlisle_Resubmission/2015Event/Classification_results/CNN_PreDef/Run1-0124.wd')
lf220 = rio.open('/home/cvssk/Carlisle_Resubmission/2015Event/Classification_results/CNN_PreDef/Run1-0220.wd')

df45 = abs(cnn45.read(1)-lf45.read(1))
df76 = abs(cnn76.read(1)-lf76.read(1))
df124 = abs(cnn124.read(1)-lf124.read(1))
df220 = abs(cnn220.read(1)-lf220.read(1))

df45 = np.ndarray.flatten(df45)
df76 = np.ndarray.flatten(df76)
df124 = np.ndarray.flatten(df124)
df220 = np.ndarray.flatten(df220)

#%%
df = [df45, df76, df124, df220]

dt = pd.DataFrame(df[0])
dt.columns = ['Error']
Q1 = dt.quantile(.005, axis = 0)
Q2 = dt.quantile(.995, axis = 0)
dt45 = dt[~(dt['Error'] < Q1.Error)]
dt45 = dt45[~(dt45['Error'] > Q2.Error)]
dt45['Error'] = dt45['Error'].astype(float).round(1)
Qls45 = pd.DataFrame({'q1': [Q1.Error], 'q2': [Q2.Error]})

dt = pd.DataFrame(df[1])
dt.columns = ['Error']
Q1 = dt.quantile(.005, axis = 0)
Q2 = dt.quantile(.995, axis = 0)
dt76 = dt[~(dt['Error'] < Q1.Error)]
dt76 = dt76[~(dt76['Error'] > Q2.Error)]
dt76['Error'] = dt76['Error'].astype(float).round(1)
Qls76 = pd.DataFrame({'q1': [Q1.Error], 'q2': [Q2.Error]})

dt = pd.DataFrame(df[2])
dt.columns = ['Error']
Q1 = dt.quantile(.005, axis = 0)
Q2 = dt.quantile(.995, axis = 0)
dt124 = dt[~(dt['Error'] < Q1.Error)]
dt124 = dt124[~(dt124['Error'] > Q2.Error)]
dt124['Error'] = dt124['Error'].astype(float).round(1)
Qls124 = pd.DataFrame({'q1': [Q1.Error], 'q2': [Q2.Error]})

dt = pd.DataFrame(df[3])
dt.columns = ['Error']
Q1 = dt.quantile(.005, axis = 0)
Q2 = dt.quantile(.995, axis = 0)
dt220 = dt[~(dt['Error'] < Q1.Error)]
dt220 = dt220[~(dt220['Error'] > Q2.Error)] 
dt220['Error'] = dt220['Error'].astype(float).round(1)
Qls220 = pd.DataFrame({'q1': [Q1.Error], 'q2': [Q2.Error]})

# %%
qls = [Qls45,Qls76,Qls124,Qls220]
QLS = pd.concat(qls)
QLS.to_csv('/home/cvssk/Carlisle_Resubmission/2015Event/ErrorDist_Results/Q99.csv')
df = [dt45['Error'],dt76['Error'],dt124['Error'],dt220['Error']]
dt45.to_csv('/home/cvssk/Carlisle_Resubmission/2015Event/ErrorDist_Results/dt45_q99.csv')
dt76.to_csv('/home/cvssk/Carlisle_Resubmission/2015Event/ErrorDist_Results/dt76_q99.csv')
dt124.to_csv('/home/cvssk/Carlisle_Resubmission/2015Event/ErrorDist_Results/dt124_q99.csv')
dt220.to_csv('/home/cvssk/Carlisle_Resubmission/2015Event/ErrorDist_Results/dt220_q99.csv')

#%%
times = ['10:00 hours December 5 2015', '18:00 hours December 5 2015', '06:00 hours December 6 2015',
         '06:00 hours December 7 2015']

#%%
d = pd.DataFrame(df[2]).describe().reset_index()
d.to_csv('/home/cvssk/Carlisle_Resubmission/2015Event/ErrorDist_Results/dt124_q99_des_stats.csv')

d = pd.DataFrame(dt124.apply(pd.value_counts).reset_index())
d.to_csv('/home/cvssk/Carlisle_Resubmission/2015Event/ErrorDist_Results/dt124_q99_counts.csv')

#%%
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib import colors

fig, axs = plt.subplots(2, 2, figsize=(10,10),dpi = 200,facecolor='w', edgecolor='k')


# We can also normalize our inputs by the total number of counts
N0, bins0, patches0=axs[0,0].hist(df[0],align='right')
axs[0,0].yaxis.set_major_formatter(PercentFormatter(xmax=len(df[0])))
axs[0,0].tick_params(direction='out', width=2,labelbottom=True)
axs[0,0].set_title(times[0],fontsize=11,fontweight='bold')


N1, bins1, patches1=axs[0,1].hist(df[1],align='mid')
axs[0,1].yaxis.set_major_formatter(PercentFormatter(xmax=len(df[1])))
axs[0,1].tick_params(direction='out', width=2,labelbottom=True)
axs[0,1].set_title(times[1],fontsize=11,fontweight='bold')

N2, bins2, patches2=axs[1,0].hist(df[2],align='mid')
axs[1,0].yaxis.set_major_formatter(PercentFormatter(xmax=len(df[2])))
axs[1,0].tick_params(direction='out', width=2,labelbottom=True)
axs[1,0].set_title(times[2],fontsize=11,fontweight='bold')

N3, bins3, patches3=axs[1,1].hist(df[3],align='mid')
axs[1,1].yaxis.set_major_formatter(PercentFormatter(xmax=len(df[3])))
axs[1,1].tick_params(direction='out', width=2,labelbottom=True)
axs[1,1].set_title(times[3],fontsize=11,fontweight='bold')

# add a big axis, hide frame

fig.add_subplot(111, frameon=False)

# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

plt.xlabel('Error (m)',ha='center',fontsize=13,fontweight='bold')
plt.ylabel('Count (%)', rotation='vertical',fontsize=13,fontweight='bold')

fig.tight_layout()

# %%
fig.savefig('/home/cvssk/Carlisle_Resubmission/2015Event/ErrorDist_Results/ErrorHistogram.png', dpi = 1200)


# %%


