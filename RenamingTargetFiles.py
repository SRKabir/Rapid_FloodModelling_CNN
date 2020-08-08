#%%
import os

InDir = '/home/cvssk/Carlisle_Resubmission/2005Event/Junk/'
OutDir= '/home/cvssk/Carlisle_Resubmission/2005Event/Target/'
key = 'Run9-'
files = []

##PROCESS TARGET DATA (Y_PARAM)
files += [each for each in os.listdir(InDir) if each.endswith('.wd')]
files.sort()
print(len(files))

#%%
for i in files:
    f = i
    f = f.split('-')[1]
    name = OutDir+key+f
    os.rename(InDir+i, name)



# %%
import rasterio as rio 
import matplotlib.pyplot as plt
src = rio.open( '/home/cvssk/Carlisle_Resubmission/2005Event/Target/Run9-0048.wd')
file = src.read(1)
plt.imshow(file)
plt.show()


# %%
