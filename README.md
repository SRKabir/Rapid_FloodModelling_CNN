This is an implementation of Convolutional Neural Network-based rapid fluvial flood modelling system.

To make sure you have the same libraries please clone/download the repo and 'cd' to directory run the following command:

conda env create -f environment.yml

and check that the new environment is available and activate it:

conda info --envs

conda activate bad_env

You may have to change some of the directory path in the python files to find the training and target data and output file locations.

The LISFLOOD-FP generated inundation flies could not be uploaded. Therefore, data_pre_process() and data() functions are unusable.

The training and testing files are already provided in the Data folder, meaning data data_pre_process() and data() functions are obsolete.

The CNN training is run on GPU. However, if a CPU is used, training time will be longer.

'Carlisle_InunMod.py' contains model functions and 'Depth_Simulations.py' can be used to follow the steps. Note, the training target (.h5) files are large and due to github file limitations couldn't be uploaded at a time. concatenate 'Y_Train_Subset1.h5' and 'Y_Train_Subset2.h5' to make it full. It is already done in 'Depth_Simulations.py'.

Hyperparameter optimisation scripts are used to find model parameters

Feature_Importance.py script is to rank inputs according to importance.

SVR.py fits Support Vector Regression models to the validation locations.

