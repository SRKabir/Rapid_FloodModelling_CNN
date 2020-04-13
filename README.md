This is an implementation of Convolutional Neural Network-based rapid fluvial flood modelling system.

To make sure you have the same libraries please clone/download the repo and 'cd' to directory 
run the following command:

conda env create -f environment.yml

and check that the new environment is available and activate it:

conda info --envs

conda activate bad_env


You may have to change some of the directory path in the python files to find the training and target data and output file locations.

The LISFLOOD-FP generated inundation flies could not be uploaded. Therefore, data_pre_process() and data() functions are unusable.

The training and testing files are already provided in the Data folder, meaning data data_pre_process() and data() functions are obsolete.

The CNN training is run on GPU. However, if a CPU is used, training time will be longer.





