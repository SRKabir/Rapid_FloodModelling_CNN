# This is an implementation of Convolutional Neural Network-based rapid fluvial flood modelling system.

This implemention trains on the outputs of a 2D hydraulic model (LISFLOOD-FP). LSIFLOOD-FP is a Bristol University software for inundaiton modelling and can be used freely for research purposes.

To acquire this software please consult http://www.bristol.ac.uk/geography/research/hydrology/models/lisflood/

Once you have the software, you would need to execute it for different boundary conditions.

The supporting LISFLOOD-FP files (.asc, .par, .bci, .bdy files are provided in the 'Data' folder)

## Example
place all LISFLOOD-FP software files and Carlisle_5m.asc, Carlisle.bci, Carlisle_run1.bdy, Carlisle_run1.par files in the same directory.
cd to the folder from command prompt/terminal. 

Then run the executable file, e.g.,

> lisfloodRelease_double_v631.exe -v Carlisle_run1.par

Run this for all .par files for generating training and testing data.

Outputs from Run2 to Run9 should be placed in the /../Target directory. These are the outputs from synthetic hydrographs and used for training.
and output files from Run1 should be placed in the /../Run1 directory. These are the outputs of 2005 event and used for testing model performance.

# Training the CNN model
To make sure you have the same libraries please clone/download the repo and 'cd' to directory run the following command:

> conda env create -f environment.yml

and check that the new environment is available and activate it:

> conda info --envs

> conda activate bad_env

You may have to change some of the directory paths in the python files to find the training and target data and output file locations.

All required CNN functions are in the 'InunMod_v1.py' file.

In the bottom of the script, example is provided.

Essentially that is the only file required for all CNN opetaions, e.g., data preprocessing, training, predictions etc.
Hyperparameter optimisation, input feature importance, plotting, and Support Vector Regression model scripts are also provided

Please note, currently the codes are not optimized. Therefore, you would need to change the directory names and files. However, necessary comments
are provided for understanding the codes.

Please contact s.r.kabir@lboro.ac.uk for any issues.

