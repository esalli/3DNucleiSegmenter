import numpy as np

#spheroid numbers to be processed
SPHEROIDS=["1","2","3","4","5","6","7","8","9","10","11","12"]
SPACINGSXY=np.array([0.088,0.132,0.086,0.085,0.078,0.101,0.093,0.073,0.069,0.078,0.069,0.100])*4 #xy spacings of spheroids
SPACINGSZ=np.array([ 1.007,1.007,1.007,1.007,1.007,1.007,1.007,1.007,1.007,1.007,1.007,1.007 ])
 

#Source directory for the additional evaluation datasets (so called independent data)
SOURCE_ADD_EVAL_DATADIR="../data/independentData/"
#

#Directories for preprocessed files
PREPROCESSED_GT_DATADIR="../data/preprocessedData/GT/"
PREPROCESSED_SPHEROIDS_DATADIR="../data/preprocessedData/spheroids/"
PREPROCESSED_SPHEROIDS_EXPANDED_DATADIR="../data/preprocessedData/spheroids/"
MARKERS_DATADIR="../data/preprocessedData/seeds/"
PREPROCESSED_SPHEROIDS_FILTERED_DATADIR="../data/preprocessedData/spheroids/filtered/"

#Directory for the preprocessed additional evaluation datasets (so called independent data)
PREPROCESSED_ADD_EVAL_INDEPENDENT_DATADIR="../data/independentData/datasets/"
PREPROCESSED_ADD_EVAL_GT_DATADIR="../data/independentData/GT/"

#Temp directory
TEMP_DATADIR="../data/temp/"

#Directory for evaluation scores
EVALUATION_SCORES_DATADIR="../data/evaluationScores/"
 
