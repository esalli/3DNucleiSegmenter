# 3DNucleiSegmenter

Segmentation of 3D Nuclei

## Prerequisities

A python environment with several python packages is required to run 3DNucleiSegmenter.
One way to setup the python environment is to install Anaconda  (www.anaconda.com), activate it and create an enviroment (called here 3DNS)

Create an environment and install packages
```
conda create -n 3DNS python=3.8
conda activate 3DNS
conda install git unzip numpy
conda install -c simpleitk simpleitk
conda install -c conda-forge pynrrd
conda install -c anaconda joblib
conda install -c conda-forge/label/cf202003 opencv
conda install  tifffile=2020.10.1
python -m pip install --upgrade pip
python -m pip install itk-morphologicalcontourinterpolation
conda install tensorflow-gpu
```
You may need to install specific version of tensorflow. If you don't have GPU hardware, install CPU only version ('tensorflow'). The CPU only version is too slow for training the models but segmentation using the provided prebuilt models should still be possible.  


Clone the repository from github
```
git clone https://github.com/esalli/3DNucleiSegmenter
```

## Acquire models
Download the prebuilt deep learning models from https://figshare.com/s/20531ed8e35c9a9b3add

Extract the zip into the root of 3DNucleiSegmenter
```
3DNucleiSegmenter$ unzip prebuiltModels.zip
```


## Acquire datasets

### Download the data of 12 spheroids

The data are availabe from https://figshare.com/s/8cd49182fcac2f37b6bc

Extract the spheroids.zip into data/preprocessedData: 
```
3DNucleiSegmenter/data/preprocessedData$ unzip 12spheroids.zip
```
### Download the Independet datasets (optional):

Liver spheroid from https://figshare.com/s/e64b456b00908d7a6751

Extarct the LiverSpheroid.zip  into data/independentData: 
```
3DNucleiSegmenter/data/independentData$ unzip LiverSpheroid.zip
```
Neurosphere ( Neurosphere_Dataset.zip) from https://sourceforge.net/projects/opensegspim/files/Sample%20Data/Neurosphere_Dataset.zip/download


Manual segmentations of other software from review_binary3dmasks.zip from  http://www.3d-cell-annotator.org/uploads/3/4/9/3/34939463/review_binary3dmasks.zip

Extract the zip files into data/independentData/3DCellAnnotator
```
3DNucleiSegmenter/data/independentData/3DCellAnnotator$ unzip Neurosphere_Dataset.zip 
3DNucleiSegmenter/data/independentData/3DCellAnnotator$ unzip review_binary3dmasks.zip 
```
Download Embryo (4May15FGFRionCD1_SU54_LM1.lsm) from https://ndownloader.figshare.com/files/5886078 into directory data/independentData/3DCellAnnotator

Download https://data.broadinstitute.org/bbbc/BBBC034/BBBC034_v1_dataset.zip and extract it into directory data/independentData/BroadInstitute

Download https://data.broadinstitute.org/bbbc/BBBC034/BBBC034_v1_DatasetGroundTruth.zip and extract it into directory data/independentData/BroadInstitute

## Preprocess datasets (optional)

Recreate the expanded datasets (optional as they are already included in the 12spheroids.zip)
```
3DNucleiSegmenter/preprocessCode$ python expandImages.py
```
Recreate the seeds (optional as they are already included in the 12spheroids.zip)
```
3DNucleiSegmenter/preprocessCode$ python nucleiMarkersIso.py
```
Preprocess the independent datasets (optional; required only to evaluate the independent datasets, not needed in training)
```
3DNucleiSegmenter/preprocessCode$ python preprocessIndependentDatasets.py
```

Create filtered images (optional; required only for the evaluation of the conventional watershed based baseline methods) 
```
3DNucleiSegmenter/preprocessCode$ python  bilateralFiltering.py
3DNucleiSegmenter/preprocessCode$ python gradientAnisotropicDiffusionFiltering.py
3DNucleiSegmenter/preprocessCode$ python nonlocalmeansFiltering.py
```


## Replication of the results of the system configurations

1. Perform masking. Navigate to the root and run:
```
python segmenterCode/mask_nuclei.py --dataset 0 --model_type 1
```
Model type is either 0,1,2 or 3 and refers to the use of 3D masks, 3D edge masks, 2D edge masks or seeds, respectively. Dataset can be 0 or 1 with 0 corresponding to the 12 spheroids and 1 to the independent datasets. Masks are generated to data/maskedData folder. 

2. Perform segmentation. Run:

```
python segmenterCode/segmentation.py --dataset 0 --ws_method 1 --mask_type M3DE --opt_mode 0 --save_segms 1
```

Dataset argument is the same as with mask_nuclei.py, ws_method refers to the use of either A (0), B (1) or C (2) watershed method, mask_type can be either M3D, M3DE, M2DE or S, opt_mode as 0 refers to the use of roundness score and as 1 to the use of optimal score and save_segms specifies whether the segmentation outputs are saved to data/segmentedData/spheroids. The script simultaneously runs evaluation and the evaluation scores are written to a numpy file which is located in data/evaluationScores. With the arguments specified above, the file would be named as B|M3DE|0|0.npy. To print out the scores, one can run:

```
import numpy as np
m = np.load("data/evaluationScores/A|M3DE|0|0.npy", allow_pickle = True).item()
scores = []
for key in m.keys():
    score = m[key][key]
    print(key, score)
```

The output should be the following:

```
1 (0.8257529521029587, 0.8214707346717058, 0.8247556355877674, 0.02666666666666667, 1.5)
5 (0.7443302871450623, 0.7524267015261906, 0.7583981167793887, 0.02877697841726619, 1.5)
6 (0.6719596901113202, 0.6559609405354259, 0.7096854189379703, 0.017937219730941704, 1)
...
```

## Training of U-Nets from scratch


1. Create training data via the ground truth masks. Run:

```
python SegmenterCode/training_data_creation.py --mask_type 3
```

Mask_type is either 0,1,2 or 3 corresponding to deep seeds, 3D masks, 2D edge masks or 3D edge masks. Training samples are saved to /data/trainingData/, inside a subfolder which specifies the mask type.    

2. Train 3D or 2D U-Net for masking. Run:

```
python SegmenterCode/training.py --mask_type 3 --model_type unet_3d --val_test_split 0,1
```

Mask_type arguments is the same as in training_data_creation.py, model_type is either unet or unet_3d, in practice unet_3d with all mask types except the 2D edge masks and val_test_split specifies the indices of spheroids which are used for validation and testing. The model name would be specified here as U_M3DE_2.h5, where 2 specifies the number of the testing spheroid, and saved along the configuration and history files in /models/U_M3DE.

