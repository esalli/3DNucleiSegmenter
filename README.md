# 3DNucleiSegmenter

A software to segment 3D Nuclei imaged using, e.g., the confocal microscopy

![3DNS](visual_demon_v2.png)

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
conda install -c anaconda scikit-image
```
You may need to install specific version of the TensorFlow. If you don't have GPU hardware, install CPU only version ('tensorflow'). The CPU only version is too slow for training the models but segmentation using the provided prebuilt models should still be possible.  


Clone the repository from github
```
git clone https://github.com/esalli/3DNucleiSegmenter
```

## Acquire models
Download the prebuilt deep learning models from  https://doi.org/10.6084/m9.figshare.1643915

Extract the zip into the root of 3DNucleiSegmenter
```
3DNucleiSegmenter$ unzip prebuiltModels.zip
```


## Acquire datasets

### Download the data of 12 spheroids

The data are available 
https://doi.org/10.6084/m9.figshare.16438314 

Extract the spheroids.zip into data/preprocessedData: 
```
3DNucleiSegmenter/data/preprocessedData$ unzip 12spheroids.zip
```
### Download the Independet datasets (optional):

Liver spheroid from
https://doi.org/10.6084/m9.figshare.16438185

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

Create filtered images (optional; required only for the evaluation of the conventional (without deep learning) watershed based baseline methods) 
```
3DNucleiSegmenter/preprocessCode$ python  bilateralFiltering.py
3DNucleiSegmenter/preprocessCode$ python gradientAnisotropicDiffusionFiltering.py
3DNucleiSegmenter/preprocessCode$ python nonlocalmeansFiltering.py
```
## Perform segmentation using conventional  methods (optional; required only for the evaluation of the conventional methods)
Run
```
3DNucleiSegmenter/evaluationCode$ python  evaluationBilateralFilteringWatershed.py
3DNucleiSegmenter/evaluationCode$ python evaluationAnisotropicFilteringWatershed.py
3DNucleiSegmenter/evaluationCode$ python evaluationNonlocalMeansFilteringWatershed.py
3DNucleiSegmenter/evaluationCode$ python evaluationNoFilteringWatershed.py
```
The evaluation scores are saved to files 3DNucleiSegmenter/data/evaluationScores/*.txt

## Replication of the results of the proposed system configurations

1. Perform masking. Navigate to the root and run:
```
3DNucleiSegmenter$ python segmenterCode/mask_nuclei.py --dataset 0 --model_type 1
```
Model type is either 0,1,2 or 3 and refers to the use of 3D masks, 3D edge masks, 2D edge masks or seeds, respectively. Dataset can be 0 or 1 with 0 corresponding to the 12 spheroids and 1 to the independent datasets. Masks are generated to data/maskedData folder. 

2. Perform segmentation. Run:

```
3DNucleiSegmenter$ python segmenterCode/segmentation.py --dataset 0 --ws_method 1 --mask_type M3DE --opt_mode 0 --save_segms 1
```

Dataset argument is the same as with mask_nuclei.py, ws_method refers to the use of either A (0), B (1) or C (2) watershed method, mask_type can be either M3D, M3DE, M2DE or S, opt_mode as 0 refers to the use of roundness score and as 1 to the use of optimal score and save_segms specifies whether the segmentation outputs are saved to data/segmentedData/spheroids when "--dataset 0" option is used. The script simultaneously runs evaluation and the evaluation scores are written to a numpy file which is located in data/evaluationScores. With the arguments specified above, the file would be named as B|M3DE|0|0.npy. To print out the scores, one can run:

```
import numpy as np
m = np.load("data/evaluationScores/B|M3DE|0|0.npy", allow_pickle = True).item()
scores = []
for key in m.keys():
    score = m[key][key]
    print(key, score)
```

The output should be the following (order of the lines may differ):

```
1 (0.8257529521029587, 0.8214707346717058, 0.8247556355877674, 0.02666666666666667, 1.5)
5 (0.7443302871450623, 0.7524267015261906, 0.7583981167793887, 0.02877697841726619, 1.5)
6 (0.6719596901113202, 0.6559609405354259, 0.7096854189379703, 0.017937219730941704, 1)
...
```

## Training of U-Nets from scratch


1. Create training data via the ground truth masks. Run:

```
3DNucleiSegmenter$ python segmenterCode/training_data_creation.py --mask_type 3
```

Mask_type is either 0,1,2 or 3 corresponding to deep seeds, 3D masks, 2D edge masks or 3D edge masks. Training samples are saved to /data/trainingData/, inside a subfolder which specifies the mask type.    

2. Train 3D or 2D U-Net for masking. Run:

```
3DNucleiSegmenter$ python segmenterCode/training.py --mask_type 3 --model_type unet_3d --val_test_split 0,1 --batch_size 4
```

Mask_type arguments is the same as in training_data_creation.py, model_type is either unet or unet_3d, in practice unet_3d with all mask types except the 2D edge masks and val_test_split specifies the indices of spheroids which are used for validation and testing. The model name would be specified here as U_M3DE_2.h5, where 2 specifies the id number of the testing spheroid, and saved along the configuration and history files in models/U_M3DE. To ensure that the same arguments are used as in our experiments, see the config.npy files in the prebuiltModels. Unfortunately, even with the same seeds we can not ensure that exactly the same network configurations will be trained.

## Segmenting new (own) datasets

The recommended file format for own datasets is NifTi or nrrd and the recommended DirectionMatrix is (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) (as returned by volume.GetDirection() of SimpleITK). At this stage, it is required to modify the file preprocessCode/preprocessAdditionalDatasets.py, or its modified copy, to segment new data. Follow the instruction given in the comments of the file to add datasets into the processing pipeline.  After editing the file, check that directories 
```
3DNucleiSegmenter/data/independentData/datasets
3DNucleiSegmenter/data/independentData/GT
3DNucleiSegmenter/data/independentData/data/maskedData/*
```
are empty from previous experiments and run
```
3DNucleiSegmenter/preprocessCode$ python preprocessIndependentDatasets.py
```
or your own modified preprocessing script.
Check the results in data/independentData/datasets and data/independentData/GT. Especially, you may want to check that the resampling to the size of [256,256] is performed on correct axis. The size of GT files should match the original data. You can view the nrrd files by 3D Slicer (www.slicer.org).  The orientation of the new datasets and GT volumes may differ from the orientation of original volume (depending on the DirectionMatrix of original files). Note that if there is no ground truth specified, the ground truth image in GT will contain only zeroes.

After the preprocessing, perform masking and segmentation, e.g.:
```
3DNucleiSegmenter$ python segmenterCode/mask_nuclei.py --dataset 1 --model_type 1
3DNucleiSegmenter$ python segmenterCode/segmentation.py --dataset 1 --ws_method 1 --mask_type M3DE --opt_mode 0 --save_segms 1
```
The results will be saved into  the directory 3DNucleiSegmenter/data/segmentedData/datasets. Note that the orientations of the results will match the preprocessed files in 3DNucleiSegmenter/data/independentData/datasets and 3DNucleiSegmenter/data/independentData/GT.  A more straightforward way to segment new datasets will be added later to 3DNucleiSegmenter.


## Citing

If you use 3DNucleiSegmenter or datasets (12 spheroids and their segmentations) in your research, please cite the following paper:
Kaseva, T., Omidali, B., Hippeläinen, E., Mäkelä T., Wilppu U., Sofiev, A.,  Merivaara, A., Yliperttula M., Savolainen, S. & Salli, E.  Marker-controlled watershed with deep edge emphasis and optimized H-minima transform for automatic segmentation of densely cultivated 3D cell nuclei. BMC Bioinformatics 23, 289 (2022). https://doi.org/10.1186/s12859-022-04827-3
 

