####################################
# Preprocess independent datasets  #
####################################


import numpy as np

import sys
sys.path.append("..") 

import SimpleITK as sitk
import configSegmenter as c

from scipy.ndimage import gaussian_filter

import tifffile



independentinfo=[]


independent={
  "independentFilename": "LiverSpheroid/hepa1a.nii",
  "GTFilename": "LiverSpheroid/GT_hepa1a.nii",  
  "expandFactor": 3
}

independentinfo.append(independent)

independent={
  "independentFilename": "3DCellAnnotator/OriginalStack.tif",
  "GTFilename": "3DCellAnnotator/Review_Binary3DMasks/Neurosphere/Neurosphere_GroundTruth.tif",  
  "expandFactor": 1,
  "spacing": (1.0,1.0,1.0)
}
independentinfo.append(independent)

independent={
  "independentFilename": "3DCellAnnotator/4May15FGFRionCD1_SU54_LM1.lsm",
  "GTFilename": "3DCellAnnotator/Review_Binary3DMasks/Embryo/Embryo_GroundTruth.tif",  
  "expandFactor": 1,
  "useTifffile": True,
}

independentinfo.append(independent)



independent={
  "independentFilename": "BroadInstitute/AICS_12_134_C=2.tif",
  "GTFilename": "BroadInstitute/ground_truth_segmented.tif",    
  "expandFactor": 1,
  "spacing": (0.65,0.65,2.9),
  "intensityClipping": 255,
}

independentinfo.append(independent)

###You can add your own data here by uncommenting and modifying the following lines
###If you don't have ground truth, don't specify GTFilename. An empty ground truth will be created as placeholder

independent={
  "independentFilename": "BroadInstitute/AICS_12_134_C=2.tif",
  "expandFactor": 1,
  "spacing": (0.65,0.65,2.9),
  "intensityClipping": 255,
}
independentinfo.append(independent)





# Independent data used

for sph in range(0,len(independentinfo)):


    
  if "useTifffile" in independentinfo[sph] and independentinfo[sph]["useTifffile"]==True:
    
    with tifffile.TiffFile(c.SOURCE_ADD_EVAL_DATADIR+independentinfo[sph]["independentFilename"]) as tif:
     volume = tif.asarray()
     lsm_metadata = tif.lsm_metadata

     xspacing=lsm_metadata['VoxelSizeX']*10000000
     yspacing=lsm_metadata['VoxelSizeY']*10000000
     zspacing=lsm_metadata['VoxelSizeZ']*10000000

     independent=sitk.GetImageFromArray(volume[0,:,0,:,:])           
     independent.SetSpacing([xspacing,yspacing,zspacing])                  

    
    
  else:
    independent=sitk.ReadImage(c.SOURCE_ADD_EVAL_DATADIR+independentinfo[sph]["independentFilename"])

  independent.SetOrigin([0.0,0.0,0.0])
  independent.SetDirection([1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0])

  independent=sitk.Cast(independent,sitk.sitkUInt16)
  if "spacing" in independentinfo[sph]:
        independent.SetSpacing(independentinfo[sph]["spacing"])
  
  if "GTFilename" in independentinfo[sph]:
    GT=sitk.ReadImage(c.SOURCE_ADD_EVAL_DATADIR+independentinfo[sph]["GTFilename"])
  else:
    GT=0*sitk.Cast(independent,sitk.sitkUInt8)

  GT.SetOrigin([0.0,0.0,0.0])
  GT.SetDirection([1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0])

  GT.SetSpacing(independent.GetSpacing())
 
    
    
  if "intensityClipping" in independentinfo[sph]:
    #independent=sitk.Cast((independent<=independentinfo[sph]["intensityClipping"]),independent.GetPixelID())*independent
    independent=sitk.IntensityWindowing(independent,0.0,independentinfo[sph]["intensityClipping"],0.0,independentinfo[sph]["intensityClipping"])
  #downsample
  new_size = [256,256,independent.GetSize()[2]]
    
  #resampling code snippet source: https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/70_Data_Augmentation.ipynb
  referenceImage = sitk.Image(new_size, independent.GetPixelIDValue()  )

  inputOrigin=independent.GetOrigin()
  newOrigin=independent.GetOrigin()
  



                              
  
  referenceImage.SetOrigin(newOrigin)
  referenceImage.SetDirection(independent.GetDirection())
  referenceImage.SetSpacing([sz*spc/nsz for nsz,sz,spc in zip(new_size, independent.GetSize(), independent.GetSpacing())])

  inputSpacing=independent.GetSpacing()
  
  outputSpacing=referenceImage.GetSpacing()


      
  y=list(newOrigin)

  #newOrigin=[inputOrigin[d] + 0.5 * (outputSpacing[d] - inputSpacing[d])]  for d in range(2)
  y[0]=inputOrigin[0] + 0.5 * (outputSpacing[0] - inputSpacing[0])
  y[1]=inputOrigin[1] + 0.5 * (outputSpacing[1] - inputSpacing[1])
  newOrigin = tuple(y)




  referenceImage.SetOrigin(newOrigin)
              
  # Resample after Gaussian smoothing.
  if (independent.GetSize()[0]>256):
     downsampledImage=sitk.Resample(sitk.DiscreteGaussian(independent, variance=((independent.GetSpacing()[0]**2),(independent.GetSpacing()[1]**2),0.0)),referenceImage,interpolator=sitk.sitkLinear)
  else:
     downsampledImage=sitk.Resample(independent,referenceImage,interpolator=sitk.sitkLinear)

  X=sitk.GetArrayFromImage(sitk.PermuteAxes(downsampledImage,[2,0,1]))
  X_norm = (X-np.min(X))/(np.max(X)-np.min(X))
  # Remove low intensity slices

  #slices starting from top

  for k in np.arange(X.shape[2]):

            slic = X_norm[:, :, k]
            slic = gaussian_filter(slic, sigma = 3)
            if np.max(slic) < 0.1:
                print("setting slice zero")
                print(np.max(slic))
                X[:, :, k]=0
            else:
                #Background slices cannot be in the middle od non-backgroun slices, so exit 
                break      
  #slices starting from bottom
  for k in np.arange(X.shape[2]-1,-1,-1):

            slic = X_norm[:, :, k]
            slic = gaussian_filter(slic, sigma = 3)

            if np.max(slic) < 0.1:
                print("setting slice zero")
                print(np.max(slic))
                X[:, :, k]=0
            else:
                #Background slices cannot be in the middle od non-backgroun slices, so exit 
                break           
    

  downsampledImage_bg0 = sitk.PermuteAxes(sitk.GetImageFromArray(X),[1,2,0])
  downsampledImage_bg0.SetSpacing(downsampledImage.GetSpacing())
  downsampledImage_bg0.SetDirection(downsampledImage.GetDirection())
  downsampledImage_bg0.SetOrigin(downsampledImage.GetOrigin())

  independentExpanded=sitk.Expand(downsampledImage_bg0,expandFactors=(1,1,independentinfo[sph]["expandFactor"]))
  sitk.WriteImage(independentExpanded,c.PREPROCESSED_ADD_EVAL_INDEPENDENT_DATADIR+str(sph+1)+'_independent_expanded.nrrd',useCompression=True)
  sitk.WriteImage(GT,c.PREPROCESSED_ADD_EVAL_GT_DATADIR+str(sph+1)+'_GT.nrrd',useCompression=True)
