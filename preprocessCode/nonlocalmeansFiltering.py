
#non-local means noise filtering for watershed

#see https://itk.org/pipermail/community/2016-April/011208.html

import sys
sys.path.append("..") 
import numpy as np
import SimpleITK as sitk
import configSegmenter as c



for patchRadius in [4,6]:
     for samplePatches in [50,100]:
        for numberOfIterations in [1]:
            for sph in c.SPHEROIDS:
                print("*********************************************************")
                print("Processing spheroid ",sph)
                print("using patchRadius",patchRadius,"samplePatches ",samplePatches, "numberOfIterations",numberOfIterations)
                print("*********************************************************")
                data = sitk.ReadImage(c.PREPROCESSED_SPHEROIDS_DATADIR+sph+'_smoothed_spheroid_expanded_3.nrrd') # read spheroids (original gray scale data)
                filter=sitk.PatchBasedDenoisingImageFilter()
                filter.SetPatchRadius(patchRadius)
                filter.SetNumberOfSamplePatches(samplePatches)
                filter.SetNumberOfIterations(numberOfIterations)
                filtered=filter.Execute(data)
                sitk.WriteImage(filtered,c.PREPROCESSED_SPHEROIDS_FILTERED_DATADIR+sph+'_smoothed_spheroid_expanded_nonlocalmeansfiltered_'+str(patchRadius)+'_'+str(samplePatches)+'_'+str(numberOfIterations)+'.nrrd',useCompression=True)















