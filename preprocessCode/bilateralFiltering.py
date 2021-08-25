


#Bilater noise filtering for watershed



import sys
sys.path.append("..") 
import numpy as np
import SimpleITK as sitk
import configSegmenter as c



for domainSigma in [1,4]:
     for rangeSigma in [10,50]:
        for numberOfRangeGaussianSamples in [25,100]:
            for sph in c.SPHEROIDS:
                print("*********************************************************")
                print("Processing spheroid ",sph)
                print("using domainSigma",domainSigma,"rangeSigma ",rangeSigma, "numberOfRangeGaussianSamples",numberOfRangeGaussianSamples)
                print("*********************************************************")
                data = sitk.ReadImage(c.PREPROCESSED_SPHEROIDS_DATADIR+sph+'_smoothed_spheroid_expanded_3.nrrd') # read spheroids (original gray scale data)
                filtered=sitk.Bilateral(data,domainSigma=domainSigma,rangeSigma=rangeSigma,numberOfRangeGaussianSamples=numberOfRangeGaussianSamples)
                sitk.WriteImage(filtered,c.PREPROCESSED_SPHEROIDS_FILTERED_DATADIR+sph+'_smoothed_spheroid_expanded_bilaterallyfiltered_'+str(domainSigma)+'_'+str(rangeSigma)+'_'+str(numberOfRangeGaussianSamples)+'.nrrd',useCompression=True)


