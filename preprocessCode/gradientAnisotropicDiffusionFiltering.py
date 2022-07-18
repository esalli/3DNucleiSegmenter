


#Gradient anistropic diffusion filtering



import sys
sys.path.append("..") 
import numpy as np
import SimpleITK as sitk
import configSegmenter as c



domainSigma=1
rangeSigma=50
numberOfRangeGaussianSamples=50

for timeStep in [0.015,0.0625]:
  for conductanceParameter in [1,4]:
    for numberOfIterations in [5,10,20]:
            for sph in c.SPHEROIDS:
                print("*********************************************************")
                print("Processing spheroid ",sph)
                print("using conductanceParameter",conductanceParameter," numberOfIterations",numberOfIterations)
                print("*********************************************************")
                data = sitk.ReadImage(c.PREPROCESSED_SPHEROIDS_DATADIR+sph+'_smoothed_spheroid_expanded_3.nrrd') # read spheroids (original gray scale data)
                filtered=sitk.GradientAnisotropicDiffusion(data,timeStep=timeStep,conductanceParameter =conductanceParameter ,numberOfIterations=numberOfIterations)
                sitk.WriteImage(filtered,c.PREPROCESSED_SPHEROIDS_FILTERED_DATADIR+sph+'_smoothed_spheroid_expanded_anisotropicfiltered_'+str(int(timeStep*10000))+'_'+str(conductanceParameter)+'_'+str(numberOfIterations)+'.nrrd',useCompression=True)


