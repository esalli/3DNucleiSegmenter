# Expands spheroids, and ground truths  by an integer factor in z-direction (i.e. adds slices). The interger factor is chosen so that the images will change to  approximately isotropic



import sys
sys.path.append("..") 
import numpy as np
import SimpleITK as sitk
import configSegmenter as c
import itk

expandFactor=((np.round(c.SPACINGSZ/c.SPACINGSXY)).astype(np.uintc))



i=0
for sph in c.SPHEROIDS:
    print("Processing spheroid ",sph)


    h=int(expandFactor[i])
    #expansion factor
    h=3
    
    hstr=str(h)
    #ground truth
    GT=sitk.ReadImage(c.PREPROCESSED_GT_DATADIR+sph+'_GT.nrrd')
    GTExpanded=sitk.Expand(GT,expandFactors=[1,1,h],interpolator=sitk.sitkNearestNeighbor)
    GTExpandedSparse=GTExpanded*0
    
    for j in range(1, GTExpanded.GetSize()[2]+1, h):
        GTExpandedSparse[:,:,j]=GTExpanded[:,:,j]
        
    
    
        
    itk_image = itk.GetImageFromArray(sitk.GetArrayFromImage(GTExpandedSparse), is_vector = GTExpandedSparse.GetNumberOfComponentsPerPixel()>1)
    itk_image.SetOrigin(GTExpandedSparse.GetOrigin())
    itk_image.SetSpacing(GTExpandedSparse.GetSpacing())   
    itk_image.SetDirection(itk.GetMatrixFromArray(np.reshape(np.array(GTExpandedSparse.GetDirection()), [3]*2)))

    
    filled = itk.morphological_contour_interpolator(itk_image,axis=2,UseDistanceTransform=False)
    filled.Update()
    
    
    # Back to a simpleitk image from the itk image
    filled_sitk_image = sitk.GetImageFromArray(itk.GetArrayFromImage(filled), isVector=filled.GetNumberOfComponentsPerPixel()>1)
    filled_sitk_image.SetOrigin(tuple(filled.GetOrigin()))
    filled_sitk_image.SetSpacing(tuple(filled.GetSpacing()))
    filled_sitk_image.SetDirection(itk.GetArrayFromMatrix(filled.GetDirection()).flatten()) 
    
     # use nearest neighbor interpolation for the bottom and top slices because itk.morphological_contour_interpolator does not seem to handle them properly
    filled_sitk_image[:,:,0]=GTExpanded[:,:,0]
    filled_sitk_image[:,:,GTExpanded.GetSize()[2]-1]=GTExpanded[:,:,GTExpanded.GetSize()[2]-1]

    #itk morphological_contour_interpolator seems to separate cells that are fused together. Next, let fix it by taking label value from NearestNeighbor interpolator in unlabeled voxels that change during morphological
    #closing operation
    filled_sitk_image_binary=filled_sitk_image>0
    
    filled_np=sitk.GetArrayFromImage(filled_sitk_image)
    GTExpanded_np=sitk.GetArrayFromImage(GTExpanded)
    
    filled_np_binary=filled_np>0
    closedImage=sitk.BinaryMorphologicalClosing(filled_sitk_image_binary,[2,2,2])
    
    closedImage_np=sitk.GetArrayFromImage(closedImage)
    
    differenceImage_np=(filled_np_binary-closedImage_np)!=0
    
    #where differenceImage is true we use NN, otherwise countour interplation
    filled_np=np.where(differenceImage_np,GTExpanded_np,filled_np)
    
    filled_sitk_image=sitk.GetImageFromArray(filled_np)
    
    filled_sitk_image.SetOrigin(GTExpanded.GetOrigin())
    filled_sitk_image.SetDirection(GTExpanded.GetDirection())
    filled_sitk_image.SetSpacing(GTExpanded.GetSpacing())
    
    sitk.WriteImage(filled_sitk_image,c.PREPROCESSED_GT_DATADIR+sph+'_GT_expanded_3_DT.nrrd',useCompression=True)
    
    #spheroids
    spheroid=sitk.ReadImage(c.PREPROCESSED_SPHEROIDS_DATADIR+sph+'_smoothed_spheroid.nrrd')
    spheroidExpanded=sitk.Expand(spheroid,expandFactors=(1,1,h))
    sitk.WriteImage(spheroidExpanded,c.PREPROCESSED_SPHEROIDS_EXPANDED_DATADIR+sph+'_smoothed_spheroid_expanded_3.nrrd',useCompression=True)
    
    i=i+1 
