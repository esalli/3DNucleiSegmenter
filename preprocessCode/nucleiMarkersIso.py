# Create almost isotropic  markers using expanded image matrix from ground truth images



import sys
sys.path.append("..") 
import numpy as np
import SimpleITK as sitk
import configSegmenter as c

for sph in c.SPHEROIDS:
    print("*********************************************************")
    print("Processing spheroid ",sph)
    print("*********************************************************")
    

    GT=sitk.ReadImage(c.PREPROCESSED_GT_DATADIR+sph+'_GT_expanded_3_DT.nrrd')

    emptyImage=GT*0
    markerImage2=emptyImage*0 #Image for seed

    
    # find maximum label number
    minMax=sitk.MinimumMaximumImageFilter()
    minMax.Execute(GT)
    maxGT=minMax.GetMaximum()
    # process all label ids

    for j in range(1,int(maxGT)+1):
        try:
          print("  Label ",j)
          moreThanOneComponentFound=0
          label=(GT==j)
          #number of components in label
          components = sitk.ConnectedComponentImageFilter()
          components.SetFullyConnected(True)
          seeds=components.Execute(label)

          if components.GetObjectCount()>0:
            print(" Connected components before erosion: ",components.GetObjectCount())
            if components.GetObjectCount()>1:
               print("********************************")
               print("Warning: More than one component")
               print("********************************")
               moreThanOneComponentFound=1
            erodedLabel=sitk.BinaryErode(label>0,[3,3,3])
            seeds=components.Execute(erodedLabel)
            print(" Connected components after [3,3,3] erosion: ",components.GetObjectCount())
            if components.GetObjectCount()!=1:
               print("Error: Marker split into two or more parts or disapperaed - trying lighter [1,1,1] erosion")
               erodedLabel=sitk.BinaryErode(label>0,[1,1,1])
               seeds=components.Execute(erodedLabel)
               print("  Connected components after lighter erosion: ",components.GetObjectCount())
               if components.GetObjectCount()!=1:
                  print("Error: Marker split into two or more parts (or disappeared altogether) even wth lighter erosion, trying [1,1,0] erosion only in xy-plane")
                  erodedLabel=sitk.BinaryErode(label>0,[1,1,0])
                  seeds=components.Execute(erodedLabel)
                  print("  Connected components after lighter erosion: ",components.GetObjectCount())
                  if components.GetObjectCount()!=1:
                       print("Error: Marker split into two or more parts (or disappeared altogether) even with lighter xy erosion - no erosion will be used")
                       erodedLabel=(label>0)
                       if moreThanOneComponentFound==1:
                         print('**********************************************************')
                         print('Warning: More than one component before and after erosion.')
                         print('Consider manual editing of the marker file')
                         print('**********************************************************')
            markerImage2=markerImage2+sitk.Cast(erodedLabel,markerImage2.GetPixelID())
          else:
            print("Warning: Label ",j," does not exist")
        except RuntimeError as e:
            print(e)
            raise
        
    sitk.WriteImage(markerImage2,c.MARKERS_DATADIR+sph+'_hugeMIiso_nearest_3_DT.nrrd',useCompression=True)
    
