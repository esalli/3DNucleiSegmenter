# ============================================================
# Evaluation of bilateral filtering +  
# conventional watershed
# Paths of data files are specified at ../configSegmenter
#
# E.Salli 2021
# ============================================================




import sys
import SimpleITK as sitk
import glob

sys.path.append("..")
sys.path.append("../evaluationCode")
sys.path.append("../segmenterCode")
import configSegmenter as c

import segmentation as segmenter_evaluator

saveResultsToFile=True

if saveResultsToFile==True:
 f=open(c.EVALUATION_SCORES_DATADIR+'completeresults_expanded_bilateralfilteringwatershed.txt', 'a') 
else: 
  f=sys.stdout

print("spheroid,filename,ws_level,pq,aji,ji,nndp", file=f)  

if 1==1:

 aji_all=[]
 pq_all=[]
 nndp_all=[]
 for sph in c.SPHEROIDS:
  print("*********************************************************")
  print("Processing spheroid ",sph)
  print("*********************************************************")

    #spheroid = sitk.ReadImage(c.PREPROCESSED_SPHEROIDS_DATADIR+sph+'_spheroid_expanded.nrrd')

  GTfilename=c.PREPROCESSED_GT_DATADIR+sph+'_GT.nrrd'

  filelist=        glob.glob(c.PREPROCESSED_SPHEROIDS_FILTERED_DATADIR+sph+'_smoothed_spheroid_expanded_bilaterallyfiltered*.nrrd')

  for filename in filelist:
   print("Processing filename ",filename)
   spheroid = sitk.ReadImage(filename)
    
   thresh_filter=sitk.OtsuThresholdImageFilter()
   thresh_filter.SetInsideValue(0)
   thresh_filter.SetOutsideValue(1)
   thresh_img = thresh_filter.Execute(spheroid)
   thresh_value = thresh_filter.GetThreshold()
   spheroid = spheroid > thresh_value
    
   for ws_level in [0.50,0.75,1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0,4.0,5.0]:
    print("Processing ws_level ",ws_level)


    segm=segmenter_evaluator.segment_nuclei(mask_img=spheroid, seeds_img = [], thold = 0.5, ws_method = 1, ws_level = ws_level)
   

    [aji, pq, ji, nndp] = segmenter_evaluator.evaluate(segm,GTfilename)
     
    print(sph,',',filename,',',ws_level,',',pq,',',aji,',',ji,',',nndp,file=f)  

