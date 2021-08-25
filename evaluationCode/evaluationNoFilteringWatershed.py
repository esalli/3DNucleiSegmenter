# ============================================================
# Evaluation of bi
# conventional watershed
# Paths of data files are specified at ../configSegmenter
#
# E.Salli 2021
# ============================================================



import sys
import SimpleITK as sitk

sys.path.append("..")
sys.path.append("../evaluationCode")
sys.path.append("../segmenterCode")
import configSegmenter as c


import segmentation as segmenter_evaluator

saveResultsToFile=True


if saveResultsToFile==True:
 f=open(c.EVALUATION_SCORES_DATADIR+'completeresults_expanded_nofilteringwatershed.txt', 'a') 
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
   filename = c.PREPROCESSED_SPHEROIDS_DATADIR+sph+'_smoothed_spheroid_expanded_3.nrrd'
   spheroid = sitk.ReadImage(filename)

   GTfilename=c.PREPROCESSED_GT_DATADIR+sph+'_GT.nrrd'

   thresh_filter=sitk.OtsuThresholdImageFilter()
   thresh_filter.SetInsideValue(0)
   thresh_filter.SetOutsideValue(1)
   thresh_img = thresh_filter.Execute(spheroid)
   thresh_value = thresh_filter.GetThreshold()
   spheroid = spheroid > thresh_value
   for ws_level in [0.50,0.75,1, 1.25, 1.5, 1.75, 2, 2.5, 3,4,5]:

     print("Processing ws_level ",ws_level)


     segm=segmenter_evaluator.segment_nuclei(mask_img=spheroid, seeds_img = [], thold = 0.5, ws_method = 1, ws_level = ws_level)
     
     #segm.SetSpacing(spheroid.GetSpacing())
     #print(segm)
     #print("saving")
     #sitk.WriteImage(segm,'segmoutput_'+sph+'_'+str(ws_level*100)+'.nrrd')
     #print("saved")
     [aji, pq, ji, nndp] = segmenter_evaluator.evaluate(segm,GTfilename)
     
     print(sph,',',filename,',',ws_level,',',pq,',',aji,',',ji,',',nndp,file=f)  

