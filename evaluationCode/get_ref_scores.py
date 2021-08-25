'''
Tuomas Kaseva, 15.5.2021
'''

# ===============================================
#            Imports
# ===============================================
import sys
import SimpleITK as sitk

import seg

from stats_utils import get_fast_aji,get_fast_pq, remap_label

import argparse
import os
import numpy as np
import glob
sys.path.append("..")
import configSegmenter as c

#uncomment to compute jaccard index using matlab (required matlab function available http://www.3d-cell-annotator.org/uploads/3/4/9/3/34939463/jaccardindex3d_ji3d.zip)
#import matlab.engine

# ===============================================
#            Functions
# ===============================================

def get_scores(GT_img, pred_img):
    if type(GT_img) == sitk.SimpleITK.Image: #make this work both with sitk images and numpy arrays
       GT = sitk.GetArrayFromImage(GT_img)
    else:
       GT=GT_img
    if type(pred_img) == sitk.SimpleITK.Image:
       pred = sitk.GetArrayFromImage(pred_img)
    else:
       pred = pred_img

    if len(pred.shape) > 3:
        pred = pred[:, ..., 2]

    # Get evaluation scores
    r = remap_label(GT)
    s = remap_label(pred)
    #pq = get_fast_pq(r,s)[0][2]
    pqa,pqb=get_fast_pq(r,s)
    print("pqa",pqa)
    pq=pqa[2]
    aji = get_fast_aji(r,s)
    [seg_score, scores] = seg.get_SEG(R = GT, S = pred)
    [JI_score, scores] = seg.get_JI(R = GT, S = pred)
    print("JI score",JI_score)
    nndp = 2*np.abs(r.max()-s.max())/(r.max() + s.max())

    return aji, pq, seg_score, nndp

# ===============================================
#            Main
# ===============================================

'''
PURPOSE: Get results of the reference algorithms.
'''

if __name__ == '__main__':

    #uncomment to compute jaccard index using matlab (required matlab function available http://www.3d-cell-annotator.org/uploads/3/4/9/3/34939463/jaccardindex3d_ji3d.zip)
    #eng = matlab.engine.start_matlab()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir", type = str,help = "Directory of the reference algorithm segmentations and GTs", default ='/proj/group/HUS/esalli/3DNucleiSegmenter/data/3rdPartyData/3DCellAnnotator/Review_Binary3DMasks/')
    parser.add_argument("--dest_dir", type = str, help="Directory to save the data", default = "/proj/group/HUS/scripts/models/cells/results")
    args = parser.parse_args()

    results = {"Embryo": {}, "Neurosphere": {}}


    #os.chdir(args.ref_dir)
    dires = ["Neurosphere", "Embryo"]
    for idx, dire in enumerate(dires):
        #os.chdir(os.path.join(args.ref_dir, dire))

        # Get ground truth
        GT_file = glob.glob(args.ref_dir+dire+"/*Ground*")[0]
        GT_img = sitk.ReadImage(GT_file)
        # Get segmentations
        annot_files = os.listdir(args.ref_dir+dire)
        for annot_file in annot_files:

            cont = annot_file.split("_")
            identif = cont[-1].split(".")[0]

            if "Ground" not in annot_file:
                segm_img = sitk.ReadImage(args.ref_dir+dire+"/"+annot_file)
                segm_img_array=sitk.GetArrayFromImage(segm_img)
                if segm_img_array.ndim==4: # this is likely a vector (RGB) image, convert it to indexed image
                    C0=sitk.VectorIndexSelectionCast(segm_img,index=0,outputPixelType=sitk.sitkInt64)
                    C1=sitk.VectorIndexSelectionCast(segm_img,index=1,outputPixelType=sitk.sitkInt64)
                    C2=sitk.VectorIndexSelectionCast(segm_img,index=2,outputPixelType=sitk.sitkInt64)
                    npC0=remap_label(sitk.GetArrayFromImage(C0))
                    npC1=remap_label(sitk.GetArrayFromImage(C1))
                    npC2=remap_label(sitk.GetArrayFromImage(C2))
                    
                    maxvalueC0=np.max(npC0)
                    maxvalueC1=np.max(npC1)
                    maxvalueC2=np.max(npC2)
                
                    L=npC0+npC1*(maxvalueC0+1)+npC2*(maxvalueC0+1)*(maxvalueC1+1)
                    segm_img=remap_label(L)


                # Calculate scores
                print(identif, GT_file)
                aji, pq, seg_score, nndp = get_scores(GT_img, segm_img)

                # Update results
                results[dire][identif]=[aji, pq, seg_score, nndp]
                
                if type(GT_img) == sitk.SimpleITK.Image:
                    r=sitk.GetArrayFromImage(GT_img)
                else:
                    r=GT_img
                if type(segm_img) == sitk.SimpleITK.Image:
                    s=sitk.GetArrayFromImage(segm_img)
                else:
                    s=segm_img
                
                #uncomment to compute jaccard index using matlab (required matlab function available http://www.3d-cell-annotator.org/uploads/3/4/9/3/34939463/jaccardindex3d_ji3d.zip)
                #r = remap_label(r)
                #s = remap_label(s)
                #r=r.astype(int)
                #s=s.astype(int)
                #rm=matlab.double(r.tolist())
                #sm=matlab.double(s.tolist())
                #result=eng.metricKaggleDataScienceBowl2018(rm,sm,0.0, 0.0, 0.0,nargout=2)
                #so=result[1]
                #JIvalue = np.mean(so);
                #print("Jaccard ",JIvalue)
                
                

    #os.chdir(args.dest_dir)
    np.save(c.EVALUATION_SCORES_DATADIR+"ref_alg_results", np.array(results))

    