# ===============================================
#            Imports
# ===============================================

import os
import numpy as np
import sys
import glob
from joblib import Parallel, delayed
import argparse
import SimpleITK as sitk
import cv2
import nrrd
from skimage.measure import label

root = os.getcwd()
sys.path.insert(1, os.path.join(root, 'evaluationCode'))
from seg import get_JI
from stats_utils import get_fast_aji,get_fast_pq, remap_label



# ===============================================
#            Functions
# ===============================================

def general_watershed(mask_img, seeds_img = 0, thold = 0.5, ws_level = 1, preproc_seeds = False, remove_small = True):

    '''
    PURPOSE:
    Function performs either H-minima transform backed or marker-based watershed.
    ARGUMENTS: 
    mask_img = Nuclei mask, can be binary or normalized to [0,1] in SITK format.
    seeds_img = Multiclass segmentation of seeds or normalized to [0, 1] in SITK format.
    thold = Threshold for mask binarization.
    ws_level = h-value used in the morphological watershed.
    preproc_seeds = Whether to label seeds with connected components filter.
    remove_small = Whether to remove segmented objects with less than 5% volume of the average segmentation.
    OUTPUTS: 
    segm_img = Segmented nuclei in SITK format.
    '''
    
    # Threshold mask image
    thresh_img = mask_img>thold

    # Label connected components
    if preproc_seeds:
        thold_seeds =seeds_img>thold
        orig_spacing = np.array(seeds_img.GetSpacing())
        components = sitk.ConnectedComponentImageFilter()
        components.SetFullyConnected(True)
        seeds_img=components.Execute(thold_seeds)
        seeds_img.SetSpacing(np.abs(orig_spacing/orig_spacing[0])) # normalize


    # Normalize spacing
    spacing = np.array(thresh_img.GetSpacing())
    thresh_img.SetSpacing(np.abs(spacing/spacing[0]))

    # Get distance image
    dist_img = sitk.SignedMaurerDistanceMap(thresh_img != 0, insideIsPositive=False, squaredDistance=False, 
                                            useImageSpacing=True)

    # Morphological or marker based watershed
    if type(seeds_img) != sitk.SimpleITK.Image:
        ws = sitk.MorphologicalWatershed(dist_img, markWatershedLine=False,level=ws_level)
    else:
        ws = sitk.MorphologicalWatershedFromMarkers(dist_img, seeds_img, markWatershedLine=False)

    ws_segm = sitk.Mask(ws, sitk.Cast(thresh_img, ws.GetPixelID()))
    segm = sitk.GetArrayFromImage(ws_segm)
    
    # Remove small nuclei segmentations
    if remove_small:
        _, avg_size, indexes = get_stats(ws_segm)

        for nuclei in list(indexes.keys()):
            ind = indexes[nuclei]
            size = len(ind[0])
            if size < 0.05*avg_size:
                segm[ind]=0

    # Update segmentation
    segm_img = sitk.GetImageFromArray(segm)
    segm_img.CopyInformation(ws_segm)

    return segm_img


def get_stats(segm_img):

    '''
    PURPOSE:
    Gathers average roudness, size and postions of each label of the segmentation.
    
    ARGUMENTS: 
    segm_img = Segmented nuclei in SITK format.
    OUTPUTS: 
    avg_rdness = Average roudness over all segmented nuclei.
    avg_size = Average size of the segmented nuclei.
    indexes = Voxel locations of each segmented nuclei.
    '''

    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(segm_img)

    rounds = []
    sizes = []
    indexes = {}
    for i in shape_stats.GetLabels():
        rounds.append(shape_stats.GetRoundness(i))
        sizes.append(shape_stats.GetPhysicalSize(i))
        flat_ind = shape_stats.GetIndexes(i)
        ind = (np.array(flat_ind[2::3]), np.array(flat_ind[1::3]), np.array(flat_ind[0::3]))
        indexes[i]=ind
        
    avg_rdness = np.mean(rounds) 
    avg_size = np.mean(sizes)
    return avg_rdness, avg_size, indexes


def segment_nuclei(mask_img, seeds_img = 0, thold = 0.5, ws_method = 0, ws_level = 1):

    '''
    PURPOSE:
    Segments nuclei using one of the three options specified by the mode.
    ARGUMENTS: 
    mask_img = Nuclei mask, can be binary or normalized to [0,1] in SITK format.
    seeds_img = Initial segmentation of seeds, can be binary or normalized to [0,1] in SITK format.
    ws_method = 0: Marked-based watershed with NN-generated seeds, 1: H-minima transfrom backed watershed using mask only, 2: Marked-based watershed with 
    H-minima transfrom backed watershed postprocessed NN-generated seeds.
    OUTPUTS: 
    segm_img = Segmented nuclei.
    '''

    if ws_method == 0:
        segm_img = general_watershed(mask_img = mask_img, seeds_img = seeds_img, preproc_seeds = True)

    elif ws_method == 1:
        segm_img = general_watershed(mask_img = mask_img, ws_level = ws_level)

    else:
        new_seeds_img = general_watershed(mask_img = seeds_img, ws_level = ws_level, remove_small = False, thold = 0.3)
        segm_img = general_watershed(mask_img = mask_img, seeds_img = new_seeds_img)

    return segm_img

def CC_segment_nuclei(mask_img, thold = 0.5):

    '''
    PURPOSE: Pefform instance segmentation using binary nuclei masks ansd connected component analysis.
    ARGUMENTS:
    mask_img = Nuclei mask, can be binary or normalized to [0,1] in SITK format.
    '''
    segm_img = mask_img>thold
    segm = sitk.GetArrayFromImage(segm_img)
    segm = label(segm)
    segm_img = sitk.GetImageFromArray(segm)
    segm_img.CopyInformation(mask_img)

    return segm_img


def get_optimal_segm(segm_imgs, ws_levels, opt_mode = 1, GT_file = 0):

    '''
    PURPOSE:
    Find the segmentation with the best roudness score from a set of segmentations.
    ARGUMENTS: 
    segm_imgs = Set of nuclei segmentations in SITK format.
    ws_levels = Set of h-values.
    mode = 0: Marked-based watershed with NN-generated seeds, 1: Level-based watershed using, 2: Marked-based watershed with 
    level-based watershed postprocessed NN-generated seeds.
    opt_mode = 0: Optimize based on roundness, 1: Optimize based on evaluation scores.
    OUTPUTS: 
    opt_segm = Optimal (per roundness) segmentation of nuclei.
    opt_lev = Optimal (per roundness) h-value used when creating opt_segm.
    '''

    opt_score = 0
    opt_segm = 0
    opt_lev = 0
    for idx, segm_img in enumerate(segm_imgs):

        # Based on roundness
        if opt_mode == 0:
            score, _, _ = get_stats(segm_img)

        # Based on evaluation metrics
        else:
            aji, _, _, pq, seg, _ = evaluate(segm_img, GT_file, verbose = 0)
            score = (aji+pq+seg)/3

        # "Insane" optimization
        if opt_mode == 2:
            if (score < opt_score) or (idx == 0):
                opt_score = score
                opt_segm = segm_img
                opt_lev = ws_levels[idx]

        else:
            if score > opt_score:
                opt_score = score
                opt_segm = segm_img
                opt_lev = ws_levels[idx]

    return opt_segm, opt_lev

def evaluate(segm_img, GT_file, segm_dire = None, save_segms = False, indices = [0,0], verbose = 1):

    '''
    PURPOSE:
    Compute AJI and PQ between the predicted segmentation and the ground truth.
    ARGUMENTS: 
    segm_img = Nuclei segmentation in SITK format.
    GT_file = Nifti-file of the ground truth.
    segm_dire = Directory where to save the segmentation.
    save_segms = Whether to save the segmentation.
    indices = [MAWAER identifier, data index, model index], used when saving the segmentation.
    verbose = Whether to print out results.
    OUTPUTS: 
    aji = Aggregated Jaccard Index.
    pq = Panoptic Quality.
    nndp = Nuclei number difference percentage.
    ji = Jaccard Index.
    S = Nuclei segmentation resized to GT.
    '''
    
    # Load the prediction and GT
    GT = sitk.ReadImage(GT_file)
    GTdata=sitk.GetArrayFromImage(GT)
    segm = sitk.GetArrayFromImage(segm_img)

    # Resample
    sf = int(segm.shape[0]/GTdata.shape[0]) # Assuming GT with smaller depth
    S = []
    for k in np.arange(GTdata.shape[0]):
        
        if sf == 1: # Every slice
            incr = k
        else:
            incr = np.min([1+k*sf, segm.shape[0]]) # Every sf-th slice

        if segm.shape[1] != GTdata.shape[1]: # From 256x256 to GT size 
            resized = cv2.resize(segm[incr].astype(np.uint16), (GTdata.shape[1], GTdata.shape[2]), interpolation = cv2.INTER_NEAREST)
            S.append(resized)
        else:
            S.append(segm[incr])

    S = np.array(S)

    # Get evaluation scores
    r = remap_label(GTdata)
    s = remap_label(S)
    dq, sq, pq = get_fast_pq(r,s)[0]
    aji = get_fast_aji(r,s)
    ji, _ = get_JI(R = GTdata, S = S)
    nndp = 2*np.abs(r.max()-s.max())/(r.max() + s.max())

    if 0: # Not used in the end
        dqs = []
        for t in np.arange(0.5,1.01,0.1):
            [dq, sq, dqsq], [paired_true, paired_pred, unpaired_true, unpaired_pred] = get_fast_pq(r,s, match_iou=t)
            dqs.append(dq)

    if verbose:
        GT_cont = GT_file.split("/")[-1]
        print("Number of components, reference and predicted")
        print(r.max())
        print(s.max())
        print("AJI: ",aji)
        print("PQ: ",pq)

    # Save segmentation
    if save_segms:
        header = "_".join([indices[0],indices[1], indices[2],"segmentation"])
        header = header + ".nrrd"

        # Get header
        _, GT_h = nrrd.read(GT_file)
        nrrd.write(os.path.join(segm_dire, header), np.swapaxes(S,0,2).astype(np.uint16), header = GT_h)

    return aji, dq, sq, pq, ji, nndp

def segment_and_evaluate(mask_file, GT_file, seeds_file = 0, ws_method = 1, merge = True, opt_mode = 0, ws_levels = [1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5], 
segm_dire = None, save_segms = False,identif = None,verbose = True):

    '''
    PURPOSE:
    Segment and get evaluation scores.
    ARGUMENTS: 
    mask_file = Nrrd file of mask segmentation.
    seeds_file = Nrrd file of seeds segmentation.
    GT_file = Nifti-file of the ground truth.
    ws_method = 0: Marked-based watershed with NN-generated seeds, 1: Level-based watershed using, 2: Marked-based watershed with 
    level-based watershed postprocessed NN-generated seeds.
    merge = Whether to merge seeds with mask.
    opt_mode = 0: Optimize based on roudness, 1: Optimize based on the evaluation scores
    ws_levels = A set of h-values used in segmentation optimization.
    segm_dire = Directory where to save the segmentation.
    save_segms = Whether to save the segmentation.
    identif = Identifier describing the MAWAER configuration.
    verbose = Whether to print out results.
    OUTPUTS: 
    indices = indices of the dataset and the model
    aji = Aggregated Jaccard Index.
    pq = Panoptic Quality.
    ji = Jaccard Index.
    nndp = Nuclei number difference percentage.
    opt_lev = Optimal h-value
    '''

    mask_identif = mask_file.split("/")[-1]
    indices = mask_identif.split("_")[0:2]

    if verbose == 1:
        print("Segmenting spheroid: ", indices[0])

    # Read mask file
    mask_img = sitk.ReadImage(mask_file)

    # Read seeds file
    if seeds_file != 0:
        seeds_img = sitk.ReadImage(seeds_file)
        if merge:
            seeds_img = seeds_img*mask_img
    else:
        seeds_img = 0

    # Segment
    if ws_method == 0:
        segm_img = segment_nuclei(mask_img=mask_img, seeds_img = seeds_img, ws_method = ws_method)
        opt_lev = 0

    elif ws_method == 3:
        segm_img = CC_segment_nuclei(mask_img=mask_img)
        opt_lev = 0

    else:
        segm_imgs = []
        for ws_level in ws_levels:
            segm_img = segment_nuclei(mask_img=mask_img, seeds_img = seeds_img, ws_method = ws_method, ws_level = ws_level)
            segm_imgs.append(segm_img)

        segm_img, opt_lev = get_optimal_segm(segm_imgs, ws_levels, GT_file = GT_file, opt_mode = opt_mode)

    # Evaluate
    aji, dq, sq, pq, ji, nndp = evaluate(segm_img, GT_file, save_segms = save_segms, segm_dire = segm_dire, indices = [identif, indices[0], indices[1]])

    return indices, aji, pq, ji, nndp, opt_lev,



# ===============================================
#            Main
# ===============================================


'''
Computes and evaluates segmentations using different system configurations.
'''


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = int, help="0: 12 spheroids, 1: independent datasets", default = 1)
    parser.add_argument("--mask_type", type = str, help="M3D, M3DE, M2DE or M3DEW", default = "M2DE")
    parser.add_argument("--ws_method", type = int, default =0, help="0: Marked-based watershed with NN-generated seeds, 1: H-minima transform backed watershed using only mask, 2: Marked-based watershed with H-minima transform backed watershed postprocessed NN-generated seeds, 3: Connected component analysis of a given nuclei mask.")
    parser.add_argument("--opt_mode", type = int, default = 0, help="0: Optimize based on roundness, 1: Optimize based on evaluation scores, 2: Optimize based on worst scores")
    parser.add_argument("--verbose", type = int, help="Whether to print out run specifics", default = 1)
    parser.add_argument("--save_segms", type = int, help="Whether to save segmentations", default = 1)
    args = parser.parse_args()

    # Define paths
    root = os.getcwd()
    mask_dire = os.path.join(root, "data/maskedData", "U_" + args.mask_type)
    seeds_dire = os.path.join(root, "data/maskedData", "U_S")
    GT_dires = [os.path.join(root, "data/preprocessedData/GT"), os.path.join(root, "data/independentData/GT")]
    GT_dire = GT_dires[args.dataset]
    dest_dire = os.path.join(root, "data/evaluationScores")
    segm_dires = [os.path.join(root, "data/segmentedData/spheroids"),os.path.join(root, "data/segmentedData/datasets")]
    segm_dire = segm_dires[args.dataset]

    # Define identifier
    ws_methods = {0: "A", 1: "B", 2: "C", 3: "D"}
    identif = "|".join([ws_methods[args.ws_method], args.mask_type, str(args.opt_mode), str(args.dataset)])


    if args.verbose:
        print("Dataset index: ", args.dataset)
        print("Watershed method ", args.ws_method)
        print("Mask type: ", args.mask_type)
        print("Optimization mode: ", args.opt_mode)

    # Get mask files
    keywords = ["spheroid", "dataset"]
    keyword = keywords[args.dataset]
    os.chdir(mask_dire)
    mask_files = glob.glob("*" + str(keyword) + "*")

    # Organize to file triplets
    triplets = []
    for mask_file in mask_files:

        # Get indices
        header_cont = mask_file.split("_")
        data_idx = header_cont[0]
        model_idx = header_cont[1]

        # Specify files
        GT_kw = "_".join([data_idx, "GT."])
        seeds_kw = "_".join([data_idx,model_idx,keyword])
        GT_file = glob.glob(GT_dire + "/" + GT_kw + "*")[0]
        if args.ws_method == 1: # no seeds used
            seeds_file = 0
        else:
            seeds_file = glob.glob(seeds_dire + "/" + seeds_kw + "*")[0]
        mask_file = os.path.join(mask_dire, mask_file)

        # Fill
        triplets.append([mask_file, GT_file, seeds_file])


    ops = Parallel(n_jobs=len(triplets))(delayed(segment_and_evaluate)(mask_file = triplet[0], GT_file = triplet[1], seeds_file = triplet[2], 
    ws_method = args.ws_method, opt_mode = args.opt_mode, segm_dire = segm_dire, save_segms = args.save_segms, identif = identif)
                    for triplet in triplets)

    # Save output as a nested list
    segm_res = {}
    for op in ops:

        if op[0][0] not in list(segm_res.keys()):
            segm_res[op[0][0]] = {}

        segm_res[op[0][0]][op[0][1]]=op[1:]

    os.chdir(dest_dire)
    np.save(identif, np.array(segm_res))

