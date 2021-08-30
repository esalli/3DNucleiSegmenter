# ===============================================
#            Imports
# ===============================================

import os
import numpy as np
#import nibabel as nib
import glob
from joblib import Parallel, delayed
from skimage.segmentation import find_boundaries
import argparse
from scipy.ndimage.morphology import binary_dilation
import nrrd
import SimpleITK as sitk


# ===============================================
#            Functions
# ===============================================

def label_converter(labels, mode = 1):

    # 3D seeds
    if mode == 0:
        pass

    # 3D masks
    elif mode == 1:
       labels[labels > 0]= 1

    # 3D masks with boundaries
    elif mode == 3:
        bound = find_boundaries(labels.astype(np.uint16),connectivity=3,mode='outer').astype(np.uint8)
        ind = np.where(bound==1)
        labels[ind]=0
        labels[labels > 0]= 1

    # 2D mask with boundaries
    else:
        for k in np.arange(labels.shape[0]):
            part_lab = np.copy(labels[k, :, :])
            bound = find_boundaries(part_lab.astype(np.uint16),connectivity=2,mode='outer').astype(np.uint8)
            ind = np.where(bound==1)
            part_lab[ind]=0
            part_lab[part_lab > 0]= 1
            labels[k, :, :]=part_lab
        
    return labels

def get_windows(max_val, interval, step):
    
    winds = []
    
    for k in np.arange(int(max_val/step)):
        
        wind = np.arange(k*step, k*step + interval)
        
        if np.max(wind) < max_val:
            winds.append(wind)
            
    # To ensure full image is windowed
    if winds[-1][-1] != max_val -1:
        winds.append(np.arange(max_val-interval, max_val))
            
    return np.array(winds)


def data_gen(img_idx, mode = 1,
    root = "", 
    dest = "",
    verbose = True, zdim = 24):


    # Load files
    spheroid, h = nrrd.read(os.path.join(root, "spheroids", str(img_idx) + "_smoothed_spheroid_expanded_3.nrrd"))
    X = np.swapaxes(spheroid, 0, 2)

    segm, hseg = nrrd.read(os.path.join(root, "GT", str(img_idx) + "_GT_expanded_3_DT.nrrd"))
    segm = np.swapaxes(segm, 0, 2)

    seeds, hs = nrrd.read(os.path.join(root, "seeds", str(img_idx) + "_hugeMIiso_3_DT.nrrd"))
    seeds = np.swapaxes(seeds, 0, 2)

    # Choose y
    if mode in [1,2,3]:
        y = segm
    else:
        y = seeds

    # Every 3rd slice if 2D
    if mode == 2:
        X = X[1::3]
        y = y[1::3]


    # Modify labels
    y = label_converter(y, mode = mode)
            
    # -----------------------------------------------
    #            Create data
    # -----------------------------------------------

    # Get windows
    if mode == 2:
        winds = get_windows(len(X), interval = 1, step = 1)
        winds = winds[:, 0]
    else:
        winds = get_windows(len(X), interval = zdim, step = int(zdim/2))

    if verbose:
       print("Index of the image: ", img_idx)


    for wind in winds:


        part_lab = y[wind, ...]
        part_imag = X[wind, ...]

        # Container initialization
        new_data = np.empty(3, dtype = object)

        if len(np.unique(part_lab)) > 1:

            # Weight map - not used in the study
            bound = find_boundaries(part_lab, mode = "thick").astype(np.uint8)
            w_map = binary_dilation(bound, iterations = 3) 
            #w_map = w_map + part_lab
            #w_map[w_map > 1]=1

            # Fill
            if not mode == 2:
                new_data[0]=np.swapaxes(part_lab.astype(np.uint8), 0, 2)
                new_data[1]=np.swapaxes(part_imag.astype(np.float32), 0, 2)
                new_data[2]=np.swapaxes(w_map.astype(np.uint8), 0, 2)
            else:
                new_data[0]=part_lab.astype(np.uint8)
                new_data[1]=part_imag.astype(np.float32)
                new_data[2]=w_map.astype(np.uint8)


            # Save
            image_filename = "_".join([str(img_idx), str(np.min(wind))+ "," + str(np.max(wind)), "data"])
            np.save(dest + "/" + image_filename, new_data)

    return 1

# ===============================================
#            Data generation
# ===============================================

'''
Create the ground truth masks for training of U-Nets.
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_type", type = int, help="0: 3D seeds, 1: 3D masks, 2: 2D edge masks, 3: 3D edge masks", 
    default = 3)
    parser.add_argument("--zdim", type = int, help="z-axis height of the patch", default = 24)
    args = parser.parse_args()

    # Get root
    root = os.getcwd()

    # Define paths
    root_dir = os.path.join(root,"data/preprocessedData/")
    mask_types = ["S", "M3D", "M2DE", "M3DE"]
    mask_type = mask_types[args.mask_type]
    dest_dir = os.path.join(root,"data/trainingData/"+mask_type)


    # Parallelization 
    process = Parallel(n_jobs=12)(delayed(data_gen)(img_id, root = root_dir, dest = dest_dir, mode = args.mask_type, zdim = args.zdim)
                    for img_id in np.arange(1, 13))
