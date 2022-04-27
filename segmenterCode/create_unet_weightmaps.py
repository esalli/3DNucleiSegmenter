# ===============================================
#            Headers
# ===============================================

import os
import numpy as np
import nibabel as nib
import nrrd
from scipy.ndimage.morphology import distance_transform_edt
import argparse
from skimage.measure import label
from skimage.segmentation import find_boundaries
from joblib import Parallel, delayed



# ===============================================
#            Functions
# ===============================================


def unet_weight_map(y, wc=None, w0 = 10, sigma = 5):

    """
    Generate weight maps as specified in the U-Net paper
    for boolean mask.

    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf

    Parameters
    ----------
   y: Numpy array
        3D labelmap array.
    wc: dict
        Dictionary of weight classes.
    w0: int
        Border weight parameter.
    sigma: int
        Border width parameter.

    Returns
    -------
    Numpy array
        Training weights. A dD array of shape (image_height, image_width, image_slices).
    """

    # Convert y to include boundaries
    y=y.astype('int16')
    labels = y
    bound = find_boundaries(labels,connectivity=3,mode='outer').astype(np.uint8)
    ind = np.where(bound==1)
    labels[ind]=0
    labels[labels > 0]= 1
    labels = label(labels)

    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))
    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], y.shape[2], len(label_ids)))
        for i, label_id in enumerate(label_ids[1:]):
            print(label_id)
            distances[:,:,:,i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=3)
        d1 = distances[:,:,:,0]
        d2 = distances[:,:,:,1]
        w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels
    else:
        w = np.zeros_like(y)

    if wc:
        y[y>0]=1
        class_weights = np.zeros_like(y)
        for k, v in wc.items():
            class_weights[y == k] = v
        w = w + class_weights
    return w

def data_gen(img_id, dest_dir):
    print("GT index: ", img_id)
    l,h = nrrd.read(str(img_id) + "_GT_expanded_3_DT.nrrd")
    wmap = unet_weight_map(l, wc={0:1, 1:3}, w0 = 10, sigma =5)
    nrrd.write(os.path.join(dest_dir, str(img_id) + "_wmap.nrrd"), wmap, header = h)
    return 1



# ===============================================
#            Data generation
# ===============================================

if __name__ == '__main__':

    root = os.getcwd()
    data_dir = os.path.join(root, "data", "preprocessedData", "GT")
    dest_dir = os.path.join(root, "data", "preprocessedData", "wmaps")
    
    # Create maps
    os.chdir(data_dir)
    process = Parallel(n_jobs=12)(delayed(data_gen)(img_id, dest_dir)
                    for img_id in np.arange(1, 13))


