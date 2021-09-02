# ===============================================
#            Imports
# ===============================================

import os
import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom, rotate, shift
import tensorflow.keras as keras
import SimpleITK as sitk
import time
import glob
tf.compat.v1.disable_eager_execution() # For faster data loading


# ===============================================
#            Functions
# ===============================================

def val_test_split(data_dir, val_test_split, split_file = "/wrk/group/HUS/tkaseva/motility/motility_meta/small_bowel_segmentation_split.npy"):

    val_files = []
    train_files = []
    test_files = []
    set_split = np.load(split_file, allow_pickle = True)
    split_ind = val_test_split
    split_ind = split_ind.split(",")
    val_set = set_split[int(split_ind[0])]

    # Whether no test set
    if len(split_ind) > 1:
        test_set = set_split[int(split_ind[1])]
    else:
        test_set = []

    # Organize files
    os.chdir(data_dir)
    files = glob.glob("*.npy")
    for file in files:
        pat = file.split("_")[0]
        file_loc = os.path.join(data_dir, file)
        if pat in val_set:
            val_files.append(file_loc)
        elif pat in test_set:
            test_files.append(file_loc)
        else:
            train_files.append(file_loc)

    return train_files, val_files, test_files

def rescale(data, scale = 0.9, interpolation = sitk.sitkNearestNeighbor):
    
    if scale < 1:
        container = np.zeros(data.shape)

    orig_size = data.shape[1]
    new_size = int(scale*orig_size)
    diff = int(np.abs(orig_size - new_size)/2)

    image = sitk.GetImageFromArray(np.copy(data.astype(np.float32)), isVector=False)
    sitk_resampled = sitk.Resample(image, [data.shape[2], new_size, new_size], sitk.Transform(), interpolation,
                             image.GetOrigin(), [1,1/scale,1/scale])
    r = sitk.GetArrayFromImage(sitk_resampled)

    if scale > 1:
        r = r[diff:diff+orig_size, diff:diff +orig_size, :]

    else:
        container[diff:diff+new_size, diff:diff+new_size, :]=r
        r = container
        
    return r


def get_dimensions(data_folder):

    # Get dimensions
    os.chdir(data_folder)
    files = os.listdir()
    lab, arr, _ = np.load(files[0], allow_pickle = True)
    
    # Extract number of classes
    if lab.shape[0:len(lab.shape)]==arr.shape[0:len(lab.shape)]:
        num_class = 1
    else:
        num_class = lab.shape[-1]

    if arr.shape[-1]>4:
        arr_shape = arr.shape + (1,)
    else:
        arr_shape = arr.shape

    return lab.shape + (2,), arr_shape, num_class # 2 channels in labels to include weights 


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=7, dim=(128, 128, 32, 1), dim_label = (128, 128, 32, 2), shuffle=True, augment = True, normalize = True):
        'Initialization'
        self.dim = dim
        self.dim_label = dim_label
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.augment = augment
        self.normalize = normalize
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim), dtype = np.float32)
        y = np.empty((self.batch_size, *self.dim_label), dtype=np.uint8)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            lab, data, weights = np.load(ID, allow_pickle = True)

            # 2D vs 3D
            if data.shape[-1] < 4:
                X[i,] = data
            else:
                X[i,] = data[:, ..., np.newaxis]

            y[i, ..., 0] = lab
            y[i, ..., 1] = weights

        # Subsets for augment
        rot_end = np.ceil(self.batch_size/1.5).astype(np.uint8)
        rot_inds = np.arange(rot_end)
        mirror_inds = np.random.choice(np.arange(self.batch_size), int(self.batch_size/2), replace = False)
        rescale_ind = np.random.choice(np.arange(self.batch_size), 1)[0]


        if self.augment:

            # Mirror
            X[mirror_inds]=np.swapaxes(X[mirror_inds], 1, 2)
            y[mirror_inds, ..., 0]=np.swapaxes(y[mirror_inds, ..., 0], 1, 2)
            y[mirror_inds, ..., 1]=np.swapaxes(y[mirror_inds, ..., 1], 1, 2)

            # Rotate
            angle = np.random.choice(np.arange(-360, 360, 10), 1)[0]
            rot_data = rotate(X[rot_inds], angle = angle, axes = (1,2), order = 3, reshape = False)
            rot_y = rotate(y[rot_inds, ..., 0], angle = angle, axes = (1,2), order = 0, reshape = False)
            rot_w = rotate(y[rot_inds, ..., 1], angle=angle, axes = (1,2), order = 0, reshape = False)

            X[rot_inds]=rot_data
            y[rot_inds, ..., 0]= rot_y
            y[rot_inds, ..., 1]= rot_w 

            # Rescale if 3D - not used in the study
            if 0:
                scale = np.random.choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4], 1)[0]
                X[rescale_ind]=rescale(np.copy(X[rescale_ind, ..., 0]), scale = scale, interpolation = sitk.sitkLinear)[:, ..., np.newaxis]
                y[rescale_ind, ..., 0]=rescale(np.copy(y[rescale_ind, ..., 0]), scale = scale)
                y[rescale_ind, ..., 1]=rescale(np.copy(y[rescale_ind, ..., 1]), scale = scale)

        if self.normalize:
            for i, ID in enumerate(list_IDs_temp):

                # Normalization to [0,1] range
                X[i,]= (X[i,]-np.min(X[i,]))/(np.max(X[i,])-np.min(X[i,]))

                # Zero mean, unit variance
                #X[i,]= (X[i,]-np.mean(X[i,]))/(np.std(X[i,]))

        return X, y


def config_saver(args):

    name = args.model_header
    config = {}
    config["epochs"]=args.epochs
    config["att_weight"]=args.att_weight
    config["batch_size"]=args.batch_size
    #config["path_dir"]=args.path_dir
    config["loss"]=args.loss
    config["optimizer"]=args.optimizer
    config['model_params']=args.unet_params
    config["model_type"]=args.model_type
    config["augment_params"]=args.augment_params
    config["val_test_split"]=args.val_test_split
    config["lr_params"]=args.lrs
    config["seeds"]=args.seeds
    config["time"]=time.asctime()

    return ["_".join([name, "config"]), np.array(config)]


# ===============================================
#            Main
# ===============================================

'''
Utilities for the training procedure. Slightly more functionality than what was discussed in the paper.
'''

if __name__ == '__main__':
    print("Nothing to run here.")






