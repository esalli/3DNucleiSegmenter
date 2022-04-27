# ===============================================
#            Imports
# ===============================================

import numpy as np
import os
import glob
from tensorflow.keras.models import load_model
import argparse
import nrrd
import SimpleITK as sitk
import time


# ===============================================
#            Functions
# ===============================================

def normalize(sample):

    norm = (sample-np.min(sample))/(np.max(sample)-np.min(sample))
    return norm

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


def get_prediction(data, model, norm = True):


    # Check model input shape
    inp_shape = model.input.shape.as_list()
    interval = inp_shape[3]
    step = np.copy(interval)
    orig_len = len(data)
    preds = np.zeros((orig_len, data.shape[1], data.shape[2]))

    # 2D vs 3D
    if interval == 1:
        mode = "2D"
    else:
        mode = "3D"

    # Get windows
    winds = get_windows(data.shape[0], interval, step)

    for wind in winds:
        if len(wind)==1:
            wind = wind[0]

        if mode == "2D":
            sample = data[wind, :, :]
        else:
            sample = data[wind, :, :]
            sample = np.swapaxes(sample, 0, 2)

        # Normalize
        if norm:
            sample = normalize(sample)

            
        # Predict
        sample = sample[np.newaxis, ...]
        pred = model.predict(sample)

        if mode == "2D":
            preds[wind, :, :]=pred[0, ..., 0]
        else:
            preds[wind, :, :]=np.swapaxes(pred[0, ..., 0],0,2)

    return preds

def NN_masker(input_file, model, dest_dir, norm = False, verbose = True, pred_header = "_NN_pred"):

    # Get data
    X, h = nrrd.read(input_file) 
    X = np.swapaxes(X, 0, 2) 

    if verbose:
        print("Predicting...")
    preds = get_prediction(X, model, norm = norm)

    if verbose:
        print("Saving...")

    os.chdir(dest_dir)
    sample_name = input_file.split("/")[-1]
    s_id = sample_name.split("_")[0]

    nrrd.write("_".join([s_id, pred_header]) + ".nrrd", np.swapaxes(preds, 0, 2).astype(np.float32), header = h)

    return 1

    

# ===============================================
#            Main
# ===============================================

'''
Create either binary masks or seeds for the given data. Use either a different model for each data sample or process all samples with the same model.
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = int, help="0: 12 spheroids, 1: independent datasets, 2: test datasets", default = 1)
    parser.add_argument("--model_type", type = int, help="0: U_M3D, 1: U_M3DE, 2: U_M2DE, 3: U_S, 4: U_M3DEW", default = 0)
    parser.add_argument("--verbose", type = int, help="Whether to print out run specifics", default = 1)
    args = parser.parse_args()

    # Get root
    root = os.getcwd()

    # Get models
    model_types = ["U_M3D", "U_M3DE", "U_M2DE", "U_S", "U_M3DEW"]
    model_type = model_types[args.model_type]
    model_dire = os.path.join(root, "prebuiltModels", model_type)
    model_files = []
    os.chdir(model_dire)
    m_files = glob.glob("*.h5")
    for file in m_files:
        model_files.append(file)

    # Get data files
    data_dires = [os.path.join(root,"data/preprocessedData/spheroids"), os.path.join(root, "data/independentData/datasets"), 
    os.path.join(root, "data/testData/datasets")]
    data_dire = data_dires[args.dataset]
    data_files = []
    os.chdir(data_dire)
    if args.dataset == 0:
        d_files = glob.glob("*expanded*")
    else:
        d_files = glob.glob("*.nrrd")
    for file in d_files:
        data_files.append(file)

    # Define destination
    if args.dataset == 2:
        kw = "data/testMaskedData"
    else:
        kw = "data/maskedData"
    dest_dire = os.path.join(root, kw, model_type)
    print(dest_dire)

    
    # Get predictions
    for model_file in model_files:

        # Load model
        model_path = os.path.join(model_dire, model_file)
        model = load_model(model_path, compile = False)

        for data_file in data_files:

            data_idx = data_file.split("_")[0] # Assumed that the data file is formatted as idx_*
            model_bottom = model_file.split("_")[-1]
            model_idx = model_bottom.split(".")[0]

            # Only choose the data file with similar index
            if args.dataset == 0:
                if int(model_idx) != int(data_idx):
                    continue

            sample_file = os.path.join(data_dire, data_file)
            if args.verbose:
                print("Model and sample: ", [model_file, data_file])

            # Format header
            if args.dataset == 0:
                pred_header = "_".join([model_idx, "spheroid_mask"])
            else:
                pred_header = "_".join([model_idx, "dataset_mask"])


            # Predict
            start = time.time()
            op = NN_masker(sample_file, model, dest_dire, norm = True, pred_header = pred_header)
            stop = time.time()
            print(stop-start)







