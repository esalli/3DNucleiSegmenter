'''
Tuomas Kaseva, 28.4.2021
'''

# ===============================================
#            Imports
# ===============================================

import os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,CSVLogger
import argparse
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
import random as python_random
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


# ===============================================
#            Import utils
# ===============================================

from models_losses import *
from training_utils import get_dimensions, DataGenerator, config_saver, val_test_split

# ===============================================
#            Training
# ===============================================

'''
Training of U-Nets for mask generation. More functionality than what was discussed in the paper. 
'''

if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",  type = int, default =200)
    parser.add_argument("--lrs",  type = str, help = "Initial learning rate, its decayers and the epochs were decays are performed.", default = "1e-3,5,10,75,110")
    parser.add_argument("--att_weight",  type = float, default =0, help ="Weight emphasis for the attentive Dice score")
    parser.add_argument("--batch_size",  type = int, default =16)
    parser.add_argument("--model_type",  type = str, default = "unet")
    parser.add_argument("--model_header",  type = str, default = "U", help="Header to help memorization of the model")
    parser.add_argument("--mask_type", type = int, help="0: 3D seeds, 1: 3D masks, 2: 2D edge masks, 3: 3D edge masks", 
    default = 3)
    parser.add_argument("--loss",  type = str, default = "bin_cross")
    parser.add_argument("--optimizer",  type = str, default = "adam")
    parser.add_argument("--val_test_split", type = str, help="Indices of the validation set and the test set", default = "0,1")
    parser.add_argument("--augment_params", type = str, help="1: whether to do augmentation, 2: whether to normalize, 3: whether to use artificial data (old): ", 
    default ="1,1,0")
    parser.add_argument("--unet_params", type = str, help="1: Number of filters in NN-layers, 2: Number of layers in the encoder, 3: Number of convolutional blocks in the encoder and decoder layers", 
    default = "16,5, 3")
    parser.add_argument("--pretrained", help="Load pretrained model given as a string pointing to the model filepath", 
    default = None) 
    parser.add_argument("--seeds", type = str, help="Check the reproducibility section of https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development",
    default = "10,10") 
    parser.add_argument("--verbose", type = int, help="Whether to print out training specifics", 
    default = 1) 
    args = parser.parse_args()

    # Assign seeds
    seeds = [int(k) for k in args.seeds.split(",")] 
    np.random.seed(seeds[0])
    python_random.seed(seeds[0])
    tf.random.set_seed(seeds[1])
    
    # Parse paths
    root = os.getcwd()
    mask_types = ["S", "M3D", "M2DE", "M3DE"]
    mask_type = mask_types[args.mask_type]
    data_dir = os.path.join(root,"data/trainingData/"+mask_type)
    model_dir = os.path.join(root, "models", "U_" + mask_type)
    split_file = os.path.join(root, "segmenterCode/cell_split.npy")

    # Parse unet parameters
    params = args.unet_params.split(",")
    unet_params = []
    for param in params:
        unet_params.append(int(param))

    # Define dataset dimensions and number of classes
    dim_lab, dim_dat, num_class = get_dimensions(data_dir)

    # Model and loss alternatives, imported from models_losses.py
    models = {"unet": unet(input_shape = dim_dat, num_class = num_class, num_filt = unet_params[0], num_layers = unet_params[1], 
        num_conv = unet_params[2]),
        "unet_dropout": unet(input_shape = dim_dat, num_class = num_class, num_filt = unet_params[0], num_layers = unet_params[1], 
        num_conv = unet_params[2], use_dropout = True),
        "unet_3d": unet(input_shape = dim_dat, num_class = num_class, num_filt = unet_params[0], num_layers = unet_params[1], 
        num_conv = unet_params[2], three_dim = True)}

    loss_functions = {"Dice": Dice(att_weight = args.att_weight, squared = False), "Dice_squared": Dice(att_weight = args.att_weight, squared = True), "tversky_loss": Tversky(att_weight = args.att_weight), "tversky_loss_orig": tversky_loss, 
    "tversky_loss_squared": Tversky(att_weight = args.att_weight, squared = True), "bin_cross": bin_crossentropy, "comb_loss": crossent_dice_comb}

    # Model name
    model_name = "_".join([args.model_header, mask_type, str(int(args.val_test_split[-1])+1)])
    args.model_header = model_name

    # Callbacks
    mcp_save = ModelCheckpoint(os.path.join(model_dir,model_name + ".h5"), save_best_only=True, monitor='val_loss', mode='min')
    filename = os.path.join(model_dir, model_name + "_history.log")
    open(filename, "a").close()
    logger = CSVLogger(filename = filename)

    # Parse augmentation parameters
    params = args.augment_params.split(",")
    aug_params = []
    for param in params:
       aug_params.append(int(param))

    # Get files
    train_files, val_files, test_files = val_test_split(data_dir = data_dir, val_test_split=args.val_test_split, split_file = split_file)
    if args.batch_size > len(val_files):
        args.batch_size = len(val_files)

    # Save config
    name, config = config_saver(args)
    os.chdir(model_dir)
    np.save(name, config)

    # Initialize generators
    train_gen = DataGenerator(train_files, args.batch_size, augment = aug_params[0], dim = dim_dat, dim_label = dim_lab, normalize = aug_params[1])
    val_gen = DataGenerator(val_files, args.batch_size, dim = dim_dat, dim_label = dim_lab, normalize = aug_params[1])

    # Set LR decay
    num_steps = int(len(train_files)/args.batch_size)
    lr_stat = [float(k) for k in args.lrs.split(",")] 
    lr_decay = PiecewiseConstantDecay([int(lr_stat[3])*num_steps, int(lr_stat[4])*num_steps], [lr_stat[0], lr_stat[0]/lr_stat[1], lr_stat[0]/lr_stat[2]])

    # Optimizers
    optims = {"sgd": tf.keras.optimizers.SGD(learning_rate =lr_decay, momentum = 0.9),
        "adam": tf.keras.optimizers.Adam(learning_rate = lr_decay)}

    # Define model and loss
    model = models[args.model_type]
    loss_func = loss_functions[args.loss]

    # Load pretrained model
    if args.pretrained != None:
        model.load_weights(args.pretrained)

    # -----------------------------------------------
    #           Fit model
    # -----------------------------------------------

    model.compile(loss=loss_func,
                optimizer=optims[args.optimizer]) 

    if args.verbose:
        print("Validation-test split: ", args.val_test_split)
        print("Seeds: ", args.seeds)
        print("Model header: ", args.model_header)
        print("Label and data dimensions: ", dim_lab, dim_dat)
        print("Number of files in training, validation and test: ", len(train_files), len(val_files), len(test_files))
        print(model.summary())


    history = model.fit(x = train_gen, epochs=args.epochs, 
                steps_per_epoch = int(len(train_files)/args.batch_size),
                validation_data=val_gen, 
                validation_steps= int(len(val_files)/args.batch_size),
                verbose=1,callbacks = [mcp_save, logger], use_multiprocessing = False, workers = 12, max_queue_size = 12) 

    history = np.array(history.history)
    np.save(os.path.join(model_dir, model_name) + "_history", history)


 


  
