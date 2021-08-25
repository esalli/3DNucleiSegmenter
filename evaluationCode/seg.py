import warnings
import numpy as np
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed

import cv2
import matplotlib.pyplot as plt

"JI by Tuomas Kaseva, 11.6.2021"

def cell_JI(label, lab_count, S, R):
    
    # Get predicted segmentation labels which are inside the reference label area
    SR = S[R==label]
    inside_labels, in_lab_counts = np.unique(SR, return_counts = True)
    
    # Reference label True-False mapping (for union calculation)
    R_label_inds=np.where(R==label)
    fR = np.zeros(R.shape)
    fR[R_label_inds] = 1
    
    # Find the matching inside label
    match_label = 0
    best_JI = 0
    for index, count in enumerate(in_lab_counts):
        
        # exclude background
        if inside_labels[index]==1:
            continue
            
        # Compute JI
        S_label = inside_labels[index]
        cross_sect = count
        S_label_inds = np.where(S == S_label)
        fS = np.zeros(S.shape)
        fS[S_label_inds] = 1
        logic_or = np.logical_or(fS, fR)
        union = len(logic_or[logic_or == 1])
        JI = cross_sect/union
        
        if JI > best_JI:
            best_JI = JI
            match_label = S_label
    return [best_JI, match_label]
    

def get_JI(R, S, num_jobs = 4, verbose = False, compute_parallel = False):

    '''
    SEG-score based on:
    https://public.celltrackingchallenge.net/documents/SEG.pdf

    Arguments:

    R = reference segmentation labels
    S = predicted segmentation labels
    num_jobs = Number of jobs for parallel processing
    verbose = Whether to print out run specifics.
    compute_parallel = Whether to use parallel processing.
    '''

    if verbose == True:
       print("Caclulating SEG-score...")
    
    
    # Initializations
    S = S+1
    R = R+1
    labels, lab_counts = np.unique(R, return_counts= True) 

    # Exclude background
    labels = labels[1:]
    lab_counts = lab_counts[1:]
    
    # Calculate SEG     
    if compute_parallel:
        scores = Parallel(n_jobs=num_jobs)(delayed(cell_SEG)(lab, lab_counts[index], S, R)
                    for index, lab in enumerate(labels))
    else:
        scores = []
        for index, lab in enumerate(labels):
            scores.append(cell_JI(lab, lab_counts[index], S, R))

    scores = np.array(scores)
    JI = np.mean(scores[:, 0])
    
    return JI, scores


"SEG by Tuomas Kaseva, 10.5.2021"

def cell_SEG(label, lab_count, S, R):
    

    '''
    if verbose == True:
       print("Label: ", label)
    '''
    
    # Get predicted segmentation labels which are inside the reference label area
    SR = S[R==label]
    inside_labels, in_lab_counts = np.unique(SR, return_counts = True)
  
    # Reference label True-False mapping (for union calculation)
    R_label_inds=np.where(R==label)
    fR = np.zeros(R.shape)
    fR[R_label_inds] = 1
    
    # Find the matching inside label
    match_label = 0
    cross_sect = 0
    for index, count in enumerate(in_lab_counts):
        
        # exclude background
        if inside_labels[index]==1:
            continue
            
        if count/lab_count> 0.5: # Matching condition
            match_label = inside_labels[index]
            cross_sect = count
            
    # Calculate Jaccard index
    if match_label == 0:
       J = 0
    else:
        S_label_inds = np.where(S == match_label)
        fS = np.zeros(S.shape)
        fS[S_label_inds] = 1
        logic_or = np.logical_or(fS, fR)
        union = len(logic_or[logic_or == 1])
        J = cross_sect/union

    return [J, match_label]
    

def get_SEG(R, S, num_jobs = 4, verbose = False, compute_parallel = False):

    '''
    SEG-score based on:
    https://public.celltrackingchallenge.net/documents/SEG.pdf

    Arguments:

    R = reference segmentation labels
    S = predicted segmentation labels
    num_jobs = Number of jobs for parallel processing
    verbose = Whether to print out run specifics.
    compute_parallel = Whether to use parallel processing.
    '''

    if verbose == True:
       print("Caclulating SEG-score...")
    
    
    # Initializations
    S = S+1
    R = R+1
    labels, lab_counts = np.unique(R, return_counts= True) 

    # Exclude background
    labels = labels[1:]
    lab_counts = lab_counts[1:]
    
    # Calculate SEG     
    if compute_parallel:
        scores = Parallel(n_jobs=num_jobs)(delayed(cell_SEG)(lab, lab_counts[index], S, R)
                    for index, lab in enumerate(labels))
    else:
        scores = []
        for index, lab in enumerate(labels):
            scores.append(cell_SEG(lab, lab_counts[index], S, R))

    scores = np.array(scores)
    seg = np.mean(scores[:, 0])
    
    return seg, scores


