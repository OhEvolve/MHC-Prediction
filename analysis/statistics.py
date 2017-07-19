

'''
Project: Neural Network for MHC Peptide Prediction
Class(s): Statisitics
Function: Performs a variety of useful statistical tests, such as auROC measures

Author: Patrick V. Holec
Date Created: 5/19/2017
Date Updated: 5/19/2017
'''

# standard libraries


# nonstandard libraries
import numpy as np
import matplotlib.pyplot as plt

def auroc(guesses,actual,binary=False,silent=False):
    """
    A clean, fast callable function to return an auROC score for two lists comparing
    guesses to actual values
    """
    # change analog input to binary if not previously defined this way
    if not binary: actual = [1 if c > np.median(actual) else 0 for c in actual] 
    pos_inc,neg_inc = 1./sum(actual),1./(len(actual)-sum(actual)) 

    # order the inputs by guesses
    pairs = sorted([(g,a) for g,a in zip(guesses,actual)],key=lambda x: x[0])
    coor,score = [(0.,0.)],0.

    # iterate across each point to trace the curve
    for p in pairs:
        if p[1] == 1: score += pos_inc*coor[-1][1]
        coor.append((coor[-1][0] + pos_inc*p[1],coor[-1][1] + neg_inc*(1-p[1])))
    
    # return finalized auROC score
    return score
