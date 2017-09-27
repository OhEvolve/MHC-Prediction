

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

def auroc(guesses,actual,params={}):
    #parameters to include - binary,silent
    """
    A clean, fast callable function to return an auROC score for two lists comparing
    guesses to actual values
    """
   
    # deal with custom parameters
    default_params = default_auroc_params()
    params = append_dicts(params,default_params)

    # change analog input to binary if not previously defined this way
    if not params['binary']: 
        median = 0.5 # np.median(actual)
        actual = [1 if c > median else 0 for c in actual] 
    pos_inc,neg_inc = 1./sum(actual),1./(len(actual)-sum(actual)) 

    # order the inputs by guesses
    pairs = sorted([(g,a) for g,a in zip(guesses,actual)],key=lambda x: -x[0])
    coor,score = [(0.,0.)],0.

    # iterate across each point to trace the curve
    for p in pairs:
        if p[1] == 0: score += neg_inc*coor[-1][1]
        coor.append((coor[-1][0] + neg_inc*(1-p[1]),coor[-1][1] + pos_inc*p[1]))

    # return finalized auROC score (with coordinates if requested)
    if params['return_coordinates']:
        return score,coor 
    else:
        return score


#####################
## DEFAULT METHODS ##
#####################

def default_auroc_params():
    return {
            'binary':False,
            'silent':False,
            'return_coordinates':False
           }

#####################
## FACTORY METHODS ##
#####################


def append_dicts(old_dict,*dict_args):
    """
    Given an original dictionary and any number of additional dicts, adds
    new entries to original if they are not already present
    """
    for dictionary in dict_args:
        for key in dictionary.keys():
            if not key in old_dict.keys():
                print 'Adding parameter: {} -> {}'.format(key,dictionary[key])
                old_dict[key] = dictionary[key]
    return old_dict
