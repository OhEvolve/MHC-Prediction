
'''

Project: Neural Network for MHC Peptide Prediction
Class(s): Standard 
Function: Takes an input dictionary of parameters defining the model, outputs a results file
          describing the predictive ability of that model

Author: Patrick V. Holec
Date Created: 2/3/2017
Date Updated: 3/20/2017

This is intended to function as a standardization for the generation of NN models.

'''

# standard libraries
import time
import pickle
import os

# nonstandard libraries
import numpy as np

# personal libraries
from models import * # libraries: kNN,cNN,spiNN
from analysis import * # libraries: visualize,statistics
from landscape import generate_test_set as gts,parameterize_data as pd,compile_data as cd


"""
Factory Methods
"""

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

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

def data_prep(scores,shape):
    """
    Attempts to interpret a list of score sets prior to plotting
    """ 
    auroc = np.reshape([np.mean(s) for s in scores],shape)
    std = np.reshape([np.std(s) for s in scores],shape)
    return auroc,std


"""
Helper Methods
"""

def save_results(my_dict,silent=False):
    """
    Finds unused namespace and pickles dictionary
    """
    # tries to find a model log path that does not exist
    for i in xrange(10001,100001):
        fname = './logs/results_{}.p'.format(i+1)
        if not os.path.exists(fname): break
    if not silent: print 'Creating log file as: {}'.format(fname) # alert user

    # pickle file
    with open(fname, 'w') as f:
        pickle.dump(my_dict,f)

"""
Testing Methods
"""

def model_testing(settings,silent=False):
    """
    Generates a model as defined by settings dictionary and attempts
    to resolve a predictive auROC across each independent fold
    """
    default_settings = {'model_type':'knn'}
    # append dictionary with default systemic parameters (i.e. fixed order, fold design)
    settings = append_dicts(settings,default_settings) 

    # initialize variables 
    scores_test,scores_train = [],[]
    datasets_test,datasets_train = [],[] # lists for storing (guesses,labels) for each type

    # declare model type, calls the right model function
    # DONE: add branch weight to spiNN system
    if settings['model_type'] == 'knn':
        print 'Starting kNN model...'
        m = kNN.BuildModel(settings)
    if settings['model_type'] == 'cnn':
        print 'Starting cNN model...'
        m = cNN.BuildModel(settings)
    if settings['model_type'] == 'spinn':
        print 'Starting spiNN model...'
        m = spiNN.BuildModel(settings)

    # folds the data by parameter specifications
    m.fold_data()

    # removed the fold matching procedure (possible useful, maybe not)
    # iterate across all data folds
    for j in xrange(m.fold_total):
        m.fold_pick(j) # select a fold of data
        m.network_initialization() # initialized the network
        m.train() # train the network on selected fold
       
        # generate model guesses
        datasets_test.append((m.predict(m.test_data),m.test_labels))
        datasets_train.append((m.predict(m.train_data),m.train_labels))

        # generate and store auROC scores 
        datasets_test.append(statistics.auroc(datasets_test[-1][0],datasets_test[-1][1]))
        datasets_train.append(statistics.auroc(datasets_train[-1][0],datasets_train[-1][1]))

        ## NOTE: to save memory, I am *not* going to save models in general runs, but this is something to change
        # m.save_model()
        
    # creates a results dictionary, pickles it in logs folder
    results = {
              'test_auroc':scores_test,
              'train_auroc':scores_train,
              'test_dataset':datasets_test,
              'train_dataset':datasets_train,
              'model_settings':settings
              }

    # save results to an unused file location, with settings metadata
    save_results(results) 


