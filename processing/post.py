

"""
pre.py

Performs preprocessing on data sets for input into ML models
"""


# standard libraries
import pickle
import os
import math
import random

# nonstandard libraries
import numpy as np
import matplotlib.pyplot as plt
from analysis import * # libraries: statistics,visualize

"""
Inputs 
    params >  dictionary with possible arguments:
        results_fname - [str], fname in logs
        rounds - [tuple], which rounds to include
        label - [str], which folder to look for data in
"""

def start(params, graphing=True):

    default_params = defaults()
    params = append_dicts(params,default_params)

    # Check if file can be found firstoff
    if not os.path.isfile('./logs/'+params['results_fname']):
        print 'Current directory:',os.getcwd()
        print 'Results file not found ({})! Exiting...'.format(params['results_fname'])
        return None
    
    # Unpickle and load
    results = pickle.load(open('./logs/'+params['results_fname'],'r'))
    print 'Data set:',params['data_label']
    print ' > Test auROC  - ',np.mean(results['test_auroc']), '(STD: ',np.std(results['test_auroc']) ,')'
    print ' > Train auROC - ',np.mean(results['train_auroc']),'(STD: ',np.std(results['train_auroc']),')'


    if graphing:
        # prep subplot figure
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)

        # extract some results
        train_predict = np.concatenate([results['train_dataset'][i][0] for i in xrange(len(results['train_dataset']))])
        train_actual = np.concatenate([results['train_dataset'][i][1] for i in xrange(len(results['train_dataset']))])
        test_predict = np.concatenate([results['test_dataset'][i][0] for i in xrange(len(results['test_dataset']))])
        test_actual = np.concatenate([results['test_dataset'][i][1] for i in xrange(len(results['test_dataset']))])

        # fills subplot 1
        visualize_params = {'title':params['data_label'] + ' testing set','fig':fig,'ax':ax1}
        visualize.accuracy(test_predict,test_actual,custom_params=visualize_params)

        # fills subplot 2
        visualize_params = {'title':params['data_label'] + ' training set','fig':fig,'ax':ax2}
        visualize.accuracy(train_predict,train_actual,custom_params=visualize_params)

        # fills subplot 3
        visualize_params = {'labels':('Train Data','Testing Data'),'fig':fig,'ax':ax3}
        visualize.auroc_plot((train_predict,test_predict),(train_actual,test_actual),custom_params=visualize_params)

        # fills subplot 4
        #visualize_params = {'labels':('Train Data','Testing Data'),'fig':fig,'ax':ax3}
        #visualize.auroc_plot((train_predict,test_predict),(train_actual,test_actual),custom_params=visualize_params)

        # display plot
        plt.show(block=False)
        raw_input('Press enter to close...')
        plt.close()



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



#######################
## PARAMETER PASSING ##
#######################

def defaults():
    return {
            'results_fname':'./logs/results_100001.p',
            'data_label':'',
           }




