

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

def single(params, graphing=True):

    default_params = defaults()
    params = append_dicts(params,default_params)

    # Check if file can be found firstoff
    if not os.path.isfile('./logs/'+params['results_fname']):
        if not os.path.isfile(params['results_fname']):
            print 'Current directory:',os.getcwd()
            print 'Results file not found ({})! Exiting...'.format(params['results_fname'])
            return None
        else:
            results = pickle.load(open(params['results_fname'],'r'))
    else:
        results = pickle.load(open('./logs/'+params['results_fname'],'r'))


    
    # Unpickle and load
    print 'Data set:',params['data_label']
    print ' > Test auROC  - ',np.mean(results['test_auroc']), '(STD: ',np.std(results['test_auroc']) ,')'
    print ' > Train auROC - ',np.mean(results['train_auroc']),'(STD: ',np.std(results['train_auroc']),')'

    # extract some results
    train_predict = np.concatenate([results['train_dataset'][i][0] for i in xrange(len(results['train_dataset']))])
    train_actual = np.concatenate([results['train_dataset'][i][1] for i in xrange(len(results['train_dataset']))])
    test_predict = np.concatenate([results['test_dataset'][i][0] for i in xrange(len(results['test_dataset']))])
    test_actual = np.concatenate([results['test_dataset'][i][1] for i in xrange(len(results['test_dataset']))])

    lim = 50
    print 'Training comparison:'
    #for a,b in zip(train_predict[:lim],train_actual[:lim]):
    #    print ' >',a,b

    print 'Average (train):',np.mean(train_actual)
    print 'Average (test):',np.mean(test_actual)
    print results.keys()

    if graphing:
        # prep subplot figure
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)

        # fills subplot 1
        visualize_params = {'title':params['data_label'] + ' testing set','fig':fig,'ax':ax1,'mode':params['mode']}
        visualize.accuracy(test_predict,test_actual,custom_params=visualize_params)

        # fills subplot 2
        visualize_params = {'title':params['data_label'] + ' training set','fig':fig,'ax':ax2,'mode':params['mode']}
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


def batch(params, specific_dicts = None, graphing = True):

    # search for whether simulations are completed or not
    print 'Loading all files...'
    log_dir = './logs'
    all_files = [log_dir+'/'+i for i in os.listdir(log_dir) 
            if os.path.isfile(os.path.join(log_dir,i)) and 'results_' in i]

    all_results = []
    for f in all_files:
        try: all_results.append(pickle.load(open(f,'r')))
        except ValueError: print 'Skipped:',f
    
    # initialize some variables
    test_aurocs,train_aurocs = [],[]

    # check through pickled files and save results (if matching specific dict)
    print 'Shuffling through files...'
    for result in all_results:
        if specific_dicts == None or result['model_settings'] in specific_dicts: 
            test_aurocs.append(result['test_auroc'])
            train_aurocs.append(result['train_auroc'])
        else:
            print 'Skipping...'

    #
    x,y = [np.mean(t) for t in test_aurocs],[np.mean(t) for t in train_aurocs]
    xerr,yerr = [np.std(t) for t in test_aurocs],[np.std(t) for t in train_aurocs]

    # Find top score
    ind = x.index(max(x))
    print 'Top scoring:'
    print 'Train: {} +- {}'.format(x[ind],xerr[ind])
    print 'Test: {} +- {}'.format(y[ind],yerr[ind])
    print '- Model parameters -'
    print all_results[ind]['model_settings']

    if graphing:
        # Start graphical component
        plt.figure()
        plt.errorbar(x,y,xerr,yerr,fmt='o', ecolor='g', capthick=2)
        plt.title('Grid search results')
        plt.xlabel('Training auROC')
        plt.ylabel('Testing auROC')

        plt.show(block=False)
        raw_input('Press enter to close...')
        plt.close()

    else:
        # Update user
        for a1,a2,b1,b2 in zip(x,xerr,y,yerr):
            print 'Diff: {}'.format(b1-a1)

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
            'mode':'attrition'
           }




