

""" 
file: /testing/batch_model.py 
function: perform unit tests on all available functions to ensure function 
author: PVH
"""



# standard libraries
import random
import time
import pickle
import os
from multiprocessing.dummy import Pool as ThreadPool

# nonstandard libraries
import numpy as np

# homemade libraries
from common_methods import *
from simulation.input import *
from simulation.output import *



"""
Factory Methods
"""

def batch_model(all_params,*args,**kwargs):
    
    # allow user to modify particular parameters
    options = {
              'silent': False,
              'thread_count': 4
              } 

    for arg in args: options.update(arg) # update with input dictionaries
    options.update(kwargs) # update with keyword arguments
    
    # assertion check

    # create thread classes specific to each exclusionary parameter
    exclusion_dict = {}
    exclusion_keys = ['model_type','data_label','fold_count','repeats']

    if not options['silent']: print 'Starting exclusion splitting...'

    for params in all_params:
        
        exclusion_values = tuple([params[key] for key in exclusion_keys if key in params.keys()])

        if not exclusion_values in exclusion_dict.keys():
            exclusion_dict[exclusion_values] = [params] 
        else:
            exclusion_dict[exclusion_values].append(params)

    exclusion_usage = [(k,[len(v),1]) for k,v in exclusion_dict.items()] 
    
    # if we have more threads than exclusionary classes, divide and conquer
    if options['thread_count'] > len(exclusion_dict):

        # go through each excess thread and assign to group in need
        for i in xrange(options['thread_count'] - len(exclusion_dict)):
            sorted(exclusion_usage, key=lambda x: x[1][0]/x[1][1])
            exclusion_usage[-1][1][1] += 1 

    # assign parameters to groups
    all_params_exclusion = []
    for (k,v) in exclusion_usage:
        random.shuffle(exclusion_dict[k])
        for p in partition(exclusion_dict[k],v[1]):
            all_params_exclusion.append({
                                       'params_list': p,
                                       'options': dict([(key,params[key]) for key in exclusion_keys 
                                            if key in exclusion_dict[k][0].keys()])
                                    })


    print 'Starting the following threads:'
    for i,p in enumerate(all_params_exclusion):
        print ' >',i+1,':',len(p['params_list'])
   
    time.sleep(1.0)

    # create a thread pool, and pass through threads 
    pool = ThreadPool(options['thread_count']) 

    results = pool.map_async(multiple_model,all_params_exclusion).get(999999999)
            

def multiple_model(batch_params):

    """ Tests a single collection of parameters """ 
    """ Pass model parameters, then exclusionary parameters """ 

    # allow user to modify particular parameters
    options = {'silent':False,
               'repeats':1,
               'fold_count':5,
               'model_type':'spinn',
               'data_label':'test.txt'}

    options.update(batch_params['options']) # update with input dictionaries

    # assertions check
    

    # generate model

    model,run_settings = choose_model(options['model_type']) 

    # load data and initialize architecture
    data,data_params = load_data(options['data_label'],
                            encoding=run_settings['encoding'],
                            silent=options['silent'])

    # initialize results
    results = {
              'auroc':[],
              'weights':{}
              }

    data_folds =  {}

    # pregenerate data to save space
    for r in xrange(options['repeats']):
        
        # randomize data each repeat
        data = randomize_data(data,silent=options['silent'])

        for f in xrange(options['fold_count']):

            # select a fold of data
            data_folds[(r,f)] = fold_data(data,fold_index=f,fold_count=options['fold_count'])


    # single model at a time simulation 
    for params in batch_params['params_list']:

        results = {
                  'auroc':[],
                  'weights':{},
                  'params':{}
                  }

        model.network_initialization(data_params,params)

        # for each requested repeat/fold 
        for r in xrange(options['repeats']):

            for f in xrange(options['fold_count']):

                # train neural network on training fold 
                choose_training(model,data_folds[(r,f)],run_settings['type'])
                
                # save auroc score for given repeat,fold
                results['auroc'].append(auroc_score(model,data_folds[(r,f)]['testing']))
                results['weights'] = dict([(k,model.sess.run(w)) for k,w in model.weights.items()])

                # reset variables in model
                model.reset_variables() 
                 

        results['params'] = params
        save_results(results,options['silent'])



"""
Helper Functions
"""

def partition(lst, n): 
    """ Splits a list in n nearly equal parts """
    division = len(lst) / float(n) 
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ]


def save_results(my_dict,silent=False):
    """
    Finds unused namespace and pickles dictionary
    """
    # check if path exists, create if not
    if not os.path.exists('./logs'):
        print '/logs not found, making new directory...'
        os.makedirs('./logs')
    # tries to find a model log path that does not exist
    for i in xrange(100001,1000001):
        fname = './logs/results_{}.p'.format(i+1)
        if not os.path.exists(fname): break
    if not silent: print 'Creating log file as: {}'.format(fname) # alert user

    # pickle file
    with open(fname, 'w') as f:
        pickle.dump(my_dict,f)

    return fname




