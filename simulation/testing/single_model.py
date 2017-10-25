

""" 
file: /testing/single_model.py 
function: test a single parameter set 
author: PVH
"""

# standard libraries
import time
import pickle
import os

# nonstandard libraries
import numpy as np

# homemade libraries
from common_methods import *
from simulation.input import *
from simulation.output import *

"""
Factory Methods
"""

def single_model(*args,**kwargs):

    """ Tests a single collection of parameters """ 
    
    # allow user to modify particular parameters
    options = {'silent':False,
               'repeats':1,
               'fold_count':5,
               'model_type':'spinn',
               'data_label':'test.txt'}

    for arg in args: options.update(arg) # update with input dictionaries
    options.update(kwargs) # update with keyword arguments

    # assertions check
    

    # generate model

    model,run_settings = choose_model(options['model_type']) 

    # load data and initialize architecture
    data,params = load_data(options['data_label'],
                            encoding=run_settings['encoding'],
                            silent=options['silent'])

    model.network_initialization(options,params)

    # initialize results
    results = {
              'step_loss':[],
              'auroc':{'testing':[],'training':[]},
              'weights':{}
              }


    # for each requested repeat (no need to save since we are just doing this for one model)
    for r in xrange(options['repeats']):

        # randomize data each repeat
        data = randomize_data(data,silent=options['silent'])

        for f in xrange(options['fold_count']):

            # select a fold of data
            data_folded = fold_data(data,fold_index=f,fold_count=options['fold_count'])

            # train neural network on training fold 
            step_loss = choose_training(model,data_folded,run_settings['type'])
            
            # save auroc score for given repeat,fold
            results['auroc']['testing'].append(auroc_score(model,data_folded['testing']))
            results['auroc']['training'].append(auroc_score(model,data_folded['training']))
            results['weights'][f] = dict([(k,model.sess.run(w)) for k,w in model.weights.items()])
            results['step_loss'] = step_loss

            # reset variables in model
            model.reset_variables() 
             
    save_results(results,silent=options['silent'])

    return results
    

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


"""
Testing Methods
"""

def test(params,silent=False):
    """
    Generates a model as defined by settings dictionary and attempts
    to resolve a predictive auROC across each independent fold
    """
    # create test specific dictionaries
    default_params = {
                    'repeats':1,
                     }

    # append dictionary with default systemic parameters (i.e. fixed order, fold design)
    settings = append_dicts(params,default_params) 

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


    # removed the fold matching procedure (possible useful, maybe not)
    
    for i in xrange(settings['repeats']): # iterate across declared repeats
        
        print 'Starting repeat {}...'.format(i+1)

        # folds the data by parameter specifications
        m.fold_data()

        for j in xrange(m.fold_total): # iterate across all data folds

            print 'Starting fold {}...'.format(j+1)

            m.fold_pick(j) # select a fold of data
            m.network_initialization() # initialized the network

            # generate pretraining model guesses
            #datasets_test.append((m.predict(m.test_data),m.test_labels))
            #datasets_train.append((m.predict(m.train_data),m.train_labels))

            m.train() # train the network on selected fold
           
            # generate model guesses
            datasets_test.append((m.predict(m.test_data),m.test_labels))
            datasets_train.append((m.predict(m.train_data),m.train_labels))

            # generate and store auROC scores 
            scores_test.append(statistics.auroc(datasets_test[-1][0],datasets_test[-1][1]))
            scores_train.append(statistics.auroc(datasets_train[-1][0],datasets_train[-1][1]))

            ## NOTE: to save memory, I am *not* going to save models in general runs, but this is something to change
            #m.save_model()
        
    # creates a results dictionary, pickles it in logs folder
    results = {
              'test_auroc':scores_test,
              'train_auroc':scores_train,
              'test_dataset':datasets_test,
              'train_dataset':datasets_train,
              'model_settings':settings
              }

    # save results to an unused file location, with settings metadata
    fname = save_results(results) 
    print results['test_auroc']
    print results['train_auroc']

    return fname 
    
    #visualize.comparison(test_dataset[0],k



