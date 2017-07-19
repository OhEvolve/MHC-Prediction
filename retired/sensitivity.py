'''

Project: Neural Network for MHC Peptide Prediction
Class(s): (none) 
Function: Organizes main pipeline execution and testing 

Author: Patrick V. Holec
Date Created: 2/3/2017
Date Updated: 3/20/2017

This is for actual data testing

'''

# standard libraries
import time

# nonstandard libraries
import numpy as np

# personal libraries
from models import * # libraries: kNN,cNN,spiNN
from analysis import * # libraries: visualize
from landscape import generate_test_set as gts,parameterize_data as pd,compile_data as cd


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def data_prep(scores,shape):
    
    auroc = np.reshape([np.mean(s) for s in scores],shape)
    std = np.reshape([np.std(s) for s in scores],shape)
    
    return auroc,std


def dynamic_testing(models,labels,added_settings,model_settings,fixed_order = False,versions = ['test','train']):
    # placeholders
    scores_test,scores_train = [],[]
    fold_orders = []

    for i,model,settings in zip(xrange(len(models)),models,added_settings):

        all_model_settings = merge_dicts(model_settings,settings)
        
        all_actuals_test,all_guesses_test = [],[]
        all_actuals_train,all_guesses_train = [],[]

        if 'knn' == model:
            print 'Starting kNN model...'
            m = kNN.BuildModel(all_model_settings)
        if 'cnn' == model: 
            print 'Starting cNN model...'
            m = cNN.BuildModel(all_model_settings)
        if 'spinn' == model: 
            print 'Starting spiNN model...'
            m = spiNN.BuildModel(all_model_settings)

        m.fold_data()

        for j in xrange(m.fold_total):
            # fix ordering if chosen
            if fixed_order:
                if i == 0:
                    fold_orders.append(m.fold_pick(j))
                m.set_fold(fold_orders[j][0],fold_orders[j][1])
            # non-fixed ordering
            else:
                m.fold_pick(j)
                
            # run model
            m.network_initialization()
            m.train()
            
            for version in ['test','train']:
                if version == 'test':
                    input_data,input_labels = m.test_data,m.test_labels
                elif version == 'train':
                    input_data,input_labels = m.train_data,m.train_labels
            
                # generate model guesses
                guess = m.predict(input_data)

                # save guesses           
                if version == 'test':
                    all_guesses_test.append(guess)
                    all_actuals_test.append(input_labels)
                elif version == 'train':
                    all_guesses_train.append(guess)
                    all_actuals_train.append(input_labels)
                    
        if 'test' in versions: 
            scores_test.append(visualize.auroc(all_guesses_test,all_actuals_test,labels = labels, graph = False))
        if 'train' in versions: 
            scores_train.append(visualize.auroc(all_guesses_train,all_actuals_train,labels = labels, graph = False))
            
        # check on guesses
        visualize.comparison(guess,input_labels)

    
    return scores_test,scores_train,fold_orders

####################
### MAIN TESTING ###
####################

# Turn this all off if you're not using different rounds of data (not huge comp. cost, though)
if True:
    data_selection = ['A12_0-1','A12_1-2','A12_2-3','A12_3-4','A12_4-5']
    # data_selection = ['A12']

    params = {
             'silent':False
             }

    enrichment = cd.Enrichment(params)
    enrichment.load_data()
    enrichment.build_profiles()

    for ds in data_selection:
        parameters = {'length':14,'aa_count':21,'characters':'ACDEFGHIKLMNPQRSTVWY_'}
        pd.ParameterizeData(parameters,label=ds)


kNN = reload(kNN)
cNN = reload(cNN)
spiNN = reload(spiNN)
visualize = reload(visualize)

model_settings = {
                 'num_epochs':40,
                 'batch_size':10,
                 'learning_rate':0.0005,
                 'data_augment':False,
                 'data_normalization':False,
                 'fc_fn':('linear','linear','linear'),
                 'fc_dropout':(1.0,1.0,1.0),
                 'fc_depth':(8,1),
                 'fc_layers':2,
                 'test_fraction': 0.25,
                 'data_label':'A12',
                 'silent':False,
                 'reg_magnitude':0.2,
                 'model_log':True
                 }


#'''# Testing for sensitivity across rounds of selection
total = 2
intercept = 0

x_axis_label = 'Data Normalization'
models = ['spinn' for i in xrange(total)]
labels = [str(bool(i)) for i in xrange(intercept,total+intercept)]
added_settings = [{'data_label':'A12_3-4','data_normalization':bool(i)} for i in xrange(intercept,total+intercept)]
#'''

'''# Testing for sensitivity across rounds of selection
total = 4
intercept = 1

x_axis_label = 'Rounds'
models = ['spinn' for i in xrange(total)]
labels = ['{}->{}'.format(i,i+1) for i in xrange(intercept,total+intercept)]
added_settings = [{'data_label':'A12_{}-{}'.format(i,i+1)} for i in xrange(intercept,total+intercept)]
#'''

'''# Testing for k-parameter sensitivity
total = 13

x_axis_label = 'K nearest neighbors'
models = ['knn' for i in xrange(total)]
labels = [str(2*i+1) for i in xrange(total)]
added_settings = [{'k_nearest':2*i+1} for i in xrange(total)]
#'''

'''# Testing for FC depth in spiNN
total = 4
intercept = 0

x_axis_label = 'FC depth (?,1)'
models = ['spinn' for i in xrange(total)]
labels = [str(2**i) for i in xrange(intercept,total+intercept)]
added_settings = [{'fc_depth':(2**i,1)} for i in xrange(intercept,total+intercept)]
#'''

'''# Testing for regularization magnitude
total = 3
intercept = 1

x_axis_label = 'Regularization Magnitude'
models = ['spinn' for i in xrange(total)]
labels = [0.05*i for i in xrange(intercept,total+intercept)]
added_settings = [{'reg_magnitude':0.05*i} for i in xrange(intercept,total+intercept)]
#'''

'''# Testing for sitewise depth
total = 5
intercept = 1

x_axis_label = 'SW Depth'
models = ['spinn' for i in xrange(total)]
labels = [i for i in xrange(intercept,total+intercept)]
added_settings = [{'sw_depth':i} for i in xrange(intercept,total+intercept)]
#'''

'''# Testing for FC function
total = 3
intercept = 1

x_axis_label = 'FC Depth'
models = ['spinn' for i in xrange(total)]
labels = ['lin','relu','sig']
added_settings = [{'fc_fn':('linear','linear')},{'fc_fn':('relu','linear')},{'fc_fn':('sigmoid','linear')}]
#'''

scores_test,scores_train,fold_orders = dynamic_testing(models,labels,added_settings,model_settings,versions=['train','test'])

visualize = reload(visualize)

auroc,std = data_prep(scores_test+scores_train,(2,total))
visualize.auroc_regime(auroc,std = std, xlabels = labels,
                       metalabels = ['Validation','Training'],
                       show_graph=False,x_axis_label = x_axis_label)

print 'Run finished!'
