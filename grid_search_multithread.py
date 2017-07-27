## TODO: make logs directory if not there previously
## T

'''
Project: Neural Network for MHC Peptide Prediction
Class(s): BuildNetwork
Function: Generates specified neural network architecture (data agnostic)

Author: Patrick V. Holec
Date Created: 2/2/2017
Date Updated: 5/18/2017
'''




'''
Library Importation
'''

# standard libraries
import itertools
from multiprocessing import Pool

# nonstandard libraries

# homegrown libraries
import standard



'''
Default Function
'''

def main():
    """ If file called as script, this function will run """
    params = default_params()
    variables = default_variables()

    grid_search(params,variables)
    print 'Finished!'




'''
Factory Methods
'''

def append_dict(old_dict,*dict_args):
    """
    Given an original dictionary and any number of additional dicts, adds
    new entries to original if they are not already present
    """
    for dictionary in dict_args:
        for key in dictionary.keys():
            if not key in old_dict.keys():
                #print 'Adding parameter: {} -> {}'.format(key,dictionary[key])
                old_dict[key] = dictionary[key]
    return old_dict




'''
Main Methods
'''

def grid_search(params,variables,silent=False):
    """
    Takes a model defined by default parameters (params) and performs a grid search
    over the the variable dictionary (variables), saving results in the log folder
    """
    # generate default model parameters
    params = default_params('spinn')
    variables = default_variables('spinn')
    
    # create list of dictionaries for parameters
    keys,values = [i[0] for i in variables.items()],[i[1] for i in variables.items()]
    variable_dicts = [dict([(k,v) for k,v in zip(keys,vals)]) for vals in list(itertools.product(*values))]

    # at this point, we can consider multithreading through dictionary arguments
    # TODO: multithread this loop

    print 'Generating settings dictionaries...'
    settings_dicts = [append_dict(variable_dict,params) for variable_dict in variable_dicts] # stitches parameters together
    print 'All settings dictionaries generated!'

    pool = Pool() # creates a worker pool equal to the number of cores
    pool.map(standard.model_testing,settings_dicts)

    ''' Old
    for variable_dict in variable_dicts:
        settings = append_dict(variable_dict,params)
        standard.model_testing(settings)
    #'''



'''
Default Model Defintions (Pretty boring stuff)
'''

def default_variables(model_type='spinn'): 
    """ Returns dictionary of common grid search parameters """ 
    if model_type == 'knn':
        return {}
    elif model_type == 'cnn':
        return {}
    elif model_type == 'spinn':
        return {
                'reg_magnitude':[0.01,0.3,0.1,0.3,1.0,3.0,10.0,30.0,100.0],
                'reg_type':['l1','l2'],
                'learning_rate':[0.1,1.0,10.0],
                'sw_pw_ratio':[0.1,0.25,0.5,0.75,0.9],
                'sw_depth':[1,2,4],
                'pw_depth':[1,2,4],
                'fc_fn':[('linear','linear'),('relu','relu'),('sigmoid','sigmoid')],
                'fc_dropout':[(1.0,1.0),(0.75,0.75),(0.5,0.5),(0.25,0.25)],
                'fc_depth':[(4,1),(8,1),(16,1),(32,1),(64,1)]
               }

# DONE: Add parameters to variables
# TODO: spiNN loss_coeff vs. loss_magnitude
def default_params(model_type='spinn'):
    """ Returns dictionary of default model parameters """ 
    base_params = {
                   # simulation settings
                   'data_augment':False,
                   'learning_rate':0.01,
                   'data_normalization': True,
                   'silent': True,
                   'test_fraction': 0.2,
                   'batch_size':100,
                   'num_epochs':50,
                   'learning_rate':0.1,
                   'data_label':'test'
                   }
    if model_type == 'knn':
        return None
    elif model_type == 'cnn':
        return None
    elif model_type == 'spinn':
        model_params = {
                        # overall network parameters
                        'model_type':'spinn',
                        'fc_layers':2,
                        'sw_pw_ratio':0.5, # relative weight of sw branch relative to pw branch (range -> [0,1])
                        # sitewise/pairwise parameters
                        'pw_depth':1,
                        'sw_depth':1,
                        # fully connected parameters
                        'fc_depth':(16,1),
                        'fc_fn':('sigmoid ','sigmoid','linear'),
                        'fc_dropout':(1.0,1.0),
                        # loss parameters
                        'loss_type':'l2',
                        'loss_magnitude':1.0,
                        # regularization parameters
                        'reg_type':'l2',
                        'reg_magnitude':0.01
                       }
        return append_dict(base_params,model_params) 
    else:
        print 'Model type not recognized, returning base parameters for default model...'
        return base_params




'''
Default Routine
'''

if __name__ == '__main__':
    main()




