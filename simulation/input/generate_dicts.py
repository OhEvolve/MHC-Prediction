
""" 
file: /input/generate_dicts.py 
function: generative techniques 
author: PVH
"""

# standard libraries
import random
import itertools


def generate_dicts(*args,**kwargs):

    """ Generates dictionaries that exhaustively cover parameter space in given dicts """
    
    if 'default' in kwargs.keys(): return default_grid_search_dicts()
    
    # compile all dictionaries into one location
    all_params = {}
    for arg in args:
        assert type(arg) == dict, 'Argument type not supported'
        for k,v in arg.items():
            assert type(v) in (tuple,list), 'Dictionary value not supported: {},{}'.format(type(v),v)
            all_params[k] = v

    # generate parameter combinations
    dict_list = []
    for params_set in list(itertools.product(*all_params.values())):
        dict_list.append(dict([(k,v) for k,v in zip(all_params.keys(),params_set)]))

    # return resulting dictionaries
    return dict_list


def append_dicts(old_dict,*dict_args):
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


""" Returns grid search dictionaries """
def default_grid_search_dicts():
    
    model_params = {
                     # training parameters
                     'batch_size':100,
                     'num_epochs':50,
                     'learning_rate':1,
                     # overall network parameters
                     'fc_layers':2,
                     'sw_pw_ratio':0.5, # relative importance of sw branch relative to pw branch
                     # sitewise/pairwise parameters
                     'pw_depth':1,
                     'sw_depth':1,
                     # fully connected parameters
                     'fc_depth':(16,1),
                     'fc_fn':('sigmoid ','sigmoid'),
                     'fc_dropout':(1.0,1.0),
                     # loss parameters
                     'loss_type':'l2',
                     'loss_magnitude':1.0,
                     # regularization parameters
                     'reg_type':'l2',
                     'reg_magnitude':0.01,
                     # system parameters 
                     'silent': False,
                     'tf_device':'CPU',
                     'seed':random.randrange(99999999)
                     }
    model_params['repeats'] = 3
    model_params['num_epochs'] = 25 
    model_params['silent'] = True
       
    base_params = { 
                  'sw_pw_ratio':[0.0,0.5,1.0],
                  'pw_depth':[1,2,4],
                  'sw_depth':[1,2,4],
                  'data_label':['A12.txt','F3.txt','5cc7.txt','226.txt']
                  }

    d1_params = { 'fc_layers':[1], 'fc_depth':[(1,)], 
            'fc_fn':[('linear',),('relu',),('sigmoid',)],
            'fc_dropout':[(1.0,)] }
    d2_params = { 'fc_layers':[2], 'fc_depth':[(2,1),(4,1),(8,1)], 
            'fc_fn':[('linear','linear'),('relu','relu'),('sigmoid','sigmoid')], 
            'fc_dropout':[(1.0,1.0)] }
    d3_params = { 'fc_layers':[3], 'fc_depth':[(4,2,1),(16,4,1),(64,8,1)], 
            'fc_fn':[('linear','linear','linear'),('relu','relu','relu'),
                ('sigmoid','sigmoid','sigmoid')], 'fc_dropout':[(1.0,1.0,1.0)] }

    all_dicts = generate_dicts(base_params,d1_params) +  \
                generate_dicts(base_params,d2_params) +  \
                generate_dicts(base_params,d3_params) 
    all_dicts = [append_dicts(d,model_params) for d in all_dicts]
    return all_dicts


