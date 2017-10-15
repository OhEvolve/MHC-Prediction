
""" 
file: /simulation/models/default_parameters.py 
function: collection of common model generation techniques
author: PVH
"""

# standard libaries
import random

def spinn():

    default_params = {
                     'data_augment':False,
                     'data_normalization': False,
                     'silent': False,
                     'test_fraction': 0.1,
                     'batch_size':1000,
                     'num_epochs':50,
                     'learning_rate':1,
                     'data_label':'A12',
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
                     'tf_device':'CPU',
                     'seed':random.randrange(99999999)
                     }

    return default_params
