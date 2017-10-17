
""" 
file: /simulation/models/default_parameters.py 
function: collection of common model generation techniques
author: PVH
"""

# standard libaries
import random

def spinn():
    """ Initialize a default cNN model """ 
    default_params = {
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

    return default_params


def cnn():
    """ Initialize a default cNN model """ 
    default_params = {
                     # training parameters
                     'batch_size':100,
                     'num_epochs':50,
                     'learning_rate':1,
                     # overall network parameters
                     'cnn_layers':1, # includes conv & pool
                     'fc_layers':2,
                     # convolutional parameters
                     'conv_stride':(1,1), # not sure if we can handle others
                     'conv_filter_width':(2,4),
                     'conv_depth':(2,4),
                     'conv_padding':('SAME','SAME'),
                     # max pool parameters
                     'pool_stride':(2,2),
                     'pool_filter_width':(3,3),
                     'pool_padding':('SAME','SAME'),
                     # fully connected parameters
                     'fc_depth':(16,1),
                     'fc_fn':('sigmoid','sigmoid','linear'),
                     'fc_dropout':(0.5,1.0),
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

    return default_params





