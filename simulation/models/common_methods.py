
""" 
file: /simulation/models/common_methods.py 
function: collection of common model generation techniques
author: PVH
"""

#  standard libraries
import collections

# nonstandard libraries
import tensorflow as tf

# homegrown libraries


"""
FACTORY METHODS
"""

def flatten(x):
    if type(x) in [list,tuple]:
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def split_tensor(tensor,axis=2):
    """ Splits input tensor across given axis into 1D slices """
    return tf.split(tensor,[1 for i in xrange(tensor.get_shape()[axis])],axis)


def join_tensor(a,b,axis=1,axis2=2):
    """ Joins two tensors together across a given axis, then squeezes on axis 2 """
    return tf.squeeze(tf.stack([a,b],axis=axis),axis=axis2)


def normal_variable(shape,seed=None):
    """Create a weight variable with appropriate initialization"""
    if seed: tf.set_random_seed(seed) # sets seed if requested
    initial = tf.truncated_normal(shape, stddev=1.0)
    return tf.Variable(initial)


def ones_constant(shape):
    """Create a tensor with ones """
    return tf.ones(shape,dtype=tf.float32)


def zeros_constant(shape):
    """Create a tensor with zeros """
    return tf.zeros(shape,dtype=tf.float32)


def uniform_variable(shape,value=0.0):
    """Create a variable initialized to specific value """
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride=1, padding='SAME'):
    """ Performs a 2D convolution of W across x given stride and padding """
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


def max_pool(x, stride=2, filter_size=2, padding='SAME'):
    """ Performs a max pool operation on x given stride, filter size and padding """ 
    if not(type(stride) == list and type(filter_size) == list):
        filter_size,stride = [1, filter_size, filter_size, 1],[1, stride, stride, 1]
    return tf.nn.max_pool(x, ksize=filter_size,strides=stride, padding=padding)


def activation_fn(inp,fn=None,dropout=1.0):
    """ Applies an activiation function on input tensor """ 
    if fn == 'relu': return tf.nn.dropout(tf.nn.relu(inp), dropout)
    elif fn == 'relu6': return tf.nn.dropout(tf.nn.relu6(inp), dropout)
    elif fn == 'sigmoid': return tf.nn.dropout(tf.nn.sigmoid(inp), dropout)
    elif fn == 'tanh': return tf.nn.dropout(tf.nn.tanh(inp), dropout)
    elif fn == 'softplus': return tf.nn.dropout(tf.nn.softplus(inp), dropout)
    elif fn == 'linear': return tf.nn.dropout(inp,dropout)
    elif type(fn) == str: print '{} function not recognized, using default ({})...'.format(fn,'linear')
    return tf.nn.dropout(inp, dropout)


def start_tf_engine(mode):
    # Choose whether to use CPU or GPU, configure devices
    if mode == 'CPU': 
        config = tf.ConfigProto(device_count = {'GPU':0}) # use this to set to CPU
    elif mode == 'GPU': 
        config = tf.ConfigProto() # use this to set to GPU
        config.log_device_placement = False # super verbose mode
        config.gpu_options.allow_growth = True
    else:
        print 'No config given, defaulting to CPU...'
        config = tf.ConfigProto(device_count = {'GPU':0})

    # Start tensorflow engine, and send out
    return tf.Session(config=config)


def loss(y,**kwargs):
    """ Predefined loss function construction method """ 
    # allow user to modify particular parameters
    options = {'silent':False,
               'loss_type':'l2',
               'loss_magnitude':1,
               'normalize':1}
    options.update(kwargs) # update options with user input

    # check on some types 

    if type(y) == dict: # convert dictionary entries into list of tensors
        y = flatten(y.values())
    elif type(y) != list: 
        y = [y] # convert single entry into list of tensors
    else:
        y = flatten(y)

    # base loss
    dtype = tf.float32
    loss_total,norm = tf.constant(0,dtype=dtype),tf.constant(1,dtype=dtype)
    for i in y:
        if options['normalize'] == True: 
            norm = tf.size(i,out_type=dtype)
        else: 
            norm = tf.constant(1,dtype=dtype) 

        if options['loss_type'] == 'l1': 
            loss_total += options['loss_magnitude']*tf.cast(tf.reduce_sum(tf.abs(i)),dtype=dtype)/norm
        elif options['loss_type'] == 'l2': 
            loss_total += options['loss_magnitude']*tf.cast(tf.nn.l2_loss(i),dtype=dtype)/norm

    # returns summed total
    return loss_total


