
""" 
file: /simulation/models/common_methods.py 
function: collection of common model generation techniques
author: PVH
"""

#  standard libraries

# nonstandard libraries
import tensorflow as tf

# homegrown libraries


"""
FACTORY METHODS
"""

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


def constant_variable(shape,value=0.0):
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


def network_loss(y,y_real,W,params):
    """ Generates a network loss function given y/y_real and regularization parameters """ 
    val = tf.constant(1.0) # make a quick float32 variable for shorter calls

    # base loss
    if params['loss_type'] == 'l1': loss = params['loss_magnitude']*(val/tf.cast(tf.size(y),tf.float32))*tf.reduce_sum(tf.abs(tf.subtract(y,y_real)))
    elif params['loss_type'] == 'l2': loss = params['loss_magnitude']*(val/tf.cast(tf.size(y),tf.float32))*tf.nn.l2_loss(tf.subtract(y,y_real))
    else: loss = 0

    # regularization loss
    if params['reg_type'] == 'l1': loss += params['reg_magnitude']*tf.reduce_sum([(val/tf.cast(tf.size(w),tf.float32))*tf.reduce_sum(tf.abs(w)) for w in W])
    elif params['reg_type'] == 'l2': loss += params['reg_magnitude']*tf.reduce_sum([(val/tf.cast(tf.size(w),tf.float32))*tf.nn.l2_loss(w) for w in W])
    return loss


