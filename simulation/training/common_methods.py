
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

def loss(y,**kwargs):
    # allow user to modify particular parameters
    options = {'silent':False,
               'loss_type':'l2',
               'loss_magnitude':1,
               'normalize':1}
    options.update(kwargs) # update options with user input

    # check on some types 

    if type(y) != list: y = [y]

    # base loss
    dtype = tf.float32
    loss_total,norm = tf.constant(0,dtype=dtype),tf.constant(1,dtype=dtype)
    for i in y:
        if options['normalize'] == True: norm = tf.size(i,out_type=dtype)
        else: norm = tf.constant(1,dtype=dtype) 
        if options['loss_type'] == 'l1': loss_total += options['loss_magnitude']*tf.cast(tf.reduce_sum(tf.abs(i)),dtype=dtype)/norm
        elif options['loss_type'] == 'l2': loss_total += options['loss_magnitude']*tf.cast(tf.nn.l2_loss(i),dtype=dtype)/norm

    # returns summed total
    return loss_total

