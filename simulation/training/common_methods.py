
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

def set_learning_fn(loss,learning_rate,mode):
    """ Generalized learning function, given a loss function """

    if mode.lower() == 'adam':
        return tf.train.AdamOptimizer(learning_rate).minimize(loss) 
    elif mode.lower() == 'gradientdescent':
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss) 
    elif mode.lower() == 'proximalgradientdescent':
        return tf.train.ProximalGradientDescentOptimizer(learning_rate).minimize(loss)
    else:
        raise KeyError('Specified learning method not found ({})'.format(mode))






