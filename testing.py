
"""
This is a building ground for new ideas and optimization 
NOT FOR RUNNING REAL SIMULATION
"""


#'''

from models import * # libraries: kNN,cNN,spiNN

for i in xrange(1,5):
    params = {
             'sw_depth':2,
             'pw_depth':i,
             'silent':True,
             }


    m = spiNN.BuildModel(params)
    m.fold_data()
    m.fold_pick(0)
    m.network_initialization()


    print 'SW depth:',m.sw_depth
    print 'PW depth:',m.pw_depth

    print 'Parameters:',m.count_trainable_parameters()

#'''

'''

import numpy as np
import tensorflow as tf

y = tf.constant([[1,2],[3,4]])

a = (tf.constant(1.0)/tf.size(y,out_type=tf.float32))

sess = tf.Session()

print sess.run(a)

#'''

