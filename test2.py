


import os
import tensorflow as tf
import numpy as np



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sess = tf.InteractiveSession()

aa_count = 4
length = 5


def matmul_tensor(tensor):
    return [tf.map_fn(lambda x: tf.matmul(tf.expand_dims(x[:,0],1),tf.expand_dims(x[:,1],0)),t) for t in tensor]    

def split_tensor(tensor,axis=2):
    return tf.split(tensor,[1 for i in xrange(tensor.get_shape()[axis])],axis)

def join_tensor(a,b,axis=1,axis2=2):
    return tf.squeeze(tf.stack([a,b],axis=axis),axis=axis2)


# Initialize placeholder
input_tensor = tf.placeholder(tf.int32, shape=(None,5))

# Find system properties
aa_count = 4
length = 5
pairs = [(i,j) for i in xrange(0,length-1) for j in xrange(i+1,length)]
pair_count = len(pairs)

# pairwise layer
layer = np.array([[1,2,3,5],[4,1,2,3],[1,1,1,1],[2,7,6,9]])
layer_pw = np.stack([layer for i in xrange(pair_count)])

# manipulate data to arrange PW matrix while preserving batch
input_sw = split_tensor(input_tensor,axis=1)
input_pw = [join_tensor(input_sw[i],input_sw[j]) for i,j in pairs]

# index
input_pw_values = [[tf.gather_nd(l,i) for i,l in zip(input_pw,layer_pw)]]
input_pw_output = tf.stack(input_pw_values,axis=1)

# get dimensions
print input_pw_output 

# create fake data and such
data = np.array([[1,3,0,2,1],[3,1,2,2,3],[1,1,1,1,1]])

# test results
feed_dict = {input_tensor:data}
print sess.run(input_pw_output,feed_dict=feed_dict)


