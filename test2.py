


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

def join_tensor(a,b,axis=2):
    return tf.squeeze(tf.stack([a,b],axis=axis),axis=3)


# Initialize placeholder
input_tensor = tf.placeholder(tf.float64, shape=(None,4,5,1))

# Find system properties
aa_count = input_tensor.get_shape()[1]
length = input_tensor.get_shape()[2]
pairs = [(i,j) for i in xrange(0,length-1) for j in xrange(i+1,length)]

# manipulate data to arrange PW matrix while preserving batch
input_split = split_tensor(tf.squeeze(input_tensor,axis=3),axis=2)
input_paired = [join_tensor(input_split[i],input_split[j]) for i,j in pairs]
input_matmul = matmul_tensor(input_paired)
input_pw = tf.stack(input_matmul,axis=1)

print input_pw.get_shape()

# create fake data and such
rx = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,0,1,0],[0,0,1,0,1]])
data = np.stack([rx for i in xrange(8)])

# test results
feed_dict = {input_tensor:data}
print sess.run(input_pw,feed_dict=feed_dict)


