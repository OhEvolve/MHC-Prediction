

import os
import tensorflow as tf
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sess = tf.InteractiveSession()

aa_count = 4
length = 5


# Initialize placeholder
input_indices = tf.placeholder(tf.int32, shape=(None,2))

# create fake indices
matrix = np.array([[5,4],[3,2]],dtype=np.int32)
output = tf.gather_nd(matrix,input_indices)

# test results
test_indices = np.array([[1,0],[0,1],[1,1]],dtype=np.int32)
feed_dict = {input_indices:test_indices}
print sess.run(output,feed_dict=feed_dict)
