

import os

import tensorflow as tf
import numpy as np



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sess = tf.InteractiveSession()

aa_count = 4
length = 5


def matmul_tensor(tensor):
    pass

def split_tensor(tensor,axis=1):
    return tf.split(tensor,[1 for i in xrange(tensor.get_shape()[axis])],axis)

# fake tensor
#rx = tf.constant([[1,0,0,0,0],[0,1,0,0,0],[0,0,0,1,0],[0,0,1,0,1]])
rx = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,0,1,0],[0,0,1,0,1]])
data = np.stack([rx for i in xrange(8)])

# Load data (this is njw straight to _sw
inp = tf.placeholder(tf.int64, shape=(None,4,5))
aa_count = inp.get_shape()[1]
length = inp.get_shape()[2]
print 'Count:',aa_count
print 'Length:',length

# create dense representation of one-hot 
x = tf.argmax(inp,axis=1)
print x.get_shape()

elems = np.array([1,2,3,4,5,6])
elems = np.array([[1,2,3,4,5,6],[3,4,5,6,7,8]])

test = tf.scan(lambda a,x: x, elems)
print 'Test:',sess.run(test)

# build full tensor system 
pairs = [(i,j) for j in xrange(i+1,length) for i in xrange(0,length)]
pw = tf.zeros([inp.get_shape()[0],len(pairs),aa_count,aa_count],tf.bool)
print 'Pairwise shape:',pw.get_shape()

indices = [[[a,b,x[a][p[0]],x[a][p[1]]] for b,p in enumerate(pairs)] for a,vec in enumerate(x)]

indices = tf.constant([[4], [3], [1], [7]])
updates = tf.constant([9, 10, 11, 12])
shape = tf.constant([8])
scatter = tf.scatter_nd(indices, updates, shape)
print sess.run(scatter)


'''
locations = tf.where(tf.equal(inp, 1))



x = tf.argmax(inp,axis=1) 

feed_dict = {inp:data}
print sess.run(x,feed_dict=feed_dict)

x = split_tensor(inp)

# change into repmated vectors
#x = [tf.tile(i,[1,int(i.get_shape()[2]),1]) for i in x]


#[matmul_tensor(x[i],x[j]) for j in xrange(i+1,len(x)) for i in xrange(0,len(x)-1)]

#print data.dtype
#print data.shape



print type(x)
print x

#print sess.run(x,feed_dict=feed_dict)

'''



'''
ax = tf.constant([[0,1,0,0,0]])
bx = tf.constant([[0,0,1,0,0]])

a = tf.stack([ax for i in xrange(8)])
b = tf.stack([bx for i in xrange(8)])

rx = tf.constant([[1,0,0,0,0],[0,1,0,0,0],[0,0,0,1,0],[0,0,1,0,1]])
x = tf.stack([rx for i in xrange(8)])

print x.get_shape()

h = tf.scan(lambda a, x: tf.matmul(x, U), embed)

mat = tf.stack([slice_product(x[:,:,i],x[:,j],length)
    for i in xrange(0,length-1) 
    for j in xrange(i+1,length)],0)
'''


