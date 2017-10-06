

'''
Project: Neural Network for MHC Peptide Prediction
Class(s): BuildNetwork
Function: Generates specified neural network architecture (data agnostic)

Author: Patrick V. Holec
Date Created: 2/2/2017
Date Updated: 5/18/2017
'''




# standard libaries
import math
import time
import random
import os
import pickle

# nonstandard libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# homegrown libraries
import load_data

# library modifications
random.seed(42)
tf.set_random_seed(42)




def main():

    data_settings = {
                     'num_epochs':100,'learning_rate':0.001,
                     'data_augment':False,'data_normalization':True,
                     'fc_fn':('linear','linear'),
                     'test_fraction': 0.1,
                     'silent':False
                    }
    

    model = BuildModel('A12',data_settings)
    model.data_format()
    model.network_initialization()
    model.train()
    guesses = model.predict()
    #vis.comparison(guesses,model.test_labels)
    vis.auroc(list(guesses),model.test_labels)
    
'''
Factory Methods
'''

def matmul_tensor(tensor):
    return [tf.map_fn(lambda x: tf.matmul(tf.expand_dims(x[:,0],1),tf.expand_dims(x[:,1],0)),t) for t in tensor]    

def split_tensor(tensor,axis=2):
    return tf.split(tensor,[1 for i in xrange(tensor.get_shape()[axis])],axis)

def join_tensor(a,b,axis=2):
    return tf.squeeze(tf.stack([a,b],axis=axis),axis=3)

# given any number of dicts, shallow copy and merge into a new dict, precedence goes to key value pairs in latter dicts.
def merge_dicts(*dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

# create a weight variable
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# create a bias variable
def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convolution filter function 
def conv2d(x, W, stride=1, padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

# max pool function (check if lists are passed in for movement)
def max_pool(x, stride=2, filter_size=2, padding='SAME'):
    if not(type(stride) == list and type(filter_size) == list):
        filter_size,stride = [1, filter_size, filter_size, 1],[1, stride, stride, 1]
    return tf.nn.max_pool(x, ksize=filter_size,strides=stride, padding=padding)

# NOTE: probably not used?
def cross_entropy(y, y_real, W1=None,W2=None,W1fc=None,W2fc=None,modifications=['NONE'],loss_type='NONE',loss_coeff=0):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_real))

# activation_fn
def activation_fn(inp,fn=None,dropout=1.0):
    if fn == 'relu': return tf.nn.dropout(tf.nn.relu(inp), dropout)
    elif fn == 'relu6': return tf.nn.dropout(tf.nn.relu6(inp), dropout)
    elif fn == 'sigmoid': return tf.nn.dropout(tf.nn.sigmoid(inp), dropout)
    elif fn == 'tanh': return tf.nn.dropout(tf.nn.tanh(inp), dropout)
    elif fn == 'softplus': return tf.nn.dropout(tf.nn.softplus(inp), dropout)
    elif fn == 'linear': return tf.nn.dropout(inp,dropout)
    elif type(fn) == str: print '{} function not recognized, using default (NONE)...'.format(fn)
    return tf.nn.dropout(inp, dropout)

def network_loss(y,y_real,W,params):
    val = tf.constant(1.0) # make a quick flaot32 variable for shorter calls
    # base loss
    if params['loss_type'] == 'l1': loss = params['loss_magnitude']*(val/tf.cast(tf.size(y),tf.float32))*tf.reduce_mean(tf.abs(tf.subtract(y,y_real)))
    if params['loss_type'] == 'l2': loss = params['loss_magnitude']*(val/tf.cast(tf.size(y),tf.float32))*tf.nn.l2_loss(tf.subtract(y,y_real))
    else: loss = 0
    # regularization loss
    if params['reg_type'] == 'l1': loss += params['reg_magnitude']*tf.reduce_sum([(val/tf.cast(tf.size(w),tf.float32))*tf.reduce_sum(tf.abs(w)) for w in W])
    if params['reg_type'] == 'l2': loss += params['reg_magnitude']*tf.reduce_sum([(val/tf.cast(tf.size(w),tf.float32))*tf.nn.l2_loss(w) for w in W])
    return loss


'''
Main Class Method
'''

# DONE: Make l1,l2 loss based on averages
# DONE: Check to see if pw_depth is working
# DONE: Reset graph after __init__
# DONE: Parameter count tool
# 


class BuildModel:
    
    def default_model(self):
        # basically every parameter defined in one dictionary
        default_params = {
                         'data_augment':False,
                         'learning_rate':0.01,
                         'data_normalization': False,
                         'silent': False,
                         'test_fraction': 0.1,
                         'batch_size':100,
                         'num_epochs':50,
                         'learning_rate':0.1,
                         'data_label':'A12',
                         # overall network parameters
                         'fc_layers':2,
                         'sw_pw_ratio':0.5, # relative importance of sw branch relative to pw branch (range -> [0,1])
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
                         'reg_magnitude':0.01
                         # logging parameters
                         }
        
        self.model_parameters = default_params.keys()
        
        # apply all changes
        for key, value in default_params.items():
            setattr(self, key, value)

    # use a dictionary to update class attributes
    def update_model(self,params={}):
        # updates parameters
        for key, value in params.items():
            setattr(self, key, value)

        # makes params dictionary onto class attributes
        if not self.silent and params: # prints log of model changes
            print 'Updating model with new parameters:'
            for key,value in params.iteritems(): print '  - {}: {}'.format(key,value)
        
        # checks for adequete coverage
        assert len(self.fc_depth) >= self.fc_layers,'Entries in depth less than number of fc layers.'

    def count_trainable_parameters(self):
        """
        Return the total number of trainable parameters in the model, 
        which is useful for system model matching
        """
        return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) 
        
    def load_data(self):

        # open file and store lines
        with open('./data/{}.txt'.format(self.data_label),'rb') as txtfile:
            reader = txtfile.readlines()
            raw_seqs = [r.split(',')[0] for r in reader] 
            raw_labels = [float(r.split(',')[1]) for r in reader] 

        # get parameters out of data
        chars = ''.join(sorted(set(''.join(raw_seqs))))
        self.length = len(raw_seqs[0]) 
        self.aa_count = len(chars)
        self.characters = chars 
        self.sequence_count = len(raw_seqs)
        self.pair_count = (self.length*(self.length-1))/2        

        if not self.silent:
            print 'Loaded data with following parameters:'
            print ' > File location:','./data/{}.txt'.format(self.data_label)
            print ' > Peptide length:',self.length
            print ' > AA count:',self.aa_count
            print ' > Character list:',self.characters
            print ' > Samples:',self.sequence_count

        # always make label array
        self.all_labels = np.reshape(np.array(raw_labels),(len(raw_labels),1,1))

        # one-hot encoding (sitewise) -> there is no more pw split
        self.all_data_sw = np.zeros((
            len(raw_seqs),
            self.aa_count,
            self.length),np.bool)
        for i,sample in enumerate(raw_seqs):
            for j,char in enumerate(sample):
                try: self.all_data_sw[i,self.characters.index(char),j] = 1
                except ValueError: # this should never occur
                    raw_input('ERROR:' + self.characters + ' - ' + char)

        # save the information that matters
        sw_dim = self.all_data_sw.shape
        self.all_data = np.reshape(self.all_data_sw,(sw_dim[0],sw_dim[1]*sw_dim[2]))


    # model initialization
    def __init__(self,params = {}):

        # print version numbers that matter
        print 'Loaded Tensorflow library {}.'.format(tf.__version__)
        
        # set all default parameters and reset graph
        self.default_model()
        tf.reset_default_graph()
        
        # modify Tensorflow's verbosity
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        # updates parameters
        self.update_model(params)
        
        print 'Initializing neural network data acquisition...'        
        
        # loads data based on label, creates all_data
        self.load_data()
        
         # update on model parameters
        if not self.silent:
            print '*** System Parameters ***'
            print '  - Sequence length:',self.length
            print '  - AA count:',self.aa_count       
            
        if not self.silent: print 'Finished acquisition!'
        
        # Create GPU configuration
        config = tf.ConfigProto()
        config.log_device_placement = False # super verbose mode
        config.gpu_options.allow_growth = True
        #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        # Start tensorflow engine
        print 'Initializing variables...'
        self.sess = tf.Session(config=config)
        self.reset_all_variables = tf.global_variables_initializer()
        
    def fold_data(self,params = {}):
        # updates parameters
        self.update_model(params)
            
        print 'Starting data formatting...'
        
        self.fold_total = int(1/self.test_fraction)
        
        # randomize data order
        self.order = np.arange(0,self.all_data.shape[0])
        self.limits = [i*int(self.test_fraction*self.all_data.shape[0]) for i in xrange(self.fold_total+1)]
        np.array(random.shuffle(self.order))

        if not self.silent: print '{} data folds generated.'.format(self.fold_total)
                
        # normalize label energy
        if self.data_normalization == True:
            self.all_labels = np.reshape(np.array([(self.all_labels - min(self.all_labels))/
                                                  (max(self.all_labels)-min(self.all_labels))]),(len(self.all_labels),1))
        
        # alternative normalization
        self.all_labels = np.reshape(np.array(self.all_labels),(len(self.all_labels),1))
        
        self.fold_pick(1)
        
        if not self.silent: print 'Finished formatting!'

        
    def fold_pick(self,fold_index):
        # define fold index
        lim = fold_index%self.fold_total
        
        if self.fold_total == 1:
            self.train_data,self.test_data = self.all_data,np.zeros((0,self.all_data.shape[1]))
            self.train_labels,self.test_labels = self.all_labels,np.zeros((0,self.all_labels.shape[1]))
            o,o_not = None,None
        else:
            o = self.order[np.arange(self.limits[lim],self.limits[lim+1])]
            o_not = np.delete(self.order,np.arange(self.limits[lim],self.limits[lim+1]),0)
        
            # split data into training and testing        
            self.train_data = self.all_data[o_not,:]
            self.test_data = self.all_data[o,:]
            self.train_labels = self.all_labels[o_not,:]
            self.test_labels = self.all_labels[o,:]

        if not self.silent:
            print 'Train data:',self.train_data.shape
            print 'Test data:',self.test_data.shape
            print 'Train labels:',self.train_labels.shape
            print 'Test labels:',self.test_labels.shape
        
        return o,o_not
    
    
    def set_fold(self,order = [],order_not = []):
        if len(order) > 0 and len(order_not) > 0:
            self.train_data = self.all_data[order,:]
            self.test_data = self.all_data[order_not,:]
            self.train_labels = self.all_labels[order,:]
            self.test_labels = self.all_labels[order_not,:]
        
        if not self.silent: print 'Data specified by user!'
        
    def set_training(self,data,labels):
        self.train_data = data
        self.train_labels = labels

    def set_testing(self,data,labels):
        self.test_data = data
        self.test_labels = labels
        
    def network_initialization(self):

        #tf.reset_default_graph() # This breaks things last time I checked
        
        if not self.silent: print 'Building model...'
        
        # initialize filters
        W_sw = weight_variable((self.aa_count,self.length,1,self.sw_depth))
        W_pw = weight_variable((self.aa_count,self.aa_count,1,self.pair_count,self.pw_depth))
    
        # TODO: Consider modifications that use less dense data types
        # Load data (this is now straight to _sw
        with tf.device('/cpu:0'):
            self.train_x = tf.placeholder(tf.float32, shape=(None, self.aa_count*self.length)) # full vector input
            self.train_y = tf.placeholder(tf.float32, shape=(None, 1)) # full energy input

            # Sitewise build
            x_image_sw = tf.reshape(self.train_x, [-1, self.aa_count, self.length, 1])

            # Create pair indices for later
            pairs = [(i,j) for i in xrange(0,self.length-1) for j in xrange(i+1,self.length)] # create list of pairs

            # Pairwise build 
            input_split = split_tensor(tf.squeeze(x_image_sw,axis=3),axis=2) # makes [(?,a),(?,a)...]
            input_paired = [join_tensor(input_split[i],input_split[j]) for i,j in pairs] # makes [(?,a,2),(?,a,2)...]
            input_matmul = matmul_tensor(input_paired) # makes [(?,a,a),(?,a,a)...]
            x_image_pw = tf.stack(input_matmul,axis=1) # makes (?,a,a,l*(l-1)/2)
            x_image_pw = tf.transpose(x_image_pw, [0, 2, 3, 1]) # makes (?,a,l*(l-1)/2,a)       
        
        # creating sitewise convolution
        conv_sw_array = [float(self.sw_pw_ratio)*conv2d(a,b,stride=1,padding='VALID') 
                         for W_sw_layer in tf.split(W_sw,[1 for i in xrange(self.sw_depth)],3)
                         for a,b in zip(tf.split(x_image_sw,[1 for i in xrange(self.length)],2),
                                   tf.split(W_sw_layer,[1 for i in xrange(self.length)],1))]
        conv_sw = tf.concat(conv_sw_array,1)   

        if not self.silent: print 'Conv. SW shape: {}'.format(conv_sw.shape)

        # creating pairwise convolution
        conv_pw_array = [(1.-self.sw_pw_ratio)*conv2d(a,b,stride=1,padding='VALID') 
                         for W_pw_layer in tf.split(W_pw,[1 for i in xrange(self.pw_depth)],4) 
                         for a,b in zip(tf.split(x_image_pw,[1 for i in xrange(self.pair_count)],3),
                                   tf.split(tf.squeeze(W_pw_layer,[4]),[1 for i in xrange(self.pair_count)],3))]  
        conv_pw = tf.concat(conv_pw_array,1)        
    
        if not self.silent: print 'Conv. PW shape: {}'.format(conv_pw.shape)

        layers = [tf.concat([conv_sw,conv_pw],1)] # join layers

        # build dimensions to be sure we get this right        
        numel = int(layers[-1].shape[1]*layers[-1].shape[2]*layers[-1].shape[3])
        layers.append(tf.reshape(layers[-1],[-1,numel]))
        if not self.silent: print 'Flattened layer shape:',layers[-1].shape

        ## FULLY CONNECTED LAYER (FC) GENERATOR ##
        # temporary variables for non-symmetry
        depth = [numel] + list(self.fc_depth)

        # create weight/bias variables
        W_fc = [weight_variable([depth[i],depth[i+1]]) for i in xrange(self.fc_layers)]  
        b_fc = [bias_variable([depth[i+1]]) for i in xrange(self.fc_layers)]
                              
        # iterate through fc_layers layers
        for i in xrange(self.fc_layers):
            layers.append(activation_fn(tf.matmul(layers[-1],W_fc[i]) + b_fc[i],
                          fn=self.fc_fn[i],dropout=self.fc_dropout[i]))
            if not self.silent: 
                print 'Layer {} fc input: {}'.format(i+1,layers[-2].shape)
                print 'Layer {} fc output: {}'.format(i+1,layers[-1].shape)
        
        ## TRAINING METHODS
        self.y_out = layers[-1]
        self.loss = network_loss(self.y_out,self.train_y,W_fc,self.__dict__)
        
        # hook for evaluating all the trained weights
        self.weights = [W_sw,W_pw,W_fc,b_fc]
    
        #self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss) # why does this print?
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss) # why does this print?
        
        # (re)-initialize all variables
        self.sess.run(self.reset_all_variables)       

        if not self.silent: print 'Finished!'        

        
    def train(self):
        # start timer
        start = time.time()

        # training via iterative epochs
        
        batches_per_epoch = int(len(self.train_data)/self.batch_size)
        num_steps = int(self.num_epochs * batches_per_epoch)
        
        if not self.silent: print 'Batchs per epoch - {} / Number of steps - {}'.format(batches_per_epoch,num_steps)

        init = tf.global_variables_initializer()

        self.sess.run(init)

        step_index,step_loss = [],[]
        epoch_loss = 0

        for step in xrange(num_steps):
            offset = (step * self.batch_size) % (self.train_data.shape[0] - self.batch_size)

            batch_x = self.train_data[offset:(offset + self.batch_size), :]
            batch_y = np.reshape(self.train_labels[offset:(offset + self.batch_size)],(self.batch_size,1))

            feed_dict = {self.train_x: batch_x, self.train_y: batch_y}

            _, batch_loss = self.sess.run([self.train_step, self.loss],feed_dict=feed_dict)
            #_, batch_loss,y_out,W_sw,test,test2 = self.sess.run([self.train_step, self.loss, self.y_out,self.weights[0],self.test, self.test2],feed_dict=feed_dict)

            epoch_loss += batch_loss

            # HUD for user
            #print 'Batch loss:',batch_loss

            if (step % batches_per_epoch == 0):

                epoch_loss /= 0.01*batches_per_epoch*self.batch_size
                
                feed_dict = {self.train_x: self.test_data, self.train_y: self.test_labels}
                batch_loss_validation = self.sess.run(self.loss,feed_dict=feed_dict)
                batch_loss_validation /= 0.01*self.test_data.shape[0]

                step_index.append(step)
                step_loss.append((epoch_loss,batch_loss_validation))
                
                # Gives readout on model
                if not self.silent: 
                    print 'Step {}: Batch loss ({})  /  Validation loss ({})'.format(step,epoch_loss,batch_loss_validation)

                epoch_loss = 0
                
                # randomize input data
                seed = np.random.randint(1,1000000) # pick a random seed
                np.random.seed(seed) # set identical seeds
                np.random.shuffle(self.train_data) # shuffle data in place
                np.random.seed(seed) # set identical seeds
                np.random.shuffle(self.train_labels) # shuffle data in place


                '''
                together = np.concatenate((self.train_data,self.train_labels),axis=1)
                np.random.shuffle(together)
                self.train_data = together[:,:-1]
                self.train_labels = np.reshape(together[:,-1],(self.train_labels.shape[0],1)) # need to add dimension to data
                '''

        print '[FINAL] Epoch loss ({})  /  Validation loss ({}) / Training time ({} s)'.format(epoch_loss,batch_loss_validation,time.time() - start)

        if not self.silent: print 'Finished!'
        
        # stores logs of stepwise losses if needed later for saving model performance
        self.step_index,self.step_loss = step_index,step_loss 
        
    def save_model(self,silent=False):

        """
        Attempts to store all useful information about the trained model in a log file, which will
        be unique from any any other model file
        """

        # tries to find a model log path that does not exist
        for i in xrange(10001,100001):
            fname = './logs/spinn_model_{}.p'.format(i+1)
            if not os.path.exists(fname): break
        if not silent: print 'Creating log file as: {}'.format(fname) # alert user
        
        model_dict = {'step_index':self.step_index,'step_loss':self.step_loss}

        # get some default parameters from the model
        for p in self.model_parameters: model_dict[p] = self.__dict__[p]
        
        # evaluate the weight matrices, and store in dict
        labels = ['W_sw','W_pw','W_fc','b_fc']
        for l,w in zip(labels,self.weights):
            weight_matrix = self.sess.run(w,feed_dict={})
            model_dict[l] = weight_matrix
            
        # pickle file
        with open(fname, 'w') as f:
            pickle.dump(model_dict,f)
        
        return i+1 # return index of model file for later reference

    def predict(self,data=[]):
        # if no inputs are specified, use the defaults
        if len(data) == 0:
            print 'No data input to be predicted, using test data!'
            data = self.test_data
        
        # create distance matrix
        assert self.train_data.shape[1:] == data.shape[1:],'Test and train data not the same shape (axis 1+).'
        

        # batch iterate over test data
        guesses = np.zeros((len(data),1))
        
        for i in xrange(0,len(data),500):
            if not self.silent: print 'Starting batch prediction at index {}...'.format(i)
            feed_dict = {self.train_x: data[i:i+self.batch_size,:]}
            guesses[i:i+self.batch_size,:] = self.sess.run(self.y_out,feed_dict=feed_dict)
        
        if not self.silent: print 'Finished guessing!'

        return guesses

        
'''
Catch if called as script
'''
if __name__ == '__main__':
    main()
