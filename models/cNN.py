
'''

Project: Baseline model for sequence landscape prediction, using convolutional neural network
Class(s): (none) 
Function: Organizes main pipeline execution and testing 

Author: Patrick V. Holec
Date Created: 5/10/2017
Date Updated: 5/10/2017

This is for actual data testing

'''

'''
BuildNetwork: Main neural network architecture generator
'''

# standard libaries
import math
import time
import random

# nonstandard libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import load_data

'''
Test Function (runs if called as script)
'''

def main():
    data_settings = {
                     'num_epochs':100,'learning_rate':0.001,
                     'data_augment':False,'data_normalization':True,
                     'data_silent':True
                    }
    

    model = BuildModel('A12',data_settings)
    model.data_format()
    model.network_initialization()
    model.train()
    guesses = model.predict()
    visualize(guesses,model.test_labels)
    
'''
Factory Methods
'''

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
def conv2d(x, W, stride=1):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

# max pool function (check if lists are passed in for movement)
def max_pool(x, stride=2, filter_size=2, padding='SAME'):
    if not(type(stride) == list and type(filter_size) == list):
        filter_size,stride = [1, filter_size, filter_size, 1],[1, stride, stride, 1]
    return tf.nn.max_pool(x, ksize=filter_size,strides=stride, padding=padding)

# TODO: probably not used?
def cross_entropy(y, y_real, W1=None,W2=None,W1fc=None,W2fc=None,modifications=['NONE'],loss_type='NONE',loss_coeff=0):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_real))

# activation_fn
def activation_fn(inp,fn=None,dropout=1.0):
    if fn == 'relu': return tf.nn.dropout(tf.nn.relu(inp), dropout)
    elif fn == 'relu6': return tf.nn.dropout(tf.nn.relu6(inp), dropout)
    elif fn == 'sigmoid': return tf.nn.dropout(tf.nn.sigmoid(inp), dropout)
    elif fn == 'tanh': return tf.nn.dropout(tf.nn.tanh(inp), dropout)
    elif fn == 'softplus': return tf.nn.dropout(tf.nn.softplus(inp), dropout)
    elif type(fn) == str: print '{} function not recognized, using default (NONE)...'.format(fn)
    return tf.nn.dropout(inp, dropout)

def network_loss(y,y_real,W,params):
    # base loss
    if params['loss_type'] == 'l1': loss = params['loss_magnitude']*tf.reduce_sum(tf.abs(tf.subtract(y,y_real)))
    if params['loss_type'] == 'l2': loss = params['loss_magnitude']*tf.nn.l2_loss(tf.subtract(y,y_real))
    else: loss = 0
    # regularization loss
    if params['reg_type'] == 'l1': loss += params['reg_magnitude']*tf.reduce_sum([tf.reduce_sum(tf.abs(w)) for w in W])
    if params['reg_type'] == 'l2': loss += params['reg_magnitude']*tf.reduce_sum([tf.nn.l2_loss(w) for w in W])
    return loss


'''
Main Method
'''

class BuildModel:
    
    def default_model(self):
        # basically every parameter defined in one dictionary
        default_params = {#'data_augment':False,
                         'learning_rate':0.01,
                         'data_normalization': False,
                         'silent': False,
                         'test_fraction': 0.1,
                         'batch_size':100,
                         'num_epochs':50,
                         'loss_coeff':0.01,
                         'learning_rate':0.01,
                         'data_label':'test',
                         # overall network parameters
                         'cnn_layers':1, # includes conv & pool
                         'fc_layers':2,
                         # convolutional parameters
                         'conv_stride':(1,1), # not sure if we can handle others
                         'conv_filter_width':(2,4),
                         'conv_depth':(2,4),
                         'conv_padding':('SAME','SAME'),
                         # max pool parameters
                         'pool_stride':(2,2),
                         'pool_filter_width':(3,3),
                         'pool_padding':('SAME','SAME'),
                         # fully connected parameters
                         'fc_depth':(16,1),
                         'fc_fn':('sigmoid','sigmoid','linear'),
                         'fc_dropout':(0.5,1.0),
                         # loss parameters
                         'loss_type':'l2',
                         'loss_magnitude':1.0,
                         # regularization parameters
                         'reg_type':'l2',
                         'reg_magnitude':0.01,
                         }
        
        self.model_parameters = default_params.keys()

        # apply all changes
        self.update_model(default_params)

    # use a dictionary to update class attributes
    def update_model(self,params={}):
        # makes params dictionary onto class attributes
        for key, value in params.items():
            setattr(self, key, value)
        
        # checks for adequete coverage
        assert len(self.conv_stride) >= self.cnn_layers,'Entries in strides less than number of conv. layers.'
        assert len(self.conv_filter_width) >= self.cnn_layers,'Entries in filter width less than number of conv. layers.'
        assert len(self.conv_depth) >= self.cnn_layers,'Entries in depth less than number of conv. layers.'
        assert len(self.conv_padding) >= self.cnn_layers,'Entries in padding less than number of conv. layers.'
        assert len(self.pool_stride) >= self.cnn_layers,'Entries in pool stride less than number of pool layers.'
        assert len(self.pool_filter_width) >= self.cnn_layers,'Entries in filter width less than number of pool layers.'
        assert len(self.pool_padding) >= self.cnn_layers,'Entries in padding less than number of pool layers.'
        assert len(self.fc_depth) >= self.fc_layers,'Entries in depth less than number of fc layers.'
        
    # model initialization
    def __init__(self,params = {}):
        
        # set all default parameters
        self.default_model()
        tf.reset_default_graph()
        
        # check to see if there is an update
        if params:
            print 'Updating model with new parameters:'
            for key,value in params.iteritems(): print '  - {}: {}'.format(key,value)
            self.update_model(params)
        
        print 'Initializing neural network data acquisition...'
        
        # load data parameters
        #data = load_data.LoadData(self.data_label) # OLD
        #self.__dict__.update(data.params)
        #self.all_data_sw = data.data_array_sw # load all variables
        #self.all_labels = data.label_array
        self.load_data()
        
        # 
        #self.sw_dim = self.all_data_sw.shape # save dimensions of original data
 
        # verified reduction of dimension (flatten) and merging
        #self.all_data = np.reshape(self.all_data_sw,
        #                                  (self.sw_dim[0],self.sw_dim[1]*self.sw_dim[2]))
         
        # update on model parameters
        if not self.silent:
            print '*** System Parameters ***'
            print '  - Sequence length:',self.length
            print '  - AA count:',self.aa_count
            print '  - Data shape:',self.all_data.shape
        
        print 'Finished acquisition!'
        
        # Create GPU configuration
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Start tensorflow engine
        print 'Initializing variables...'
        self.sess = tf.Session(config=config)
        
    def load_data(self):

        # open file and store lines
        with open('./data/{}.txt'.format(self.data_label),'rb') as txtfile:
            reader = txtfile.readlines()
            raw_seqs = [r.split(',')[0] for r in reader] 
            raw_labels = [float(r.split(',')[1]) for r in reader] 

        assert len(raw_seqs) == len(raw_labels), 'Sequence count not same as label count'

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
        self.all_data = np.zeros((len(raw_seqs),self.aa_count,self.length),np.int)

        # create all data
        for i,sample in enumerate(raw_seqs):
            for j,char in enumerate(sample):
                try: self.all_data[i,self.characters.index(char),j] = 1
                except ValueError: # this should never occur
                    raw_input('ERROR:' + self.characters + ' - ' + char)

        if not self.silent: print 'Finished generating data!'

    """ Formats data prior to model execution (separate into train and test sets) """ 
    def fold_data(self,params = {}):
        # check to see if there is an update
        if params:
            print 'Updating model with new parameters:'
            for key,value in params.iteritems(): print '  - {}: {}'.format(key,value)
            self.update_model(params)
            
        print 'Starting data formatting...'
        
        self.fold_total = int(1/self.test_fraction)
        
        # randomize data order
        self.order = np.arange(0,self.sequence_count)
        self.limits = [i*int(self.test_fraction*self.sequence_count) 
                for i in xrange(self.fold_total+1)]
        np.array(random.shuffle(self.order))

        print '{} folds generated, fold {} arbitrary picked.'.format(self.fold_total,1)
        
        self.fold_pick(1)
        
        # normalize label energy
        if self.data_normalization == True:
            self.all_labels = np.reshape(np.array([(self.all_labels - min(self.all_labels))/
                                                  (max(self.all_labels)-min(self.all_labels))]),(len(self.all_labels),1))
        
        # alternative normalization
        self.all_labels = np.reshape(np.array(self.all_labels),(len(self.all_labels),1))
        
        print 'Finished formatting!'

        
    def fold_pick(self,fold_index):
        # define fold index
        lim = fold_index%5
        o = self.order[np.arange(self.limits[lim],self.limits[lim+1])]
        o_not = np.delete(self.order,np.arange(self.limits[lim],self.limits[lim+1]),0)
        
        # split data into training and testing        
        self.train_data = self.all_data[o_not,:,:]
        self.test_data = self.all_data[o,:,:]
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
            self.train_data = self.all_data[order,:,:]
            self.test_data = self.all_data[order_not,:,:]
            self.train_labels = self.all_labels[order,:]
            self.test_labels = self.all_labels[order_not,:]
        
        print 'Data specified by user!'
        
        
    def network_initialization(self):
        
        print 'Building model...'
        
        # create placeholder variables, and format data
        self.train_x = tf.placeholder(tf.float32, shape=(None, self.aa_count,self.length)) 
        self.train_y = tf.placeholder(tf.float32, shape=(None, 1)) # full energy input
        layers = [tf.reshape(self.train_x, [-1, self.aa_count, self.length, 1])]
        
                  
        ## CONVOLUTIONAL LAYER (CNN) GENERATOR ##
        # temporary variables for non-symmetry
        width,depth = list(self.conv_filter_width),[1] + list(self.conv_depth)

        # create weight variables
        W = [weight_variable([self.aa_count,width[i],depth[i],depth[i+1]]) for i in xrange(self.cnn_layers)]        

        # iterate through conv./pool layers
        for i in xrange(self.cnn_layers):
            layers.append(conv2d(layers[-1],W[i],stride=self.conv_stride[i])) # build conv layer
            print 'Layer {} conv. output: {}'.format(i+1,layers[-1].shape)
            
            #window = [1,int(layers[-1].shape[1]),self.pool_stride[i],1]
            filter_size = [1,int(layers[-1].shape[1]),self.pool_filter_width[i],1]
            stride = [1,1,self.pool_stride[i],1]
            
            layers.append(max_pool(layers[-1],stride = stride, filter_size=filter_size))
            print 'Layer {} pool output: {}'.format(i+1,layers[-1].shape)
            
                  
        ## FLATTEN LAYER ##
        # build dimensions to be sure we get this right
        numel = int(layers[-1].shape[1]*layers[-1].shape[2]*layers[-1].shape[3])
        layers.append(tf.reshape(layers[-1],[-1,numel]))
        print 'Flattened layer shape:',layers[-1].shape
            
                  
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
            print 'Layer {} fc output: {}'.format(i+1,layers[-1].shape)
        
        ## TRAINING METHODS
        self.y_out = layers[-1]
        self.loss = network_loss(self.y_out,self.train_y,W+W_fc,self.__dict__)
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        
        print 'Finished!'
        
        
    def train(self):
        # start timer
        start = time.time()
        
        # training via iterative epochs
        batches_per_epoch = int(len(self.train_data)/self.batch_size)
        num_steps = int(self.num_epochs * batches_per_epoch)
        
        print 'Batchs per epoch - {} / Number of steps - {}'.format(batches_per_epoch,num_steps)
        
        init = tf.global_variables_initializer()
        
        self.sess.run(init)
        
        epoch_loss = 0
        epoch_acc = 0
        
        for step in xrange(num_steps):
            offset = (step * self.batch_size) % (self.train_data.shape[0] - self.batch_size)

            batch_x = self.train_data[offset:(offset + self.batch_size), :]
            batch_y = np.reshape(self.train_labels[offset:(offset + self.batch_size)],(self.batch_size,1))

            feed_dict = {self.train_x: batch_x, self.train_y: batch_y}
            
            _, batch_loss = self.sess.run([self.train_step, self.loss],feed_dict=feed_dict)
            #print 'Batch loss:',batch_loss 
            epoch_loss += batch_loss
                        
            if (step % batches_per_epoch == 0):

                epoch_loss /= batches_per_epoch*self.batch_size

                feed_dict = {self.train_x: self.test_data, self.train_y: self.test_labels}
                batch_loss_validation = self.sess.run(self.loss,feed_dict=feed_dict)

                print 'Avg batch loss at step %d: %f' % (step, epoch_loss)
                print 'Batch loss ({})  /  Validation loss ({})'.format(batch_loss,batch_loss_validation)

                epoch_loss = 0
                # randomize input data
                seed = np.random.randint(1,1000000) # pick a random seed
                np.random.seed(seed) # set identical seeds
                np.random.shuffle(self.train_data) # shuffle data in place
                np.random.seed(seed) # set identical seeds
                np.random.shuffle(self.train_labels) # shuffle data in place

                #together = np.concatenate((self.train_data,self.train_labels),axis=1)
                #np.random.shuffle(together)
                #self.train_data = together[:,:-1]
                #self.train_labels = np.reshape(together[:,-1],(self.train_labels.shape[0],1)) 
                 
        print 'Training time: ', time.time() - start
        print 'Finished!'
        
        
    def save_model(self,silent=False):

        """
        Attempts to store all useful information about the trained model in a log file, which will
        be unique from any any other model file
        """

        # tries to find a model log path that does not exist
        for i in xrange(10001,100001):
            fname = './logs/cnn_model_{}.p'.format(i+1)
            if not os.path.exists(fname): break
        if not silent: print 'Creating log file as: {}'.format(fname) # alert user
        
        # get some default parameters from the model
        for p in self.model_parameters: model_dict[p] = self.__dict__[p]
        
        # TODO: evaluate the weight matrices, and store in dict
        # labels = ['W_sw','W_pw','W_fc','b_fc']
        # for l,w in zip(labels,self.weights):
        #     weight_matrix = self.sess.run(w,feed_dict={})
        #     model_dict[l] = weight_matrix
            
        # pickle file
        with open(fname, 'w') as f:
            pickle.dump(model_dict,f)
        
        return i+1 # return index of model file for later reference


    def predict(self,data=[]):
        # if no inputs are specified, use the defaults
        if len(data) == 0:
            data = self.test_data
        
        # create distance matrix
        assert self.train_data.shape[1:] == data.shape[1:],'Test and train data not the same shape (axis 1+).'
        
        # create guesses for each string in data
        guesses = []

        feed_dict = {self.train_x: data}
        guesses = self.sess.run(self.y_out,feed_dict=feed_dict)
        
        print 'Finished guessing!'

        return guesses

    

    
    
'''
Catch if called as script
'''
if __name__ == '__main__':
    main()
        
        
        
        
    
