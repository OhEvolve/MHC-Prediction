

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
from common_methods import *

# library modifications


def main():

    model = BuildModel()
    model.network_initialization()
    

'''
Main Class Method
'''


class BuildModel:
    
    def default_model(self):
        # basically every parameter defined in one dictionary
        default_params = {
                         'data_augment':False,
                         'data_normalization': False,
                         'silent': False,
                         'test_fraction': 0.1,
                         'batch_size':1000,
                         'num_epochs':50,
                         'learning_rate':1,
                         'data_label':'A12',
                         # overall network parameters
                         'fc_layers':2,
                         'sw_pw_ratio':0.5, # relative importance of sw branch relative to pw branch
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
        self.all_data = np.zeros((len(raw_seqs),self.length),np.int)

        # create all data
        for i,sample in enumerate(raw_seqs):
            for j,char in enumerate(sample):
                try: self.all_data[i,j] = self.characters.index(char)
                except ValueError: # this should never occur
                    raw_input('ERROR:' + self.characters + ' - ' + char)

        if not self.silent: print 'Finished generating data!'

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
        #config = tf.ConfigProto() # use this to set to GPU
        config = tf.ConfigProto(device_count = {'GPU':0}) # use this to set to CPU

        config.log_device_placement = False # super verbose mode
        #config.gpu_options.allow_growth = True
        #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        # Start tensorflow engine
        print 'Initializing variables...'
        self.sess = tf.Session(config=config)
        
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

        if not self.silent: print 'Building model...'
        
        # initialize filters
        # TODO: convert all to indexable matrix
        W_sw = [[normal_variable((self.aa_count,)) for i in xrange(self.length)] for j in xrange(self.sw_depth)]
        W_pw = [[normal_variable((self.aa_count,self.aa_count)) for i in xrange(self.pair_count)] for j in xrange(self.pw_depth)]
    
        # Create pair indices for later
        pairs = [(i,j) for i in xrange(0,self.length-1) for j in xrange(i+1,self.length)] # create list of pairs

        # TODO: Consider modifications that use less dense data types
        # Load data (this is now straight to _sw
        self.train_x = tf.placeholder(tf.int64, shape=(None, self.length)) # full vector input
        self.train_y = tf.placeholder(tf.float32, shape=(None, 1)) # full energy input

        # create sw/pw indices
        input_sw = split_tensor(self.train_x,axis=1)
        input_pw = [join_tensor(input_sw[i],input_sw[j]) for i,j in pairs]
        
        # create SW indexed system
        output_sw = tf.stack([[self.sw_pw_ratio*tf.gather_nd(w,i) for w,i in zip(W,input_sw)] for W in W_sw],axis=1) # really hard to explain why there is a squared

        # create PW indexed system
        output_pw = tf.stack([[(1.-self.sw_pw_ratio)*tf.gather_nd(w,i) for w,i in zip(W,input_pw)] for W in W_pw],axis=1)

        # reshape to expected layer type
        output_sw_flat = tf.contrib.layers.flatten(tf.transpose(output_sw,[2,1,0]))
        output_pw_flat = tf.contrib.layers.flatten(tf.transpose(output_pw,[2,1,0]))
        layers = [tf.concat([output_sw_flat,output_pw_flat],1)] 

        ## FULLY CONNECTED LAYER (FC) GENERATOR ##
        # temporary variables for non-symmetry
        sw_size,pw_size = self.length*self.sw_depth,self.pair_count*self.pw_depth
        depth = [sw_size + pw_size] + list(self.fc_depth)

        # create weight/bias variables
        W_fc = [normal_variable([depth[i],depth[i+1]]) for i in xrange(self.fc_layers)]  
        b_fc = [bias_variable([depth[i+1]]) for i in xrange(self.fc_layers)]
        #W_fc = [ones_constant([depth[i],depth[i+1]]) for i in xrange(self.fc_layers)]  
        #b_fc = [zeros_constant([depth[i+1]]) for i in xrange(self.fc_layers)]
                              
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
    
        self.set_learning_fn()
        #if params['loss_type'] == 'l1': loss = params['loss_magnitude']*(val/tf.cast(tf.size(y),tf.float32))*tf.reduce_sum(tf.abs(tf.subtract(y,y_real)))
        
        # (re)-initialize all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        if not self.silent: print 'Finished!'        

    def set_learning_fn(self,learning_rate=None):

        if learning_rate == None: # set learning rate to default if not specified
            learning_rate = self.learning_rate 

        ### Choose your learning method ###
        #self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss) # why does this print?
        #self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss) # why does this print?
        self.train_step = tf.train.ProximalGradientDescentOptimizer(learning_rate).minimize(self.loss) # why does this print?
        
    def train(self):
    
        # start timer
        start = time.time()

        # training via iterative epochs
        
        batches_per_epoch = int(len(self.train_data)/self.batch_size)
        num_steps = int(self.num_epochs * batches_per_epoch)
        
        if not self.silent: print 'Batchs per epoch - {} / Number of steps - {}'.format(batches_per_epoch,num_steps)

        step_index,step_loss = [],[]
        epoch_loss,step,learning_rate_mod = 0,0,1.0
        finished = False

        while not finished: 
            offset = (step * self.batch_size) % (self.train_data.shape[0] - self.batch_size)

            # split data
            batch_x = self.train_data[offset:(offset + self.batch_size), :]
            batch_y = np.reshape(self.train_labels[offset:(offset + self.batch_size)],(self.batch_size,1))

            # train and log batch loss
            feed_dict = {self.train_x: batch_x, self.train_y: batch_y}
            _,batch_loss = self.sess.run([self.train_step,self.loss],feed_dict=feed_dict)
            epoch_loss += batch_loss

            if (step % batches_per_epoch == batches_per_epoch - 1):
                
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
            # add one to step 
            step += 1

            # create early exist conditions
            if step >= num_steps: # check if at threshold of learning steps
                finished = True
            if np.isnan(batch_loss): # check if training has spiraled into NaN space
                step = 0
                learning_rate_mod *= 0.5
                init = tf.global_variables_initializer()
                self.sess.run(init)
                self.set_learning_fn(self.learning_rate*learning_rate_mod)
                print 'Lowering learning rate and restarting...'

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
        else:
            pass
        
        # create distance matrix
        assert self.train_data.shape[1:] == data.shape[1:],'Test and train data not the same shape (axis 1+).'

        # batch iterate over test data
        guesses = np.zeros((len(data),1))
        
        for i in xrange(0,len(data),self.batch_size):
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
