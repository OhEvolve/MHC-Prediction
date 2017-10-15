

'''
Project: Neural Network for MHC Peptide Prediction
Class(s): BuildNetwork
Function: Generates specified neural network architecture (data agnostic)

Author: Patrick V. Holec
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
import default_parameters

# library modifications


def main():

    model = BuildModel()
    model.network_initialization()
    

'''
Main Class Method
'''


class BuildModel:
    
    def default_model(self):
        """ Pulls a default parameter list and applies parameters to model """
        # basically every parameter defined in one dictionary
        default_params = default_parameters.spinn() 
        self.model_parameters = default_params.keys()
        
        # apply all changes
        for key, value in default_params.items():
            setattr(self, key, value)

    # use a dictionary to update class attributes
    def update_model(self,params={}):
        """ Updates model attributes after passing a dictionary """
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
        
    # model initialization
    def __init__(self,params = {}):
        """ Initialize class, potentially with specified parameters """ 

        # print version numbers that matter
        print 'Loaded Tensorflow library {}.'.format(tf.__version__)
        
        # set all default parameters, potentially replace
        self.default_model()

        # clear TF graph, modify TF's verbosity
        self.clear_graph()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
    def clear_graph():
        """ Clear graph, be sure its actually clear """ 
        tf.reset_default_graph()

    def reset_variables():
        """ Resets variables on graph, using given random seed """ 
        tf.set_random_seed(self.seed)
        tf.global_variables_initializer()
    
    def network_initialization(self,params={}):
        """ Constructs new network given current parameters """

        # update model with potentially new values
        self.update_model(params)

        # Choose whether to use CPU or GPU, configure devices
        if 'tf_device' == 'CPU': 
            config = tf.ConfigProto(device_count = {'GPU':0}) # use this to set to CPU
        elif 'tf_device' == 'GPU': 
            config = tf.ConfigProto() # use this to set to GPU
            config.log_device_placement = False # super verbose mode
            config.gpu_options.allow_growth = True

        # clear TF graph
        self.clear_graph()

        # Start tensorflow engine
        print 'Initializing variables...'
        self.sess = tf.Session(config=config)

        if not self.silent: print 'Building model...'
        
        # initialize filters
        W_sw = [[normal_variable((self.aa_count,)) 
            for i in xrange(self.length)] for j in xrange(self.sw_depth)]
        W_pw = [[normal_variable((self.aa_count,self.aa_count)) 
            for i in xrange(self.pair_count)] for j in xrange(self.pw_depth)]
    
        # Create pair indices for later
        pairs = [(i,j) for i in xrange(0,self.length-1) 
                for j in xrange(i+1,self.length)] # create list of pairs

        # Load data (this is now straight to _sw
        self.train_x = tf.placeholder(tf.int64, shape=(None, self.length)) # full vector input
        self.train_y = tf.placeholder(tf.float32, shape=(None, 1)) # full energy input

        ## Sitewise/Pairwise LAYER (SP) GENERATOR ##
        # create sw/pw indices
        input_sw = split_tensor(self.train_x,axis=1)
        input_pw = [join_tensor(input_sw[i],input_sw[j]) for i,j in pairs]
        
        # create SW indexed system
        output_sw = tf.stack([[self.sw_pw_ratio*tf.gather_nd(w,i) 
            for w,i in zip(W,input_sw)] for W in W_sw],axis=1)

        # create PW indexed system
        output_pw = tf.stack([[(1.-self.sw_pw_ratio)*tf.gather_nd(w,i) 
            for w,i in zip(W,input_pw)] for W in W_pw],axis=1)

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
                              
        # iterate through fc_layers layers
        for i in xrange(self.fc_layers):
            layers.append(activation_fn(tf.matmul(layers[-1],W_fc[i]) + b_fc[i],
                          fn=self.fc_fn[i],dropout=self.fc_dropout[i]))
            if not self.silent: 
                print 'Layer {} fc input: {}'.format(i+1,layers[-2].shape)
                print 'Layer {} fc output: {}'.format(i+1,layers[-1].shape)

        
        ## TRAINING METHODS
        self.y_out = layers[-1]
        self.weights = {'SP':[W_sw,W_pw],'FC':[W_fc,b_fc]}

        # Alert user
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
