
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
import random
import os

# nonstandard libraries
import tensorflow as tf
import numpy as np

# homegrown libraries
from common_methods import *
import default_parameters


'''
Main Call Function
'''

def main():
    """ Mild testing function """
    model = BuildModel()
    

'''
Main Class Method
'''

class BuildModel:
    


    def default_model(self):
        """ Pulls a default parameter list and applies parameters to model """
        # basically every parameter defined in one dictionary
        default_params = default_parameters.cnn() 
        self.model_parameters = default_params.keys()
        
        # apply all changes
        for key, value in default_params.items():
            setattr(self, key, value)
            


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
        assert len(self.conv_stride) >= self.cnn_layers, \
            'Entries in strides less than number of conv. layers.'
        assert len(self.conv_filter_width) >= self.cnn_layers, \
            'Entries in filter width less than number of conv. layers.'
        assert len(self.conv_depth) >= self.cnn_layers, \
            'Entries in depth less than number of conv. layers.'
        assert len(self.conv_padding) >= self.cnn_layers, \
            'Entries in padding less than number of conv. layers.'
        assert len(self.pool_stride) >= self.cnn_layers, \
            'Entries in pool stride less than number of pool layers.'
        assert len(self.pool_filter_width) >= self.cnn_layers, \
            'Entries in filter width less than number of pool layers.'
        assert len(self.pool_padding) >= self.cnn_layers, \
            'Entries in padding less than number of pool layers.'
        assert len(self.fc_depth) >= self.fc_layers, \
            'Entries in depth less than number of fc layers.'



    def count_trainable_parameters(self):
        """
        Return the total number of trainable parameters in the model, 
        which is useful for system model matching
        """
        return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) 



    def __init__(self,*args):
        """ Initialize class, potentially with specified parameters """ 

        # print version numbers that matter
        print 'Loaded Tensorflow library {}.'.format(tf.__version__)
        
        # set all default parameters, potentially replace
        self.default_model()

        # update model with potentially new values
        for arg in args: 
            self.update_model(arg)

        # clear TF graph, modify TF's verbosity
        self.clear_graph()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



    def clear_graph(self):
        """ Clear graph, be sure its actually clear """ 
        tf.reset_default_graph()



    def reset_variables(self):
        """ Resets variables on graph, using given random seed """ 
        tf.set_random_seed(self.seed)
        self.sess.run(tf.global_variables_initializer())



    def network_initialization(self,*args):
        """ Constructs new network given current parameters """

        # update model with potentially new values
        for arg in args:
            self.update_model(arg)

        # clear TF graph
        self.clear_graph()

        # Start tensorflow engine
        self.sess = start_tf_engine(self.tf_device)

        if not self.silent: print 'Building model...'
        
        # create placeholder variables, and format data
        self.train_x = tf.placeholder(tf.float32, shape=(None, self.aa_count,self.length)) 
        self.train_y = tf.placeholder(tf.float32, shape=(None, 1)) # full energy input
        layers = [tf.reshape(self.train_x, [-1, self.aa_count, self.length, 1])]
        
        ## CONVOLUTIONAL LAYER (CNN) GENERATOR ##
        # temporary variables for non-symmetry
        width,depth = list(self.conv_filter_width),[1] + list(self.conv_depth)

        # create weight variables
        W = [normal_variable([self.aa_count,width[i],depth[i],depth[i+1]]) for i in xrange(self.cnn_layers)]        

        # iterate through conv./pool layers
        for i in xrange(self.cnn_layers):
            layers.append(conv2d(layers[-1],W[i],stride=self.conv_stride[i])) # build conv layer
            print 'Layer {} conv. output: {}'.format(i+1,layers[-1].shape)
            
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
        W_fc = [normal_variable([depth[i],depth[i+1]]) for i in xrange(self.fc_layers)]  
        b_fc = [uniform_variable([depth[i+1]]) for i in xrange(self.fc_layers)]
                              
        # iterate through fc_layers layers
        for i in xrange(self.fc_layers):
            layers.append(activation_fn(tf.matmul(layers[-1],W_fc[i]) + b_fc[i],
                          fn=self.fc_fn[i],dropout=self.fc_dropout[i]))
            print 'Layer {} fc output: {}'.format(i+1,layers[-1].shape)
        
        ## TRAINING METHODS
        self.y_out = layers[-1]
        self.weights = {'w_conv':W,
                        'w_fc':W_fc,
                        'b_fc':b_fc}

        # define loss function
        self.loss = loss(tf.subtract(self.y_out,self.train_y),
                         loss_type = self.loss_type,
                         loss_magnitude = self.loss_magnitude,
                         silent = self.silent,
                         normalize = False)

        self.loss += loss(self.weights,
                         loss_type = self.reg_type,
                         loss_magnitude = self.reg_magnitude,
                         silent = self.silent,
                         normalize = True)

        # initialize variable
        self.reset_variables()

        # Alert user
        if not self.silent: print 'Finished!'        

        
        
    # TODO: Fix this
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
        
        
        
        
    
