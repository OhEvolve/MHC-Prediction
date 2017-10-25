

'''
Project: Neural Network for MHC Peptide Prediction
Class(s): BuildNetwork
Function: Generates specified neural network architecture (data agnostic)
Author: Patrick V. Holec ''' 


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
        default_params = default_parameters.spinn() 
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
        assert len(self.fc_depth) >= self.fc_layers,'Entries in depth less than number of fc layers.'



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
        b_fc = [uniform_variable([depth[i+1]]) for i in xrange(self.fc_layers)]
                              
        # iterate through fc_layers layers
        for i in xrange(self.fc_layers):
            layers.append(activation_fn(tf.matmul(layers[-1],W_fc[i]) + b_fc[i],
                          fn=self.fc_fn[i],dropout=self.fc_dropout[i]))
            if not self.silent: 
                print 'Layer {} fc input: {}'.format(i+1,layers[-2].shape)
                print 'Layer {} fc output: {}'.format(i+1,layers[-1].shape)

        
        ## TRAINING METHODS
        self.y_out = layers[-1]
        self.weights = {'sitewise':W_sw,
                        'pairwise':W_pw,
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
            print 'WM:',w
            weight_matrix = self.sess.run(w,feed_dict={})
            model_dict[l] = weight_matrix
            
        # pickle file
        with open(fname, 'w') as f:
            pickle.dump(model_dict,f)
        
        return i+1 # return index of model file for later reference

    def predict(self,data=[]):
        
        # batch iterate over test data
        guesses = np.zeros((len(data),1))
        
        for i in xrange(0,len(data),self.batch_size):
            if not self.silent: print 'Starting batch prediction at index {}...'.format(i)
            feed_dict = {self.train_x: data[i:i+self.batch_size,:]}
            print 'YO:',self.y_out
            guesses[i:i+self.batch_size,:] = self.sess.run(self.y_out,feed_dict=feed_dict)

        if not self.silent: print 'Finished guessing!'

        return guesses

        
'''
Catch if called as script
'''
if __name__ == '__main__':
    main()
