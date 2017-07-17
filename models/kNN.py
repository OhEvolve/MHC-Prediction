
'''

Project: Baseline model for sequence landscape prediction, using k-nearest neighbors algorithm
Class(s): (none) 
Function: Organizes main pipeline execution and testing 

Author: Patrick V. Holec
Date Created: 5/9/2017
Date Updated: 5/9/2017

This is for actual data testing

'''

'''
BuildNetwork: Main neural network architecture generator
'''

# standard libaries
import gzip
import math
import os.path
import random

# nonstandard libraries
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
                     'fc_fn':('linear','linear'),
                     'test_fraction': 0.2,
                     'silent':False,
                     'data_label':'test'
                    }
    

    model = BuildModel(data_settings)
    model.fold_data()
    guesses = model.predict()
    #vis.comparison(guesses,model.test_labels)
    vis.auroc(guesses,model.test_labels)     
    
class BuildModel:
    
    def default_model(self):
        # basically every parameter defined in one dictionary
        default_params = {#'data_augment':False,
                         'data_normalization': False,
                         'silent': False,
                         'test_fraction': 0.1,
                         'k_nearest': 4,
                         'data_label':'test'
                         #'batch_size':100,
                         #'num_epochs':500,
                         #'sw_depth':1,
                         #'pw_depth':1,
                         #'dropout':1.0
                         }
        
        # apply all changes
        self.update_model(default_params)

    # use a dictionary to update class attributes
    def update_model(self,params={}):
        for key, value in params.items():
            setattr(self, key, value)
    
    # model initialization
    def __init__(self,params = {}):
        
        # set all default parameters
        self.default_model()
        
        # check to see if there is an update
        if params:
            print 'Updating model with new parameters:'
            for key,value in params.iteritems(): print '  - {}: {}'.format(key,value)
            self.update_model(params)
        
        print 'Initializing neural network data acquisition...'
        
        # load data parameters
        data = load_data.LoadData(self.data_label)
        self.__dict__.update(data.params)
        self.all_data_sw = data.data_array_sw # load all variables
        self.all_labels = data.label_array
        
        # 
        self.sw_dim = self.all_data_sw.shape # save dimensions of original data
 
        # verified reduction of dimension (flatten) and merging
        self.all_data = np.reshape(self.all_data_sw,
                                          (self.sw_dim[0],self.sw_dim[1]*self.sw_dim[2]))
         
        # update on model parameters
        if not self.silent:
            print '*** System Parameters ***'
            print '  - Sequence length:',self.length
            print '  - AA count:',self.aa_count
            print '  - Data shape:',self.all_data.shape
        
        print 'Finished acquisition!'

    def fold_data(self,params = {}):
        # check to see if there is an update
        if params:
            print 'Updating model with new parameters:'
            for key,value in params.iteritems(): print '  - {}: {}'.format(key,value)
            self.update_model(params)
            
        print 'Starting data formatting...'
        
        self.fold_total = int(1/self.test_fraction)
        
        # randomize data order
        self.order = np.arange(0,self.all_data.shape[0])
        self.limits = [i*int(self.test_fraction*self.all_data.shape[0]) for i in xrange(self.fold_total+1)]
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
        
        print 'Data specified by user!'
                
            
    def network_initialization(self):
        if not self.silent: print 'kNN models do not require explicit network building, skipping...'
        
        
    def train(self):
        if not self.silent: print 'kNN models do not require explicit training, skipping...'
            
            
    def predict(self,data=[]):
        # if no inputs are specified, use the defaults
        if len(data) == 0:
            data = self.test_data
        
        # create distance matrix
        assert self.train_data.shape[1:] == data.shape[1:], 'Test and train data not the same shape (axis 1+).'
        
        # create guesses for each string in data
        guesses = []
        for d in data:
            dists = np.array([np.count_nonzero(d!=t) for t in self.train_data])
            inds = dists.argsort()[:self.k_nearest]
            guesses.append(np.mean([self.train_labels[i] for i in inds]))
        
        print 'Finished guessing!'
        
        return guesses


'''
Catch if called as script
'''
if __name__ == '__main__':
    main()
