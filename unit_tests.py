

""" 
file: /simulation/unit_tests.py 
function: perform unit tests on all available functions to ensure function 
author: PVH
"""

#  standard libraries
import os
import unittest

# nonstandard libraries
import numpy as np
import tensorflow as tf

# homegrown libraries
from simulation.input import *
from simulation.training import *
from simulation.models import *

# library modifications
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestInputFunctions(unittest.TestCase):

    def test_load_data(self):
        # use load_data function on sample file
        data,params = load_data('sample.txt',silent=True)
        
        # unit tests on parameterization 
        self.assertEqual(params['length'],5)
        self.assertEqual(params['aa_count'],6)
        self.assertEqual(params['characters'],'ABCDEF')
        self.assertEqual(params['sequence_count'],100)

    def test_sample_randomize_data(self):
        # make fake data vectors for verification
        data = (np.reshape(np.repeat(xrange(0,20),4),(20,4)),np.reshape(xrange(0,20),(20,1,1)))
        data_orig,data_random  = data[:],randomize_data(data,silent=True)

        # unit tests
        matching_check = (data_orig == data_random)
        ordering_check = (np.mean(data_random[0],axis=1,dtype=int) == np.squeeze(data_random[1])).all()
        self.assertTrue(matching_check) # check if the arrays have actually changed
        self.assertTrue(ordering_check) # check if the arrays have same ordering

    def test_save_data(self):
        # make fake data for multiple encodings 
        numerical = np.array([[4,3,2,4,6],[7,6,5,2,3],[4,3,1,2,1],
                      [0,1,2,3,4],[5,3,1,2,3],[0,0,0,2,1]])
        one_hot = np.zeros((6,8,5))
        labels = np.reshape(np.arange(6),(6,1,1))
        for i in xrange(6): one_hot[i,numerical[i,:],np.arange(5)] = 1
        numerical_data,one_hot_data = (numerical,labels),(one_hot,labels)
         
        # try to save data and load data
        numerical_fname = save_data(numerical_data,encoding = 'numerical',silent=True)
        one_hot_fname = save_data(one_hot_data,encoding = 'one-hot',silent=True)
        numerical_data_load,_ = load_data(numerical_fname,encoding = 'numerical',silent=True)
        one_hot_data_load,_ = load_data(one_hot_fname,encoding = 'one-hot',silent=True)

        # unit tests
        numerical_check = all([(a == b).all() for a,b in zip(numerical_data,numerical_data_load)])
        one_hot_check = all([(a == b).all() for a,b in zip(one_hot_data,one_hot_data_load)])
        self.assertTrue(numerical_check)
        self.assertTrue(one_hot_check)

    def test_fold_data(self):
        # load sample data, attempt fold split
        data,params = load_data('sample.txt',silent=True)
        data_folded = fold_data(data,fold_index=2,fold_count=5)

        # unit tests
        self.assertEqual(len(data_folded['testing'][0]),len(data_folded['testing'][1]),20)
        self.assertEqual(len(data_folded['training'][0]),len(data_folded['training'][1]),80)
        

class TestTrainingFunctions(unittest.TestCase):

    # Method is now in /models, also check for dictionary support 
    """
    def test_common_methods(self):
        tensor = tf.constant([[3,2],[-1,1]],dtype=tf.float32)
        
        with tf.Session() as sess:
            result_l1 = sess.run(common_methods.loss(tensor,loss_type='l1',loss_magnitude=1))
            result_l2 = sess.run(common_methods.loss(tensor,loss_type='l2',loss_magnitude=1))

        self.assertEqual(result_l1,1.75)
        self.assertEqual(result_l2,1.875)
    """

    def test_training(self):
        
        # generate model
        model = spiNN.BuildModel()

        # load data and initialize architecture
        data,params = load_data('test.txt',encoding='numerical',silent=True)

        add_params = {'num_epochs':50,
                      'reg_magnitude':0.01}

        model.network_initialization(params,add_params)
        data_folded = fold_data(data,fold_index=0,fold_count=5)

        # train model
        train_nn(model,data_folded)
        model.reset_variables() 
        
class TestTestingFunction(unittest.TestCase):

    def test_single_model(self):
        
        data,params = load_data('test.txt',encoding='numerical',silent=True)
        single_model(params)


if __name__ == '__main__':
    print 'Starting unit tests on input functions...\n'
    suite = unittest.TestLoader().loadTestsFromTestCase(TestInputFunctions)
    unittest.TextTestRunner(verbosity=2).run(suite)

    print 'Starting unit tests on training functions...\n'
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTrainingFunctions)
    unittest.TextTestRunner(verbosity=2).run(suite)

    print 'Starting unit tests on training functions...\n'
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTestingFunctions)
    unittest.TextTestRunner(verbosity=2).run(suite)
