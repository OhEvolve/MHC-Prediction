

""" 
file: /simulation/unit_tests.py 
function: perform unit tests on all available functions to ensure function 
author: PVH
"""

#  standard libraries
import unittest

# nonstandard libraries
import numpy as np

# homegrown libraries
from simulation.input import *

class TestInputFunctions(unittest.TestCase):

    def test_sample_load_data(self):
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
        matching_check = (data_orig == data_random)
        ordering_check = (np.mean(data_random[0],axis=1,dtype=int) == np.squeeze(data_random[1])).all()

        # unit tests
        self.assertTrue(matching_check) # check if the arrays have actually changed
        self.assertTrue(ordering_check) # check if the arrays have same ordering

if __name__ == '__main__':
    print 'Starting unit tests...\n'
    suite = unittest.TestLoader().loadTestsFromTestCase(TestInputFunctions)
    unittest.TextTestRunner(verbosity=2).run(suite)
