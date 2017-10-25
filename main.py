
""" 
file: /simulation/unit_tests.py 
function: perform unit tests on all available functions to ensure function 
author: PVH
"""

#  standard libraries
import os
import unittest
from sys import argv

# nonstandard libraries
import numpy as np
import tensorflow as tf

# homegrown libraries
from simulation.input import *
from simulation.training import *
from simulation.testing import *
from simulation.models import *

from analysis import *

# library modifications
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(args):
                
    for command in args['commands']: 
        if command == 'b': 
            all_params = generate_dicts(default = True)
            batch_model(all_params,thread_count = 12)
            print 'Finished!'

        if command == 's':

            params = {
                     'data_label':'A12.txt',
                     'num_epochs':20,
                     'learning_rate':20,
                     'reg_magnitude':0.01
                     }

            results = single_model(params)

        if command == 'p': # p - probe a file
             
            fname = 'results_100003.p'
            
            if os.path.isfile(fname): 
                data = pickle.load(open(fname),'rb')
            elif os.path.isfile('/logs' + fname): 
                data = pickle.load(open('/logs/' + fname),'rb')
            else:
                print 'File not found!'
                return None
            
             
""" 
Formatting
"""

""" Collects command line arguments """
# note: could accept keywords as opposed to key characters if support is desired
def getopts(argv):
    opts = {'commands':[]} # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            if argv[0][1] == '-': # Pulls in a new option if double-dashed
                opts[argv[0][:argv[0].index('=')]] = argv[0][argv[0].index('=')+1:]
            else: # Otherwise loads boot commands
                opts['commands'].append(argv[0][1:])
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    # place a translator for keyworks if desired
    opts['commands'] = ''.join(opts['commands'])
    return opts
                                                                                                            

def open_results_pickle(fname):
    if os.path.isfile(fname): 
        data = pickle.load(open(fname),'rb')
    elif os.path.isfile('/logs' + fname): 
        data = pickle.load(open('/logs/' + fname),'rb')
    else:
        print 'File not found!'
        return None
    return data

""" namespace catch if called as script (which it should) """ 
if __name__ == "__main__":
    args = getopts(argv)
    main(args)
else:
    print "main.py not intended to be loaded as a module."


