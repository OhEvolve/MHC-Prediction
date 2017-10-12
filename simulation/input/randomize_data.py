
""" 
file: /simulation/inputs/randomize_data.py 
function: enter location of a file, returns data and parameter location
author: PVH
"""

#  standard libraries
import random 
import sys

# nonstandard libraries
import numpy as np

# homegrown libraries

def randomize_data(data,**kwargs):
    # allow user to modify particular parameters
    options = {'silent':False, 
               'seed':random.randrange(999999999)}
    options.update(kwargs) # update options with user input

    # make some assertions about what the user options are now
    assert type(options['seed']) in (int,long,float), 'Random seed needs to be of type <int,float,long>' 
    assert len(data) == 2, 'Data passed has unusual length (len = {})'.format(data)
    assert len(data[0]) == len(data[1]), 'Sample and label count not equivalent'

    # update user on seed generation
    if not options['silent']: print 'Randomizing data under seed: {}'.format(options['seed']) 

    # set random seed twice to ensure both systems get randomized identically
    for d in data:
        np.random.seed(options['seed']) # set identical seeds
        np.random.shuffle(d) # shuffle data in place
    
    # return shuffled data
    return data 
