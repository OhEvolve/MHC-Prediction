
""" 
file: /simulation/inputs/save_data.py 
function: saves a particular fold in a given folder
author: PVH
"""

# standard libraries
import os


# nonstandard libraries
import numpy as np

# homegrown libraries

def decoder(data,chars,mode='one-hot'):

    seqs = []

    # decoder for one-hot data
    if mode == 'one-hot':
        for seq in data:
            seqs.append('')
            for aa in seq.transpose():
                seqs[-1] += chars[int(np.argwhere(aa == 1))]

    # decoder for numerical indexed data
    elif mode == 'numerical':
        for seq in data:
            seqs.append('') 
            for aa in seq:
                seqs[-1] += chars[aa]
    
    return seqs

def save_data(data,**kwargs):
    # allow user to modify particular parameters
    options = {'silent':False,
               'data_directory':'./data',
               'input_type':'one-hot',
               'characters':'ABCDEFGHIJKLMNOPQRSTUVWXYZ_'}
    options.update(kwargs) # update options with user input


    # make some assertions about what the user options are now
    assert len(data) == 2, 'Data passed has unusual length (len = {})'.format(data)
    assert len(data[0]) == len(data[1]), 'Sample and label count not equivalent'

    # check if directory exists, make if not so 
    if not os.path.isdir(options['data_directory'] + '/' + 'folds'):
        os.makedirs(options['data_directory'] + '/' + 'folds')

    # write to file in data folder


    if not options['silent']: 'Saved current data under filename: {}'.format(fname)

    return fname 

if __name__ == '__main__':
    a = np.reshape(np.tile(np.identity(4),(20,1,1)),(20,4,4))
    characters  = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
    print decoder(a,characters)
