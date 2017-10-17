
""" 
file: /simulation/inputs/fold_data.py 
function: picks a fold in the dataset
author: PVH
"""

# standard libraries

# nonstandard libraries
import numpy as np

# homegrown libraries


def fold_data(data,**kwargs):
    # allow user to modify particular parameters
    options = {'silent':False,
               'fold_index':1,
               'fold_count':5}
    options.update(kwargs) # update options with user input

    # make some assertions about what the user options are now
    assert len(data) == 2, 'Data passed has unusual length (len = {})'.format(data)
    assert len(data[0]) == len(data[1]), 'Sample and label count not equivalent'
    assert type(options['fold_count']) == int, 'Fold count must be int'
    assert type(options['fold_index']) == int, 'Fold index must be int'

    #
    fold_index = options['fold_index']%options['fold_count']
    start = (fold_index)*len(data[0])/options['fold_count'] 
    stop = (fold_index+1)*len(data[0])/options['fold_count'] 

    # split data and recompose
    wd1,fd,wd2 = np.split(data[0],[start,stop],axis=0)
    wl1,wl,wl2 = np.split(data[1],[start,stop],axis=0)
    
    # result
    data_fold = {'testing':(fd,wl),
                 'training':(np.concatenate((wd1,wd2),axis=0),np.concatenate((wl1,wl2),axis=0))}

    # returns (fold, remaining data)
    return data_fold


