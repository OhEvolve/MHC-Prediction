
""" 
file: /testing/common_methods.py 
function: collection of common functions that assist testing functions
author: PVH
"""

from simulation.models import *
from simulation.training import *

def choose_model(mode):

    """ Selects a model based on input string """ 

    if mode.lower() == 'spinn':
        model = spiNN.BuildModel()
        settings = {'encoding':'numerical','type':'nn'}
    elif mode.lower() == 'cnn':
        model = cNN.BuildModel()
        settings = {'encoding':'one-hot','type':'nn'}
    elif mode.lower() == 'knn':
        model = kNN.BuildModel()
        settings = {'encoding':'numerical','type':'ml'}
    else:
        raise KeyError('{} not found in model selection'.format(mode))

    return model,settings


def choose_training(model,data,mode):

    """ Select a training method based on input string """

    if mode == 'nn':
        train_nn(model,data)
    elif mode == 'ml':
        train_ml(model,data)



