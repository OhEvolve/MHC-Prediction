
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
from simulation.testing import *
from simulation.models import *

# library modifications
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



all_params = generate_dicts(default = True)
batch_model(all_params,thread_count = 12)
print 'Finished!'

