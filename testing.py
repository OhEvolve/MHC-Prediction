
"""
This is a building ground for new ideas and optimization 
NOT FOR RUNNING REAL SIMULATION
"""



from analysis import * # libraries: visualize,statistics
import numpy as np

scale = 0.00001

actual = np.arange(0,1,0.01)
guesses = actual + np.random.normal(scale=scale,size=actual.shape)

print statistics.auroc(guesses,actual)




