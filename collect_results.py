
'''

Results Collector

Function: Compiles log files that are located in the log folder

'''

# standard libraries
import os,glob
import pickle

# nonstandard libaries
import numpy as np
import matplotlib.pyplot as plt
import plotly
plotly.tools.set_credentials_file(username='Pandyr',api_key='HCQRVKP8DriBovEHMGXt')
import plotly.plotly as py
import plotly.graph_objs as go

# homegrown libraries


'''
Main Routine
'''


# Sampling parmeters
all_logs = False 
log_rng = (10002,10152)

os.chdir('./logs') # change directories

# Generate log files 
if all_logs:
    log_files = [file for file in glob.glob('*.p')]
else:
    log_files = ['results_{}.p'.format(i) for i in xrange(log_rng[0],log_rng[1])]

# Extract dictionaries
log_dicts = [pickle.load(open(f,'rb')) for f in log_files]

# pullout auROCs
train_aurocs,test_aurocs = [],[]
for i,log in enumerate(log_dicts):
    train_aurocs.append(np.mean([log['train_dataset'][i+1] for i in xrange(0,len(log['train_dataset']),2)]))
    test_aurocs.append(np.mean([log['test_dataset'][i+1] for i in xrange(0,len(log['test_dataset']),2)]))


# plot results
data = [go.Scatter(x = train_aurocs,y = test_aurocs,mode = 'markers')]
py.plot(data, filename = 'auroc-comparison')
#print test_aurocs

# other shit

