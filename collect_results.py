
'''

Results Collector

Function: Compiles log files that are located in the log folder

'''

# standard libraries
import os,glob
import pickle
import csv

# nonstandard libaries
import numpy as np
import matplotlib.pyplot as plt
import plotly
plotly.tools.set_credentials_file(username='Pandyr',api_key='HCQRVKP8DriBovEHMGXt')
import plotly.plotly as py
import plotly.graph_objs as go
import joblib

# homegrown libraries


'''
Main Routine
'''


# Sampling parmeters
all_logs = True
log_rng = (100002,100152)

os.chdir('./logs') # change directories

# Generate log files 
if all_logs:
    log_files = [file for file in glob.glob('*.p')]
else:
    log_files = ['results_{}.p'.format(i) for i in xrange(log_rng[0],log_rng[1])]

# Extract dictionaries
train_aurocs,test_aurocs = [],[]
data_extract = []


header = ['Index','Train AUROC','Test AUROC','Train AUROC STD','Test AUROC STD'] + pickle.load(open(log_files[0],'rb'))['model_settings'].keys()

for i,f in enumerate(log_files):
    if i%1000 == 0: print 'Starting file {}...'.format(i+1)
    #log_dict = joblib.load(f)
    try: log_dict = pickle.load(open(f,'rb'))
    except ValueError: print 'Failure on file {}'.format(f)
    train_auroc,test_auroc = np.mean(log_dict['train_auroc']),np.mean(log_dict['test_auroc'])
    train_auroc_std,test_auroc_std = np.std(log_dict['train_auroc']),np.std(log_dict['test_auroc'])
    data_extract.append([i+1,test_auroc,train_auroc,test_auroc_std,train_auroc_std] + log_dict['model_settings'].values())
    
os.chdir('..')

with open('east4_results.csv','wb') as f:
    writer = csv.writer(f)
    writer.writerows([header] + data_extract)

print 'Finished!'

    #train_aurocs.append(np.mean(log_dict['train_auroc']))
    #test_aurocs.append(np.mean(log_dict['test_auroc']))



# plot results
#data = [go.Scatter(x = train_aurocs,y = test_aurocs,mode = 'markers')]
#py.plot(data, filename = 'auroc-comparison')
#print test_aurocs

# other shit

