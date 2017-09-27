
import os
import math

import numpy as np
import matplotlib.pyplot as plt


    
'''
Testing Code
'''

def main():
    
    params = {
             'silent':False
             }
        
    enrichment = Enrichment(params)
    enrichment.load_data()
    enrichment.build_profiles()
    
    
'''
Factory Methods
'''

def FindFolder(folder_name,max_depth = 1):
    for i in xrange(max_depth+1):
        # walk through file tree at current level in search of file
        print 'Starting search in: {}'.format(os.getcwd())
        for dirpath, dirnames, filenames in os.walk("."):
            for filename in [d for d in dirnames if d == folder_name]:
                return os.path.join(dirpath, filename)
        os.chdir('..')
    return False # in case of failure


# check if input is representable by an integer
def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

    
    
'''
Main Method
'''

class Enrichment:
    
    # creates a set of default parameters for the model
    def default_model(self):
        # basically every parameter defined in one dictionary
        default_params = {
                         'randomize_sequences':False,
                         'data_label':'A12',
                         'log_enrichment_base':2,
                         'naive_count_threshold':3,
                         'evolved_count_threshold':3,
                         'silent':False
                         }
        
        # apply all changes
        self.update_model(default_params,alert = False)

    # use a dictionary to update class attributes
    
    
    # updates the models with a new parameter dictionary
    def update_model(self,params={},alert=False):
        
        # if sent at a surprising time, let user know what was updated
        if alert:
            print 'Updating model with new parameters:'
            for key,value in params.iteritems(): print '  - {}: {}'.format(key,value)
            self.update_model(params)
        
        # makes params dictionary onto class attributes
        for key, value in params.items():
            setattr(self, key, value)
        
        # checks for adequete coverage
        
        
    # model initialization
    def __init__(self,params = {}):
    
        # set all default parameters
        self.default_model()
        
        # check to see if there is an update
        if params: self.update_model(params,alert = True)
            
    
    # load sequence count information for each round of selection
    def load_data(self,params = {}):
        
        # check to see if there is an update
        if params: self.update_model(params,alert = True)
            
        # finds the requested folder and uses it to identify rounds of selection
        directory = FindFolder(self.data_label)
        rounds = [file for file in os.listdir(directory) if file.startswith('round')]
        seqs = [file for file in os.listdir(directory) if file.startswith('peptide') or file.startswith('seq')]
        
        # check for basic parameters
        assert len(seqs) == 1, 'Number of entries is not 1 ({}), exiting...'.format(len(seqs))
        assert len(rounds) > 1, 'Number of entries in selection data is under 2 ({}), exiting...'.format(len(rounds))
        
        # scan through each of the rounds files
        self.total_rounds = len(rounds)
        self.results = []
        
        # identify the list of peptides corresponding to each row in the count files
        with open(directory+'/'+seqs[0],'r') as f:
            content = f.readlines()
            self.sequences = [x.strip() for x in content[1:]] 
            
        # identify count sequence info on a round-by-round basis
        for i in xrange(len(rounds)):
            # catch the rounds by their names, rather than order in acquistion
            possible_fnames = [r for r in rounds if r.endswith('{}.txt'.format(i))] # find matching index
            fname = possible_fnames[0] # TODO: make a check here that there is only one entry
            with open(directory+'/'+fname,'r') as f:
                content = f.readlines()
            # you may also want to remove whitespace characters like `\n` at the end of each line
            content = [x.strip() for x in content] 
            self.results.append([int(x) for x in content[1:] if RepresentsInt(x)])
        
        # test results
        for f,r in zip(rounds,self.results):
            print '{}:{}'.format(f,len(r))
        
    
    # builds enrichment profiles that correspond to round comparisons
    def build_profiles(self):
        
        print 'Depositing data in {}'.format(os.getcwd())
                
        # follow the filter parameters to put together enrichment scores
        for i in xrange(self.total_rounds-1):
            seqs,scores = [],[] # initialize variables
            for p,s1,s2 in zip(self.sequences,self.results[i],self.results[i+1]):
                if s1 >= self.naive_count_threshold and s2 >= self.evolved_count_threshold:
                    seqs.append(p)
                    scores.append(math.log(float(s2)/s1,self.log_enrichment_base))
            # print to file
            output_file = '{}_{}-{}_seqs.txt'.format(self.data_label,i,i+1)
            with open(output_file,'w') as f:
                for item in ['{},{}'.format(a,b) for a,b in zip(seqs,scores)]:
                    f.write("%s\n" % item)
            print 'Wrote enrichment data to {}.'.format(output_file)
        