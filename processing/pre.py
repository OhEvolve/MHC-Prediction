
"""
pre.py

Performs preprocessing on data sets for input into ML models
"""


# standard libraries
import pickle
import random
import os
import csv
import math
import random

# nonstandard libraries
import numpy as np

"""
Inputs 
    params >  dictionary with possible arguments:
        mode - [str], ('enrichment','attrition'), which sequence interpretation strategy
        rounds - [tuple], which rounds to include
        label - [str], which folder to look for data in
"""

def start(params):
    default_params = defaults()
    params = append_dicts(params,default_params)
    
    # reassign variables for easier access
    label = params['data_label']
    mode = params['mode']
    rounds = params['rounds']
    thresh = params['threshold']
    silent = params['silent']

    # check parameter types
    assert (type(label)==str),'Parameter "label" of type <{}>, not <str>'.format(type(label)) 
    assert (type(mode)==str),'Parameter "label" of type <{}>, not <str>'.format(type(mode)) 
    assert (type(rounds)==tuple),'Parameter "label" of type <{}>, not <tuple>'.format(type(rounds)) 

    # make sure files are in appropriate paths 
    if not os.path.isdir('./data/{}'.format(label)): 
        raise IOError('Directory (/data/{}) does not exist!'.format(label))
    if not os.path.isfile('./data/{}/{}.csv'.format(label,label)): 
        raise IOError('File (/data/{}/{}.csv) does not exist!'.format(label,label))

    # load data 
    with open('./data/{}/{}.csv'.format(label,label),'rb') as csvfile:
        reader = csv.reader(csvfile)
        data = [[r[0]]+map(int,r[1:]) for r in reader] # create data list with each peptide followed by rounds

    total_rounds = len(data[0]) - 1
   
    # make sure round indices are within bounds
    if rounds[1] >= total_rounds:
        print 'Adjusting rounds to be below maximum index ({}->{})'.format(rounds[1],total_rounds-1)
        rounds = (rounds[0],total_rounds-1)

    ### MODE PROCESSING ###
    
    if mode == 'enrichment':
        results = [[d[0],str(math.log(float(d[rounds[1]+1])/float(d[rounds[0]+1]),2))] 
                for d in data if d[rounds[1]+1] >= thresh and d[rounds[0]+1] >= thresh]
    elif mode == 'attrition':
        results = []
        for d in data:
            if (np.count_nonzero(d[1:]) >= 2): # if appearance in 2+ rounds of selection
                results.append([d[0],max(i for i,j in enumerate(d[1:]) if j > 0)])

        # Put together a dictionary with distribution
        rdict = dict([(i,0) for i in xrange(1,total_rounds)])
        for r in results: rdict[r[1]] += 1

        # Normalize entries
        rmin,rmax = min([r[1] for r in results]),max([r[1] for r in results]) 
        results = [[r[0],str((float(r[1]) - rmin)/float(rmax-rmin))] for r in results]

        # Gives some heads up to user if requested
        if not silent:
            print 'Label:',label
            print ' > Total:',len(results)
            for k,v in rdict.items():
                print ' > {}: {}'.format(k,v)
            print 'Examples:'
            for ind in [random.randint(0,len(results)) for i in xrange(15)]:
                print ' > {}: {}'.format(*results[ind])

    # Build test files
    with open('./data/'+label+'.txt','w') as seqfile:
        print 'Writing to file:','./data/'+label+'.txt'
        for r in results:
            seqfile.write(','.join(r)+'\n')

# END #



#####################
## FACTORY METHODS ##
#####################

def append_dicts(old_dict,*dict_args):
    """
    Given an original dictionary and any number of additional dicts, adds
    new entries to original if they are not already present
    """
    for dictionary in dict_args:
        for key in dictionary.keys():
            if not key in old_dict.keys():
                print 'Adding parameter: {} -> {}'.format(key,dictionary[key])
                old_dict[key] = dictionary[key]
    return old_dict



#######################
## PARAMETER PASSING ##
#######################

def defaults():
    return {
            'mode':'enrichment',
            'rounds':(2,3),
            'data_label':'A12',
            'threshold':1,
            'silent':False
           }



# taking a hard datafile and doing something useful with it

class ParameterizeData:
    def __init__(self,parameters,label='test'):
        if not os.path.isfile(label+'_seqs.txt'): 
            print 'Sequence file not found. Exiting...'    
            return False
        if not all(k in parameters for k in ('length','aa_count','characters')):        
            print 'Missing one of the following keys: length, aa_count, characters. Exiting...'
            return False 
         
        # find number of sequences in selected file
        with open(label+'_seqs.txt') as fname: 
            for i,l in enumerate(fname): pass
        parameters['sequence_count'] = i
        parameters['label'] = label

        # creates a pickled file with a map of specifications
        pickle.dump(parameters,open('{}_params.p'.format(label),'wb'))
         
