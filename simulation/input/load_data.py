
""" 
file: /simulation/inputs/load_data.py 
function: enter location of a file, returns data and parameter location
author: PVH
"""

#  standard libraries
import os

# nonstandard libraries
import numpy as np

# homegrown libraries



def load_data(fname,**kwargs):
    # allow user to modify particular parameters
    options = {'silent':False, 
               'encoding':'one-hot'}
    options.update(kwargs) # update options with user input

    # make some assertions about what the user options are now
    accepted_encodings = ('one-hot','numerical')
    assert options['encoding'] in accepted_encodings,  \
            '{} not in accepted input types {}'.format(options['encoding'],accepted_encodings)

    # try to find file in multiple contexts
    if os.path.isfile(fname): pass
    elif os.path.isfile('./data/'+fname): fname = './data/' + fname
    elif os.path.isfile('./'+fname): fname = './' + fname
    else: raise IOError('Could not find file at {}'.format(fname))

    # open file and store lines
    with open(fname,'rb') as txtfile:
        reader = txtfile.readlines()
        raw_seqs = [r.split(',')[0] for r in reader] 
        raw_labels = [float(r.split(',')[1]) for r in reader] 

    assert len(raw_seqs) == len(raw_labels), 'Sequence count not same as label count'

    # get parameters out of data
    chars = ''.join(sorted(set(''.join(raw_seqs))))
    params = {'data_location':fname,
              'length':len(raw_seqs[0]),
              'aa_count':len(chars),
              'characters':chars,
              'sequence_count':len(raw_seqs),
              'pair_count':(len(raw_seqs[0])*(len(raw_seqs[0])-1))/2}

    # give user a little heads up
    if not options['silent']:
        print 'Loaded data with following parameters:'
        for k,v in params.items(): print ' > {} : {}'.format(k,v)

    # always make label array
    all_labels = np.reshape(np.array(raw_labels),(len(raw_labels),1))

    if options['encoding'] == 'one-hot':
        # one-hot encoding (sitewise)
        all_data = np.zeros((params['sequence_count'],params['aa_count'],params['length']),np.int)
        for i,sample in enumerate(raw_seqs):
            for j,char in enumerate(sample):
                all_data[i,params['characters'].index(char),j] = 1

    elif options['encoding'] == 'numerical':
        # numerical encoding
        all_data = np.zeros((params['sequence_count'],params['length']),np.int)
        for i,sample in enumerate(raw_seqs): # go across sequences
            for j,char in enumerate(sample): # go across residues in sequence
                all_data[i,j] = params['characters'].index(char)

    if not options['silent']: print 'Finished generating data!'

    # return a tuple of data/labels, and discovered parameters for merging
    return (all_data,all_labels),params


