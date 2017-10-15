
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
               'data_directory':'./data/folds',
               'data_label':'test',
               'encoding':'one-hot',
               'characters':'ABCDEFGHIJKLMNOPQRSTUVWXYZ_'}
    options.update(kwargs) # update options with user input

    # make some assertions about what the user options are now
    assert len(data) == 2, 'Data passed has unusual length (len = {})'.format(data)
    assert len(data[0]) == len(data[1]), 'Sample and label count not equivalent'

    # check if directory exists, make if not so 
    if not os.path.isdir(options['data_directory']):
        os.makedirs(options['data_directory'])

    # decode data from one-hot or numerical encodings and put together string
    data_decoded = decoder(data[0],options['characters'],options['encoding'])
    labels_decoded = list(np.squeeze(data[1])) # quick conversion to list
    text = '\n'.join([a+','+str(b) for a,b in zip(data_decoded,labels_decoded)])
    
    # try to find filename that isn't used
    for i in xrange(99999):
        fname = '{}/{}_{}.txt'.format(options['data_directory'],options['data_label'],i)
        if os.path.isfile(fname): continue
        with open(fname,'w') as f: f.write(text) # write to file
        break

    # update user on new filename
    if not options['silent']: print 'Saved current data under filename: {}'.format(fname)

    # return filename
    return fname 

# testing to see if things work
if __name__ == '__main__':
    # create both numerical/one-hot test sets
    numerical = np.array([[4,3,2,4,6],[7,6,5,2,3],[4,3,1,2,1],
                  [0,1,2,3,4],[5,3,1,2,3],[0,0,0,2,1]],dtype=float)
    one_hot = np.zeros((6,8,5),dtype=float)
    labels = np.arange(6)
    for i in xrange(6): one_hot[i,numerical[i,:],np.arange(5)] = 1
     
    # try to save data
    save_data((numerical,labels),encoding = 'numerical')
    save_data((one_hot,labels),encoding = 'one-hot')





