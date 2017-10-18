
"""
file: /analysis/library_complexity.py
function: displays the complexity of any input datafile 
author: PVH
"""

# standard libraries
import os
import csv
import random

# nonstandard libraries
import matplotlib.pyplot as plt

def library_complexity(fname,*args,**kwargs):

    """ Tests a single library """

    # allow user to modify particular parameters
    options = {'silent':False,
               'axis_handle':None,
               'data_label':'A12',
               'show_graph':True
               }

    for arg in args: options.update(arg) # update with input dictionaries
    options.update(kwargs) # update with keyword arguments

    # reassignments
    label = options['data_label']

    # prepare for plotting
    if options['axis_handle'] == None:
        f,ax = plt.figure()
    else:
        ax = options['axis_handle']


    # make sure files are in appropriate paths 
    if not os.path.isdir('./data/{}'.format(label)): 
        raise IOError('Directory (/data/{}) does not exist!'.format(label))
    if not os.path.isfile('./data/{}/{}.csv'.format(label,label)): 
        raise IOError('File (/data/{}/{}.csv) does not exist!'.format(label,label))

    # load data 
    with open('./data/{}/{}.csv'.format(label,label),'rb') as csvfile:
        reader = csv.reader(csvfile)
        data = [[r[0]]+map(int,r[1:]) for r in reader] 

    for ii in xrange(1,len(data[0])):
        uniques,ydata = {},[0]
        round_data = [d[ii] for d in data]
        seq_data = [i for i,r in enumerate(round_data) for ind in xrange(r)] 
        random.shuffle(seq_data)

        for s in seq_data:
            try:
                uniques[s] += 1
            except KeyError:
                uniques[s] = 1
            ydata.append(len(uniques)) 
             
        ax.plot(xrange(len(ydata)),ydata,label='Round {}'.format(ii-1))
    
    ax.plot([0,ax.get_ylim()[1]],[0,ax.get_ylim()[1]],'--',color='k')

    # plot formatting
    ax.set_xlabel('Sequence Read #')
    ax.set_ylabel('Unique #')
    ax.set_title('Data Set: {}'.format(options['data_label']))
    ax.legend(loc = 0)

    if options['show_graph'] == True:
        # show plot
        plt.show(block=False)
        raw_input('Press enter to close...')
        plt.close()
        


