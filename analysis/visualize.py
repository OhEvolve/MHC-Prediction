
# coding: utf-8

# In[32]:


'''
Project: Neural Network for MHC Peptide Prediction
Class(s): Visualize
Function: Tries to visualize some of the abilities of othe toools were building

Author: Patrick V. Holec
Date Created: 2/2/2017
Date Updated: 2/2/2017
'''

import os.path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# makeshift checker for prediction performance
def accuracy(guesses,actual,custom_params={}):

    default_params = default_accuracy_params()
    params = append_dicts(custom_params,default_params)

    rounds = len(np.unique(actual))

    # link figure if specified
    if params['fig'] == None: fig = plt.figure()
    else: fig = params['fig']

    # link axis if specified 
    if params['ax'] == None: ax = plt.gca()
    else: ax = params['ax']

    # fix the axes
    plt.sca(ax)

    # fix the x-axis
    if params['mode'] == 'attrition':
        raw_input('HERE')
        actual_with_noise = actual + np.random.uniform(low=-1./(2*rounds),high=1./(2*rounds),size=actual.shape)
        plt.scatter(guesses,actual_with_noise,c=actual,s=1,cmap='copper')
        for i in xrange(1,rounds):
            plt.plot([0, 1], [(float(i)/(rounds-1)) - (1./(2*(rounds-1))) for _ in xrange(2)], 'k--')
        ytick_labels = ['R'+str(i+1) for i in xrange(rounds)]
        ytick_values = [(float(i)/(rounds-1)) for i in xrange(rounds)]
        plt.yticks(ytick_values,ytick_labels)
    else:
        plt.scatter(guesses,actual)


    plt.title(params['title'])
    plt.xlabel(params['xlabel'])
    plt.ylabel(params['ylabel'])

    if not params['skip_show']:
        plt.show(block=False)
        raw_input('Press enter to close...')
        plt.close()
    
def auroc_plot(guesses,actual,custom_params={}):
    """
    Uses a set of guesses and actual values, produces a predictive auROC
    Optional graphing argument
    Note: This used to include its own auroc solver, but it has since been removed
    """

    from analysis import statistics

    # add interable to single inputs for guesses
    if type(guesses) != list and type(guesses) != tuple:
        guesses = [guesses]
    else:
        print "Multiple inputs to auROC plot detected (correct me if I'm wrong)"

    # add interable to single inputs for actual
    if type(actual) != list and type(actual) != tuple:
        actual = [actual]
    else:
        print "Multiple inputs to auROC plot detected (correct me if I'm wrong)"

    default_params = default_auroc_plot_params()
    params = append_dicts(custom_params,default_params)

    # initialize variables (this is always set in auroc_plot
    auroc_params = {'return_coordinates':True} 
    scores,coors = [],[]

    # take each guess,label set and find auroc
    for g,a in zip(guesses,actual):
        score,coor = statistics.auroc(g,a,auroc_params)
        scores.append(score)
        coors.append(coor)

    # link figure if specified
    if params['fig'] == None: fig = plt.figure()
    else: fig = params['fig']

    # link axis if specified 
    if params['ax'] == None: ax = plt.gca()
    else: ax = params['ax']

    # fix the axes to user specification
    plt.sca(ax)

    plt.plot([0, 1], [0, 1], 'k--')

    # this is a little tricky, since it 
    if params['labels'] == None: params['labels'] = ['set_'+str(i+1) for i in xrange(len(guesses))]

    # TODO: assertions for g,a,labels length

    for coor,l,s in zip(coors,params['labels'],scores):
        for i in xrange(100):
            pass#coor[-i][0],coor[-i][1]
        plt.plot([c[0] for c in coor],[c[1] for c in coor],label=l)

    for s in scores: print s 

    plt.xlim((0.,1.))
    plt.ylim((0.,1.))

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    
    if not params['skip_show']:
        plt.show(block=False)
        raw_input('Press enter to close...')
        plt.close()
    
    return scores
    
def auroc_regime(auroc, std = [], xlabels = [], metalabels = [],x_axis_label = '',show_graph = True):
    
    # make fake labels and block legend if nothing passed
    if len(metalabels) == 0: metalabels,legend = ['' for a in auroc],False
    else: legend = True
        
    if len(std) == 0:
        std = [[0 for _ in a] for a in auroc]

    fig, ax = plt.subplots()
    for a,s,l in zip(auroc,std,metalabels):
        ax.errorbar(np.arange(len(a)), a, yerr=s, fmt='o',label=l)

    if len(xlabels) > 0: plt.xticks(np.arange(auroc.shape[1]),xlabels)
    if legend: ax.legend(loc='best')
    
    ax.set_title('AUROC Scores')
    plt.ylim((0,1))
    plt.xlabel(x_axis_label)
    
    # results
    for i in xrange(100):
        fname = 'figure_{}.pdf'.format(i+1)
        if not os.path.exists(fname):
            plt.savefig(fname, bbox_inches='tight')
            print 'Graph saved as:',fname
            break
    
    # display graph if desired
    if show_graph:
        plt.show(block=False)
        raw_input('Press enter to close...')
        plt.close()
            

def bar_chart(data,error,labels,custom_params={}):

    # parameter assignment  
    default_params = {} 
    params = append_dicts(custom_params,default_params)

    # link figure if specified
    if params['fig'] == None: fig = plt.figure()
    else: fig = params['fig']

    # link axis if specified 
    if params['ax'] == None: ax = plt.gca()
    else: ax = params['ax']

    # fix the axes
    plt.sca(ax)



#####################
## DEFAULT METHODS ##
#####################

def default_accuracy_params():
    return {
            'title':'Predicted vs. Actual Score',
            'xlabel':'Predicted Score',
            'ylabel':'Last Observed Round of Selection',
            'fig':None,
            'ax':None,
            'skip_show':True,
            'mode':'attrition'
           }


def default_auroc_plot_params():
    return {
            'skip_show':True
           }

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
