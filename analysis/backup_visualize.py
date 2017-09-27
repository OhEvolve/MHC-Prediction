

"""

This was the visualization method before removal of auroc (I dont think its used but im not sure)

"""


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
    if params['fig'] == None: 
        print 'I SHOULDNT BE HERE'
        fig = plt.figure()
    else: fig = params['fig']

    # link axis if specified 
    if params['ax'] == None: ax = plt.gca()
    else: ax = params['ax']

    # fix the axes
    plt.sca(ax)

    # TODO: make params argument
    actual_with_noise = actual + np.random.uniform(low=-1./(2*rounds),high=1./(2*rounds),size=actual.shape)

    plt.scatter(guesses,actual_with_noise,c=actual,s=1,cmap='copper')
    
    # fix the x-axis
    for i in xrange(1,rounds):
        plt.plot([0, 1], [(float(i)/(rounds-1)) - (1./(2*(rounds-1))) for _ in xrange(2)], 'k--')

    ytick_labels = ['R'+str(i+1) for i in xrange(rounds)]
    ytick_values = [(float(i)/(rounds-1)) for i in xrange(rounds)]
    plt.yticks(ytick_values,ytick_labels)

    plt.title(params['title'])
    plt.xlabel(params['xlabel'])
    plt.ylabel(params['ylabel'])

    if not params['skip_show']:
        plt.show(block=False)
        raw_input('Press enter to close...')
        plt.close()
    
def auroc_plot(guesses,actual,labels = [],graph = True,silent=False):
    """
    Uses a set of guesses and actual values, produces a predictive auROC
    Optional graphing argument
    """
    if len(actual) != len(guesses):
        if not silent: print 'Note: Assumed first entry of real values apply to all guesses...'
        actual = [actual for i in xrange(len(guesses))]
             
    if not labels:
        labels = ['Method {}'.format(i+1) for i in xrange(len(guesses))]
    
    guesses = [np.reshape(np.array([(guess - min(guess))/
                                (max(guess)-min(guess))]),(len(guess),1)) for guess in guesses]
    
    actual = [np.reshape(np.array([(a - min(a))/
                                (max(a)-min(a))]),(len(a),1)) for a in actual]
    
    bin_actual = [[1 if c >= np.median(a) else 0 for c in a] for a in actual]
    
    lims = np.arange(1.,0.,-0.01)
    
    x,y = [],[]

    for guess,bin_a in zip(guesses,bin_actual):
        x.append([])
        y.append([])
        for lim in lims:
            # check for set 1
            tpr,fpr = 0.,0.
            bin_guess = [1 if g > lim else 0 for g in guess]
            for g,t in zip(bin_guess,bin_a):
                if g == 1:
                    if g == t: tpr += 1
                    else: fpr += 1
            x[-1].append(fpr/(len(bin_a)-sum(bin_a)))
            y[-1].append(tpr/sum(bin_a))

    '''
    print 'Start 1:'
    for x,y in zip(x1,y1):
        print x,y
    print 'Start 2:'
    for x,y in zip(x2,y2):
        print x,y
    '''

    scores = []
    for i,j,z in zip(x,y,xrange(len(x))):
        scores.append(1.-np.trapz(i,j))
        print 'Method {} auROC: {}'.format(z+1,scores[-1])    
    
    if graph:
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        for i,j,l in zip(x,y,labels):
            plt.plot(i,j,label=l)

        plt.xlim((0.,1.))
        plt.ylim((0.,1.))

        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
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
