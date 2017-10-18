
"""
file: /analysis/library_clustering.py
function: displays the complexity of any input datafile 
author: PVH
"""

# standard libraries
import os
import csv
import random
import math

# nonstandard libraries
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly.graph_objs import *
import networkx as nx

# homemade libraries
from simulation.input import *

""" Tests a single library """

def library_clustering(*args,**kwargs):

    # allow user to modify particular parameters
    options = {'silent':False,
               'data_label':'A12',
               }

    for arg in args: options.update(arg) # update with input dictionaries
    options.update(kwargs) # update with keyword arguments

    # reassignments
    label = options['data_label']

    # make sure files are in appropriate paths 
    if not os.path.isdir('./data/{}'.format(label)): 
        raise IOError('Directory (/data/{}) does not exist!'.format(label))
    if not os.path.isfile('./data/{}/{}.csv'.format(label,label)): 
        raise IOError('File (/data/{}/{}.csv) does not exist!'.format(label,label))

    # load data 
    with open('./data/{}/{}.csv'.format(label,label),'rb') as csvfile:
        reader = csv.reader(csvfile)
        data = [[r[0]]+map(int,r[1:]) for r in reader] 

    # cluster each round in the same figure
    seqs_info = []
    for seq in data:
        if count_nonzero(seq[1:]) < 2: continue
        i = 1
        for s in seq[1:]:
            if s == 0: break
            else: i += 1 
        seqs_info.append([seq[0],math.log10(1 + sum(seq[1:])),])

    seqs = [s[0] for s in seqs_info]
       
    print 'Total sequences:',len(seqs_info)

    # heavy plotly stuff

    G = nx.Graph()
    G.add_nodes_from(range(len(seqs)))
    
    for i,seq in enumerate(seqs):
        edges = []
        for j,seq2 in enumerate(seqs):
            h = hamdist(seq,seq2)
            if h > 0 and h < 9:
                edges.append((i,j))

        G.add_edges_from(edges)


    nx.draw(G)
    plt.show()

    raw_input('wait')

# FACTORY METHODS

def hamdist(str1, str2,prevMin=None):
    diffs = 0
    if len(str1) != len(str2):
      return max(len(str1),len(str2))
    for ch1, ch2 in zip(str1, str2):
      if ch1 != ch2:
          diffs += 1
          if prevMin is not None and diffs>prevMin:
              return None
    return diffs

def count_nonzero(my_list):
    return sum(1 for x in my_list if (x != 0))






