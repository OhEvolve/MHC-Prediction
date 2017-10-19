
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
import igraph as ig
import plotly.plotly as py
from plotly.graph_objs import *

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
    seqs_temp = []
    for seq in data:
        if count_nonzero(seq[1:]) < 3: continue
        seqs_temp.append([seq[0],6*math.log(1. + sum(seq[1:])),count_nonzero(seq[1:])])

    info = {
            'seqs':[s[0] for s in seqs_temp],
            'read_count':[s[1] for s in seqs_temp],
            'last_round':[s[2] for s in seqs_temp]
           }

    # heavy plotly stuff

    edges,edge_width = [],[]
    for i,seq in enumerate(info['seqs']):
        for j,seq2 in enumerate(info['seqs']):
            h = hamdist(seq,seq2)
            if h > 0 and h < 2:
                edges.append((i,j))
                edge_width.append(h)
    
    print 'Edge count:', len(edges)
    print 'Sequence count:',len(info['seqs'])
   
    #graph2D(label,info,edges)
    graph3D(label,info,edges)

def graph3D(label,info,edges): 
    # Start putting together graph
    labels,group = info['seqs'],info['last_round']
    N = len(info['seqs'])

    G = ig.Graph(edges,directed = False) 
    G.add_vertices(N)

    layt=G.layout('kk', dim=3)

    Xn=[layt[k][0] for k in range(N)]# x-coordinates of nodes
    Yn=[layt[k][1] for k in range(N)]# y-coordinates
    Zn=[layt[k][2] for k in range(N)]# z-coordinates

    Xe,Ye,Ze = [],[],[]

    for e in edges:
        Xe+=[layt[e[0]][0],layt[e[1]][0], None]# x-coordinates of edge ends
        Ye+=[layt[e[0]][1],layt[e[1]][1], None]
        Ze+=[layt[e[0]][2],layt[e[1]][2], None]

    marker = Marker(symbol='dot',size=info['read_count'],color=group,
            colorscale='Viridis',line=Line(color='rgb(50,50,50)',width=0.5),
            opacity=0.0)

    trace1 = Scatter3d(x=Xe,y=Ye,z=Ze,
                mode='lines',line=Line(color='rgb(0,0,0)', width=2),
                hoverinfo='none')
    trace2 = Scatter3d(x=Xn,y=Yn,z=Zn,mode='markers',name='actors',
                marker=marker,text=labels,hoverinfo='text')

    axis=dict(showbackground=False,
              showline=False,
              zeroline=False,
              showgrid=False,
              showticklabels=False,
              title='')

    layout = Layout(title="Data: {}".format(label),width=1000,height=1000,showlegend=False,
                scene=Scene(xaxis=XAxis(axis),yaxis=YAxis(axis),zaxis=ZAxis(axis)),
                margin=Margin(t=100),hovermode='closest')

    # plot this formatting trash
    data=Data([trace1, trace2])
    fig=Figure(data=data, layout=layout)
    py.plot(fig, filename='Data: {}'.format(label),auto_open=False)
   
    resolution = 200
    return None
    
    for i in xrange(resolution):
        
        if os.path.isfile('./video/{}_{}.png'.format(label,'%05d' % i)): continue

        x = 3*math.cos(2*3.1415*float(i)/resolution)
        y = 3*math.sin(2*3.1415*float(i)/resolution)
        z = 0 

        camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=x, y=y, z=z))

        fig['layout'].update(
            scene=dict(camera=camera),
            )

        if not os.path.exists('./video'): os.makedirs('./video')
        py.image.save_as(fig, filename='./video/{}_{}.png'.format(label,'%05d' % i))
        print 'Finished image {}!'.format(i+1)



    print 'Finished saving image!'

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





