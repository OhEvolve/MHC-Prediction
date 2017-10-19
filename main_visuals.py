
import matplotlib.pyplot as plt

from analysis import *


### LIBRARY COMPLEXITY ###

def run_library_complexity():
    f, axes = plt.subplots(2,2)

    axes_list = [axes[0,0],axes[1,0],axes[0,1],axes[1,1]]
    label_list = ['A12','F3','226','5cc7']

    for ax,label in zip(axes_list,label_list):

        options = {'show_graph':False,
                   'axis_handle':ax,
                   'data_label':label}

        library_complexity(label,options)
        print 'Finished {}!'.format(label)

    plt.show(block = False)
    raw_input('Press enter to close...')
    plt.close()


### CLUSTERING ###

def run_library_clustering():
    label_list = ['A12']#,'F3','226','5cc7']

    for label in label_list:

        options = {
                  'show_graph':False,
                  'data_label':label
                  }

        library_clustering(options)
        print 'Finished {}!'.format(label)



### NAMESPACE CATCH ### 

#run_library_complexity()
run_library_clustering()


