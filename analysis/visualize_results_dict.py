
""" 
file: /analysis/model_assessment.py 
function: takes results dictionary, tries to display in a visually pleasing way
author: PVH
"""

# standard libraries

# nonstandard libraries
import matplotlib.pyplot as plt

def visualize_results_dict(results, *args, **kwargs):
     
    """ Takes a dictionary, looks for entries to visualize """ 
    
    # allow user to modify particular parameters
    options = {
              'silent':False
              } 

    for arg in args: options.update(arg) # update with input dictionaries
    options.update(kwargs) # update with keyword arguments

    # iterate through each parameter in this dictionary 

    for k,w in results.items():

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1) 

        ### AUROC VISUALS ### 
        if k == 'auroc':

            colors = ['r','b']

            for i,aurocs in enumerate([w['testing'],w['training']]):
                x = [i+1 for _ in xrange(len(aurocs))]
                plt.scatter(x,aurocs,color=colors[i])


            plt.xticks([1,2],['testing','training'])

            plt.show(block=True)
            raw_input('Press enter to close...')
            plt.close()

