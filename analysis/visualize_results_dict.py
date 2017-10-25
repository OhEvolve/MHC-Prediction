
""" 
file: /analysis/model_assessment.py 
function: takes results dictionary, tries to display in a visually pleasing way
author: PVH
"""

def visualize_results_dict(results, *args, **kwargs):
     
    """ Takes a dictionary, looks for entries to visualize """ 
    
    # allow user to modify particular parameters
    options = {
              'silent':False
              } 

    for arg in args: options.update(arg) # update with input dictionaries
    options.update(kwargs) # update with keyword arguments

    # iterate through each parameter in this dictionary 

    for k,w in results:
        
        ### AUROC VISUALS ### 
        if k == 'auroc':

            colors = ['r','b']

            for i,aurocs in enumerate([w['testing'],w['training']):
                x1 = [i+1 for _ in len(aurocs)]
                plt.xticks(x1,['testing','training']
                plt.scatter(x,aurocs,colors[i])

                plt.show(block=True)
                raw_input('Press enter to close...')
                plt.close()

