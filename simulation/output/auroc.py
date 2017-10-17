
""" 
file: /output/auroc.py 
function: takes a model and input data, and outputs an auROC
author: PVH
"""

def auroc_score(model,data,**kwargs)
    results = auroc(model,data,**kwargs)
    return results[0] 

def auroc_coordinates(model,data,**kwargs)
    results = auroc(model,data,**kwargs)
    return results[1]

def auroc(model,data):
    """
    A clean, fast callable function to return an auROC score for two lists comparing
    guesses to actual values
    """
   
    """ Tests a single collection of parameters """ 
    
    # allow user to modify particular parameters
    options = {
              'silent': False,
              }

    options.update(kwargs) # update with keyword arguments
    
    guess = model.predict(data[0])
    actual = data[1]

    # change analog input to binary if not previously defined this way

    median = 0.5 # np.median(actual)
    actual = [1 if c > median else 0 for c in actual] 
    pos_inc,neg_inc = 1./sum(actual),1./(len(actual)-sum(actual)) 

    # order the inputs by guesses
    pairs = sorted([(g,a) for g,a in zip(guesses,actual)],key=lambda x: -x[0])
    coor,score = [(0.,0.)],0.

    # iterate across each point to trace the curve
    for p in pairs:
        if p[1] == 0: score += neg_inc*coor[-1][1]
        coor.append((coor[-1][0] + neg_inc*(1-p[1]),coor[-1][1] + pos_inc*p[1]))

    # return finalized auROC score (with coordinates if requested)
    return (score,coor) 


