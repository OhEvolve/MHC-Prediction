
""" 
file: /simulation/training/train.py 
function: holds all major training methods
author: PVH
"""

# standard libraries
import time

# nonstandard libraries
import numpy as np

# homegrown libraries
from common_methods import *

### FACTORY METHODS ###
def ceildiv(a, b):
    ''' ceiling division ''' 
    return -(-a // b)

def train_nn(model,data,**kwargs):
    # allow user to modify particular parameters
    options = {'silent':False,
               'learning_rate':1.0,
               'learning_mode':'gradientdescent',
               'loss_type':'l2',
               'loss_magnitude':1,
               'reg_type':'l2',
               'reg_magnitude':0.1,
               'batch_size':10,
               'num_epochs':20}

    options.update(kwargs) # update options with user input

    # if data is submitted not as a dictionary, change
    if type(data) in [tuple,list] and len(data) == 2: data = {'training': data}

    # check assertions 
    model_reqs = ['loss','reset_variables','train_x','train_y','sess'] 
    assert all([a in dir(model) for a in model_reqs]), 'Model missing necessary attributes'
    assert len(data['training'][0]) == len(data['training'][1]),'Training data not balanced'

    # start timer
    start = time.time()

    # create train step
    train_step = set_learning_fn(
            model.loss,options['learning_rate'],options['learning_mode'])

    # map parameters into training and testing data initializations
    sample_count = len(data['training'][0])
    batches_per_epoch = ceildiv(sample_count,options['batch_size'])
    num_steps = options['num_epochs']*batches_per_epoch

    # alert user
    if not options['silent']: 
        print 'Batchs per epoch - {} / Number of steps - {}'.format(batches_per_epoch,num_steps)

    step_loss = [] # log storage
    step,learning_mod = 0,1.0

    epoch_loss,validation_loss = 0.,None
    finished = False

    while not finished: 
        offset = (step % batches_per_epoch) * options['batch_size']

        # split data
        batch_x = data['training'][0][offset:(offset + options['batch_size'])]
        batch_y = data['training'][1][offset:(offset + options['batch_size'])]

        # train and log batch loss
        feed_dict = {model.train_x: batch_x, model.train_y: batch_y}
        _,batch_loss = model.sess.run([train_step,model.loss],feed_dict=feed_dict)
        epoch_loss += batch_loss

        if (step % batches_per_epoch == 0) and not step == 0:
            
            if 'testing' in data.keys(): # calculate testing set loss
                feed_dict = {model.train_x: data['testing'][0], model.train_y: data['testing'][1]}
                validation_loss = model.sess.run(model.loss,feed_dict=feed_dict)/len(data['testing'][0])

            epoch_loss /= sample_count

            step_loss.append((step,epoch_loss,validation_loss))
            
            # Gives readout on model
            if not options['silent']: 
                print 'Step {}: Batch loss ({})  /  Validation loss ({})'.format(
                        step,epoch_loss,validation_loss)

            epoch_loss = 0
            
            # randomize input data
            seed = np.random.randint(1,1000000) # pick a random seed
            for ind in xrange(2):
                np.random.seed(seed) # set identical seeds
                np.random.shuffle(data['training'][ind]) # shuffle data in place

        # add one to step 
        step += 1

        # create early exist conditions
        if step >= num_steps: # check if at threshold of learning steps
            finished = True
        if np.isnan(batch_loss): # check if training has spiraled into NaN space
            learning_mod *= 0.5
            init,step = model.reset_variables(),0
            model.sess.run(init)
            train_step = set_learning_fn(
                    model.loss,learning_mod*options['learning_rate'],options['learning_mode'])
            if not options['silent']: print 'Lowering learning rate and restarting...'

    print '[FINAL] Epoch loss ({})  /  Validation loss ({}) / Training time ({} s)'.format(
            epoch_loss,validation_loss,time.time() - start)


    if not options['silent']: print 'Finished!'
    
    # returns logs of stepwise losses if needed later for saving model performance
    return step_loss
        


def train_knn(data,**kwargs):
    # allow user to modify particular parameters
    options = {'silent':False,
               'fold_index':1,
               'fold_count':5}
    options.update(kwargs) # update options with user input

    # make some assertions about what the user options are now
    pass
