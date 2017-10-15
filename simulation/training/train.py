
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


def train_nn(model,data,**kwargs):
    # allow user to modify particular parameters
    options = {'silent':False,
               'loss_type':'l2',
               'loss_magnitude':1,
               'reg_type':'l2',
               'reg_magnitude':0.1,
               'batch_size':10,
               'num_epochs':20}

    options.update(kwargs) # update options with user input

    # make some assertions about what the user options are now

    # start timer
    start = time.time()

    batches_per_epoch = len(train_data)/options['batch_size']
    num_steps = options['num_epochs']*batches_per_epoch

    if not options['silent']: 
        print 'Batchs per epoch - {} / Number of steps - {}'.format(batches_per_epoch,num_steps)

    step_loss = []
    epoch_loss,step,learning_rate_mod = 0,0,1.0
    finished = False

    while not finished: 
        offset = (step * self.batch_size) % (train_data.shape[0] - self.batch_size)

        # split data
        batch_x = train_data[0][offset:(offset + options['batch_size'])]
        batch_y = train_data[1][offset:(offset + options['batch_size'])]

        # train and log batch loss
        feed_dict = {model.train_x: batch_x, model.train_y: batch_y}
        _,batch_loss = self.sess.run([self.train_step,self.loss],feed_dict=feed_dict)
        epoch_loss += batch_loss

        if (step % batches_per_epoch == batches_per_epoch - 1):
            
            epoch_loss /= 0.01*batches_per_epoch*self.batch_size
            
            feed_dict = {self.train_x: test_data[0], self.train_y: test_labels[1]}
            batch_loss_validation = self.sess.run(self.loss,feed_dict=feed_dict)
            batch_loss_validation /= 0.01*self.test_data.shape[0]

            step_loss.append((step,epoch_loss,batch_loss_validation))
            
            # Gives readout on model
            if not self.silent: 
                print 'Step {}: Batch loss ({})  /  Validation loss ({})'.format(step,epoch_loss,batch_loss_validation)

            epoch_loss = 0
            
            # randomize input data
            seed = np.random.randint(1,1000000) # pick a random seed
            np.random.seed(seed) # set identical seeds
            np.random.shuffle(self.train_data) # shuffle data in place
            np.random.seed(seed) # set identical seeds
            np.random.shuffle(self.train_labels) # shuffle data in place

            '''
            together = np.concatenate((self.train_data,self.train_labels),axis=1)
            np.random.shuffle(together)
            self.train_data = together[:,:-1]
            self.train_labels = np.reshape(together[:,-1],(self.train_labels.shape[0],1)) # need to add dimension to data
            '''
        # add one to step 
        step += 1

        # create early exist conditions
        if step >= num_steps: # check if at threshold of learning steps
            finished = True
        if np.isnan(batch_loss): # check if training has spiraled into NaN space
            step = 0
            learning_rate_mod *= 0.5
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.set_learning_fn(self.learning_rate*learning_rate_mod)
            print 'Lowering learning rate and restarting...'

    print '[FINAL] Epoch loss ({})  /  Validation loss ({}) / Training time ({} s)'.format(epoch_loss,batch_loss_validation,time.time() - start)

    if not self.silent: print 'Finished!'
    
    # stores logs of stepwise losses if needed later for saving model performance
    self.step_index,self.step_loss = step_index,step_loss 
        


def train_knn(data,**kwargs):
    # allow user to modify particular parameters
    options = {'silent':False,
               'fold_index':1,
               'fold_count':5}
    options.update(kwargs) # update options with user input

    # make some assertions about what the user options are now
