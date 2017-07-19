

'''
Project: Neural Network for MHC Peptide Prediction
Class(s): BuildNetwork
Function: Generates specified neural network architecture (data agnostic)

Author: Patrick V. Holec
Date Created: 2/2/2017
Date Updated: 5/18/2017
'''


'''
Library Importation
'''

# standard libraries

# nonstandard libraries

# homegrown libraries


'''
Default Function
'''

def main():
    """
    If file called as script, this function will run
    """
    print 'Finished!'


'''
Factory Methods
'''

def default_model():
    return {
             'data_augment':False,
             'learning_rate':0.01,
             'data_normalization': False,
             'silent': False,
             'test_fraction': 0.1,
             'batch_size':100,
             'num_epochs':50,
             'loss_coeff':0.01,
             'learning_rate':0.1,
             'data_label':'test',
             # overall network parameters
             'fc_layers':2,
             'sw_pw_ratio':0.5, # relative importance of sw branch relative to pw branch (range -> [0,1])
             # sitewise/pairwise parameters
             'pw_depth':1,
             'sw_depth':1,
             # fully connected parameters
             'fc_depth':(16,1),
             'fc_fn':('sigmoid ','sigmoid','linear'),
             'fc_dropout':(1.0,1.0),
             # loss parameters
             'loss_type':'l2',
             'loss_magnitude':1.0,
             # regularization parameters
             'reg_type':'l2',
             'reg_magnitude':0.01
             # logging parameters
           }


def append_dicts(old_dict,*dict_args):
    """
    Given an original dictionary and any number of additional dicts, adds
    new entries to original if they are not already present
    """
    for dictionary in dict_args:
        for key in dictionary.keys():
            if not key in old_dict.keys():
                print 'Adding parameter: {} -> {}'
                old_dict[key] = dictionary[key]
    return old_dict


'''
Main Methods
'''

def grid_search(defaults,parameters,silent=False):
            
    

'''
Default Routine
'''

if __name__ == '__main__':
    main()




