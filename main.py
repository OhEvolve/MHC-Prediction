
#!/usr/bin/env python

'''
Main execution for spiNN

Utilizes command line arguments to execute different functions 

'''


# standard libraries
from sys import argv

# nonstandard libraries

# homegrown libraries
from testing import * # libraries: model_test 
from models import * # libraries: kNN,cNN,spiNN
from processing import * # libraries: pre, post
from landscape import * # libraries: simulate,parameterize_data (TODO: remove latter)

def main(args):
    for command in args['commands']:
        # Fake data testing
        if command == 'f':
            print 'Generated data testing protocol starting...'
            test = simulate.Landscape()
            test.define_landscape()
            test.generate_sequences()

            model_params = default_model_params()
            model_params['data_label'] = test.data_label
            single_model.test(model_params)

        # Testing protocol
        if command == 't':
            print 'Testing protocol starting...'
            model_params = default_model_params()
            single_model.test(model_params)
             
        # Model build test
        # TODO: make this into a test script
        if command == 'b':
            model_params = default_model_params()
            if model_params['model_type'] == 'spinn':
                print 'Starting spiNN model...'
                m = spiNN.BuildModel(model_params)

        if command == 'p':
            print 'Results analytics protocol starting...'

            results_dict = {'A12': 'results_100016.p',
                            'F3':  'results_100024.p',
                            '5cc7':'results_100025.p',
                            '226': 'results_100023.p'}

            for k,v in results_dict.items():
                post_params= default_post_params() 
                post_params['results_fname'] = v
                post_params['data_label'] = k
                post.start(post_params,graphing=True) 

        if command == 'd':
            print 'Data processing protocol starting...'

            model_params = default_model_params()
            model_params['num_epochs'] = 5 
            pre_params = default_pre_params()
            pre_params['mode'] = 'attrition'

            #for sets in ['A12','F3','5cc7','226']:
            for sets in ['A12']:

                print 'Starting data set {}...'.format(sets)
                

                model_params['data_label'] = sets
                pre_params['data_label'] = sets
                
                # run scripts
                pre.start(pre_params) # pre-process datasets
                single_model.test(model_params) # single model test for given data

                 
                 


""" Collects command line arguments """ 
# note: could accept keywords as opposed to key characters if support is desired
def getopts(argv):
    opts = {'commands':[]} # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            if argv[0][1] == '-': # Pulls in a new option if double-dashed
                opts[argv[0][:argv[0].index('=')]] = argv[0][argv[0].index('=')+1:]
            else: # Otherwise loads boot commands
                opts['commands'].append(argv[0][1:])
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    # place a translator for keyworks if desired
    opts['commands'] = ''.join(opts['commands'])
    return opts

""" Returns pre-processing parameters """
def default_pre_params():
    return {}
        
""" Returns pre-processing parameters """
def default_post_params():
    return {}
        

""" Returns model parameters """
def default_model_params():
    return { 
        # testing settings
        'repeats':3,
        # simulation settings
        'data_augment':False,
        'data_normalization': True,
        'silent': True,
        'test_fraction': 0.2,
        'batch_size':1000,
        'num_epochs':50,
        'learning_rate':1,
        'data_label':'A12',
        # overall network parameters
        'model_type':'spinn',
        'fc_layers':2,
        'sw_pw_ratio':0.25, # relative weight of sw branch relative to pw branch (range -> [0,1])
        # sitewise/pairwise parameters
        'pw_depth':1,
        'sw_depth':1,
        # fully connected parameters
        'fc_depth':(8,1),
        'fc_fn':('linear','linear'),
        'fc_dropout':(1.0,1.0),
        # loss parameters
        'loss_type':'l2',
        'loss_magnitude':1.0,
        # regularization parameters
        'reg_type':'l1',
        'reg_magnitude':0.00001
        }

""" namespace catch if called as script (which it should) """ 
if __name__ == "__main__":
    args = getopts(argv)
    main(args)
else:
    print "main.py not intended to be loaded as a module."
