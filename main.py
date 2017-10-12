
#!/usr/bin/env python

'''
Main execution for spiNN

Utilizes command line arguments to execute different functions 

'''


# standard libraries
import os
from sys import argv
import itertools
import pickle
import time

# nonstandard libraries
from multiprocessing import Pool

# homegrown libraries
from testing import * # libraries: model_test 
from models import * # libraries: kNN,cNN,spiNN
from processing import * # libraries: pre, post
from landscape import * # libraries: simulate,parameterize_data (TODO: remove latter)

def main(args):
    for command in args['commands']:

        # cNN testing
        if command == 'c':
            print 'Generated data testing protocol starting...'
            
            test = simulate.Landscape()
            test.define_landscape()
            test.generate_sequences()

            model_params = default_cnn_model_params()
            model_params['data_label'] = test.data_label
            model_params['data_label'] = 'A12'
            model_params['learning_rate'] = 0.001
            model_params['num_epochs'] = 50
            model_params['model_type'] = 'cnn'
        
            results_fname = single_model.test(model_params)


        # Fake data testing
        if command == 'f':
            print 'Generated data testing protocol starting...'

            test = simulate.Landscape()
            test.define_landscape()
            test.generate_sequences()

            model_params = default_spinn_model_params()
            model_params['data_label'] = test.data_label
            results_fname = single_model.test(model_params)
                        
            post_params= default_post_params() 
            post_params['results_fname'] = results_fname 
            post_params['mode'] = 'enrichment'
            post_params['data_label'] = 'Test data' 
            post.single(post_params,graphing=False) 

        # Testing protocol
        if command == 't':
            print 'Testing protocol starting...'
            model_params = default_spinn_model_params()
            model_params['data_label'] = 'A12'
            single_model.test(model_params)
             
        # Model build test
        # TODO: make this into a test script
        if command == 'b':
            model_params = default_spinn_model_params()
            if model_params['model_type'] == 'spinn':
                print 'Starting spiNN model...'
                m = spiNN.BuildModel(model_params)

        if command == 'p':
            print 'Results analytics protocol starting...'

            '''
            results_dict = {'A12': 'results_100016.p',
                            'F3':  'results_100024.p',
                            '5cc7':'results_100025.p',
                            '226': 'results_100023.p'}
            '''

            results_dict = {'test':'results_100035.p'}

            for k,v in results_dict.items():
                post_params= default_post_params() 
                post_params['results_fname'] = v
                post_params['data_label'] = k
                post.single(post_params,graphing=False) 

        if command == 'd':
            print 'Data processing protocol starting...'

            model_params = default_spinn_model_params()
            model_params['num_epochs'] = 25 
            pre_params = default_pre_params()
            pre_params['mode'] = 'attrition'

            for sets in ['A12']:
            #for sets in ['5cc7']:
            #for sets in ['A12','F3','5cc7','226']:

                print 'Starting data set {}...'.format(sets)

                model_params['data_label'] = sets
                pre_params['data_label'] = sets
                
                # run scripts
                pre.start(pre_params) # pre-process datasets
                single_model.test(model_params) # single model test for given data

        """ GRID SEARCH [g] """
        if command == 'g': 
            print 'Grid search protocol starting...'

            # generate a list of dictionaries to test model on 
            all_dicts = default_grid_search_dicts()
            all_dicts_count = len(all_dicts)

            # search for whether simulations are completed or not
            log_dir = './logs'
            all_files = [log_dir+'/'+i for i in os.listdir(log_dir) 
                    if os.path.isfile(os.path.join(log_dir,i)) and 'results_' in i]

            all_found_dicts = []
            for f in all_files:
                try: all_found_dicts.append(pickle.load(open(f,'r'))['model_settings'])
                except ValueError: print 'Skipped:',f

            #all_found_dicts = [ for f in all_files]
            all_dicts = [d for d in all_dicts if not d in all_found_dicts]

            # give user update on how many tests have already been performed
            print 'Found {} completed tasks!'.format(len(all_found_dicts))
            print 'Reduced task count from {} to {}.'.format(all_dicts_count,len(all_dicts))
            time.sleep(1.0)
            
            from multiprocessing import Pool
            pool = Pool(processes=8) # creates a worker pool equal to the number of cores
            # fn: single_model.test(params)
            # all_dicts: [params1,params2,...]
            tasks = pool.map_async(single_model.test,all_dicts).get(9999999)
            tasks.wait()
            
            print 'Finished!'

        """ GRID SEARCH ANALYSIS [h] """
        if command == 'h':
            print 'Grid search analysis protocol starting...'

            # create a default library
            post_params = default_post_params()
            
            # tell model which dictionaries to look out for
            all_dicts = default_grid_search_dicts()
            post.batch(post_params, all_dicts, graphing = True)

             
            

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

def append_dicts(old_dict,*dict_args):
    """
    Given an original dictionary and any number of additional dicts, adds
    new entries to original if they are not already present
    """
    for dictionary in dict_args:
        for key in dictionary.keys():
            if not key in old_dict.keys():
                #print 'Adding parameter: {} -> {}'.format(key,dictionary[key])
                old_dict[key] = dictionary[key]
    return old_dict

""" Generates dictionaries that exhaustively cover parameter space in given dicts """
def generate_dicts(*args):
    # compile all dictionaries into one location
    all_params = {}
    for arg in args:
        assert type(arg) == dict, 'Argument type not supported'
        for k,v in arg.items(): 
            assert type(v) in (tuple,list), 'Dictionary value not supported: {},{}'.format(type(v),v)
            all_params[k] = v
                
    # generate parameter combinations 
    dict_list = []
    for params_set in list(itertools.product(*all_params.values())):
        dict_list.append(dict([(k,v) for k,v in zip(all_params.keys(),params_set)]))
    
    # return resulting dictionaries
    return dict_list 

""" Returns grid search dictionaries """
def default_grid_search_dicts():
    
    model_params = default_spinn_model_params()
    model_params['repeats'] = 3
    model_params['num_epochs'] = 25 
    model_params['silent'] = True
       
    base_params = { 
                  'sw_pw_ratio':[0.0,0.5,1.0],
                  'pw_depth':[1,2,4],
                  'sw_depth':[1,2,4],
                  'data_label':['A12','F3','5cc7','226']
                  }

    d1_params = { 'fc_layers':[1], 'fc_depth':[(1,)], 
            'fc_fn':[('linear',),('relu',),('sigmoid',)],
            'fc_dropout':[(1.0,)] }
    d2_params = { 'fc_layers':[2], 'fc_depth':[(2,1),(4,1),(8,1)], 
            'fc_fn':[('linear','linear'),('relu','relu'),('sigmoid','sigmoid')], 
            'fc_dropout':[(1.0,1.0)] }
    d3_params = { 'fc_layers':[3], 'fc_depth':[(4,2,1),(16,4,1),(64,8,1)], 
            'fc_fn':[('linear','linear','linear'),('relu','relu','relu'),
                ('sigmoid','sigmoid','sigmoid')], 'fc_dropout':[(1.0,1.0,1.0)] }

    all_dicts = generate_dicts(base_params,d1_params) +  \
                generate_dicts(base_params,d2_params) +  \
                generate_dicts(base_params,d3_params) 
    all_dicts = [append_dicts(d,model_params) for d in all_dicts]
    return all_dicts


""" Returns pre-processing parameters """
def default_pre_params():
    return {}
        
""" Returns pre-processing parameters """
def default_post_params():
    return {}
        

""" Returns model parameters """
def default_spinn_model_params():
    return { 
        # testing settings
        'repeats':1,
        # simulation settings
        'data_augment':False,
        'data_normalization': False, # probably definitely change back to True
        'silent': False,
        'test_fraction': 0.2,
        'batch_size':256,
        'num_epochs':20,
        'learning_rate':0.5,
        'data_label':'test',
        # overall network parameters
        'model_type':'spinn',
        'sw_pw_ratio':0.5, # relative weight of sw branch relative to pw branch (range -> [0,1])
        # sitewise/pairwise parameters
        'pw_depth':2,
        'sw_depth':4,
        # fully connected parameters
        'fc_layers':1,
        'fc_depth':(1,),
        'fc_fn':('linear',),
        'fc_dropout':(1.0,),
        #'fc_depth':(8,1),
        #'fc_fn':('linear','linear'),
        #'fc_dropout':(1.0,1.0),
        # loss parameters
        'loss_type':'l2',
        'loss_magnitude':1.0,
        # regularization parameters
        'reg_type':'l2',
        'reg_magnitude':0.1
        }

def default_cnn_model_params():
    return  {
             'learning_rate':0.01,
             'data_normalization': False,
             'silent': False,
             'test_fraction': 0.1,
             'batch_size':100,
             'num_epochs':50,
             'loss_coeff':0.01,
             'learning_rate':0.01,
             'data_label':'test',
             # overall network parameters
             'cnn_layers':1, # includes conv & pool
             'fc_layers':2,
             # convolutional parameters
             'conv_stride':(1,1), # not sure if we can handle others
             'conv_filter_width':(2,4),
             'conv_depth':(2,4),
             'conv_padding':('SAME','SAME'),
             # max pool parameters
             'pool_stride':(2,2),
             'pool_filter_width':(3,3),
             'pool_padding':('SAME','SAME'),
             # fully connected parameters
             'fc_depth':(16,1),
             'fc_fn':('sigmoid','sigmoid','linear'),
             'fc_dropout':(0.5,1.0),
             # loss parameters
             'loss_type':'l2',
             'loss_magnitude':1.0,
             # regularization parameters
             'reg_type':'l2',
             'reg_magnitude':0.01,
            }

""" namespace catch if called as script (which it should) """ 
if __name__ == "__main__":
    args = getopts(argv)
    main(args)
else:
    print "main.py not intended to be loaded as a module."




