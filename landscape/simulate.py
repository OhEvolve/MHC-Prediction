

'''
This project was started to try and determine how well we can implement regularization in our curent model
This particular script generates test landscapes and sequences at various sample sizes
Intended to be used with a downstream reconstruction script
'''

# standard libraries
import pickle
import random
import os

# nonstandard libraries
import numpy as np


# configuration
np.random.seed(42)

# main

def main():
    # main hypothesis testing
    test = GenerateLandscape()
    test.define_landscape()
    test.generate_sequences()
    print 'Finished!'
# helper functions

# generation class

class Landscape:
    def default_model(self):
        # basically every parameter defined in one dictionary
        default_params = {
                         'length':5,
                         'aa_count':6,
                         'sw_mag':1,
                         'pw_mag':0,
                         'noise_mag':0,
                         'sequence_count':2000,
                         'unique_sequence_count':None, # TODO: figure out what this does
                         'characters':'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                         'data_label':'test'
                         }
        
        # if you are ever interested in which parameters are actually needed
        self.model_parameters = default_params.keys()
        
        # apply all changes
        for key, value in default_params.items():
            setattr(self, key, value)

    # use a dictionary to update class attributes
    def update_model(self,params={}):
        # updates parameters
        for key, value in params.items():
            setattr(self, key, value)

    def __init__(self,params={}):
        self.default_model()
        self.update_model(params)

    def define_landscape(self):
        """ 
        Generates landscapes that live up to parameter specs 
        Seperate so you can recurrently call new sequences from same landscape
        """
        self.sw_landscape = np.random.normal(0,self.sw_mag,(self.length,self.aa_count))        
        #self.sw_landscape = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0]])
        self.pw_landscape = np.random.normal(0,self.pw_mag,(self.length-1,self.length-1,self.aa_count**2))        
        #self.pw_landscape = np.zeros((self.length-1,self.length-1,self.aa_count**2))        
        #self.pw_landscape[:,:,0] = 1

    def generate_sequences(self,count=None,unique_count=None):
        # Quick access to the number of sequences you want
        if count:
            self.sequence_count = count
        if unique_count:
            self.unique_sequence_count = unique_count
        
        # library size marker
        lib_size = self.aa_count**self.length
        print 'Library size:',lib_size

        # convert sequences to unique sequences (don't waste memory space)
        if self.unique_sequence_count == None: 
            spent_sequences,self.unique_sequence_count,= 0,0
            for i in xrange(self.sequence_count):
                if spent_sequences < self.sequence_count and self.unique_sequence_count < lib_size:
                    spent_sequences += float(lib_size)/(lib_size - self.unique_sequence_count) 
                    self.unique_sequence_count += 1 
                else: break 
            print '{} sampled sequences, approx. {} uniques'.format(self.sequence_count,
                                                                    self.unique_sequence_count) 
        else: 
            if self.unique_sequence_count > lib_size:
                print '{} unique sequences > {} library size, reducing...'.format(
                                self.unique_sequence_count,lib_size) 
                self.unique_sequence_count = lib_size
            print '{} declared unique sequences'.format(self.unique_sequence_count)

        raw_seqs = random.sample(range(0,lib_size),self.unique_sequence_count)
        raw_seqs = [[(a/(self.aa_count**l))%self.aa_count for l in xrange(self.length)] for a in raw_seqs]

        # check for viability
        self.check_params()

        # create sequence list
        self.sequences = [''.join([self.characters[i] for i in row]) for row in raw_seqs]
        self.scores = [sum([self.sw_landscape[ind][self.characters.index(i)] for ind,i in enumerate(row)]+
            [self.pw_landscape[ind1][ind2-1][self.aa_count*self.characters.index(i) + self.characters.index(j)] for ind1,i in enumerate(row[:-1]) for ind2,j in enumerate(row[ind1:])]) for row in self.sequences]

        # add noise to scores
        noise = np.random.normal(0,self.noise_mag,len(self.scores))
        self.scores = list(noise + np.array(self.scores))
    
        # write file
        with open('./data/{}.txt'.format(self.data_label),'w') as myfile:
            for seq,score in zip(self.sequences,self.scores):
                myfile.write('{},{}\n'.format(seq,score))


        # creates a pickled file with a map of all mutational energy landscapes
        all_landscapes = {'sw_landscape':self.sw_landscape,'pw_landscape':self.pw_landscape}
        pickle.dump(all_landscapes,open('./data/{}_map.p'.format(self.data_label),'wb'))

    # creates maps for true landscapes and project specifications
    def generate_records(self):
        print 'WARNING: generate records depreciated.'        

    # check for parameter validity prior to code execution
    def check_params(self):
        assert self.length <= len(self.characters),'Not enough characters to represent amino acid repetoire.'
     
    

# namespace identifier

if __name__ == '__main__':
    main()

# closing notes



