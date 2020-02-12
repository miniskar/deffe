"""
  sam_seq_gen.py: This is the sampling sequence generator
"""
import numpy as np
from numpy import random

class SampleSeqGenerator():
    """
      SamplingSeqGenerator:  generate training and validation sequences
             Init   :  raw sequence, initial # training, # validation per step, shuffle or not
           
             Initial Sample :  first 'n_train' as training seq
                               next 'n_val' as validation seq

             Following Steps:
                       source next 'n_val' in sequence as the new_val_seq

                       new_train_seq = train_seq + first half of val_seq
                       new_val_seq = second half of val_seq + new_val_seq

                       till the end of input sequence has been reached

             Methods:
                 step() - make a new step of sampling
                       rc = True if successful

                 training_seq:
                 val_seq:       return current training and validation sequences
                 testing_seq:   return remaining entries in not yet sampled as testing

    """
    def __init__(self, org_seq, n_train, n_val, shuffle=True):
        self._seq = org_seq
        self._n_train = n_train
        self._n_val = n_val
        self._pos = 0
        self._len = len(org_seq)
        self._step = 0
        self._train_idx = []
        self._val_idx = []
        self._exhausted = False
        
        assert n_train>1, 'Bummer: number of training has to be >1'
        assert n_val>1, 'Bummer: number of validation has to be >1'
        assert n_val+n_train<=self._len, 'Bummer: input sequence is too small: {} + {} < {}'.format(
            n_train, n_val, self._len)
        
        if shuffle:
            np.random.shuffle( self._seq )

        self._train_idx = self._seq[0:self._n_train]
        self._val_idx = self._seq[self._n_train: self._n_train+self._n_val]
        self._pos = self._n_train + self._n_val

    """
       Take one step, generate the sequence of training and validation sets
            if rc = True, sequence has not been exhausted
               rc = False, no more unexplored values in the sequnece
    """
    def step_with_inc(self, inc):
        for i in range(inc):
            self.step()

    def step(self):
        if self._exhausted:
            return False
        
        new_pos = self._pos + self._n_val
        if new_pos > self._len:
            print('Insufficient remaining samples in sequence, truncating num of new samples from {} to {}'.format(
                self._n_val, new_pos-self._len ))
            new_pos = self._len
            self._exhausted = True

        new_val = self._seq[ self._pos: new_pos ]
        tmp = len(self._val_idx)//2
        self._train_idx = np.concatenate((self._train_idx, self._val_idx[range(0,tmp)]), axis=None)
        self._val_idx = np.concatenate((self._val_idx[range(tmp,len(self._val_idx))], new_val), axis=None)
        self._pos = new_pos
        self._step = self._step + 1 
        return True
    
    @property 
    def training_seq(self):
        return( self._train_idx )

    @property
    def val_seq(self):
        return( self._val_idx )

    @property
    def testing_seq(self):
        return( np.setdiff1d(self._seq, np.concatenate((self._train_idx, self._val_idx),axis=None), assume_unique=True ))
    

def run_test1():
    print('Test 1, n_train=2, n_val=4')
    seq = np.array( range(10,50) )
    s1 = SampleSeqGenerator( seq, 2,4, shuffle=False)
    print('Training set: {}'.format( s1.training_seq ))
    print('Val set: {}'.format( s1.val_seq ))
    print('Testing set: {}'.format( s1.testing_seq ))

    s1.step()
        
    print('After one step')
    print('Training set: {}'.format( s1.training_seq ))
    print('Val set: {}'.format( s1.val_seq ))
    print('Testing set: {}'.format( s1.testing_seq ))
    
    pass

def run_test2():
    print('Test 2, n_train=3, n_val=5, exhaustive search, w/ shuffling')
    seq = np.array( range(10,50) )
    s1 = SampleSeqGenerator( seq, 3,5)
    counter = 0
    print('=== Step {}'.format( counter))
    print('Training set: {}'.format( s1.training_seq ))
    print('Val set: {}'.format( s1.val_seq ))
    print('Testing set: {}'.format( s1.testing_seq ))
    counter = counter + 1
    
    while s1.step():
        print('=== Step {}'.format( counter))
        print('Training set: {}'.format( s1.training_seq ))
        print('Val set: {}'.format( s1.val_seq ))
        print('Testing set: {}'.format( s1.testing_seq ))
        counter = counter+1
    
    
if __name__ == "__main__":
    run_test1()
    run_test2()
    pass
