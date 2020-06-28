## Copyright 2020 UT-Battelle, LLC.  See LICENSE.txt for more information.
###
# @author Narasinga Rao Miniskar, Frank Liu, Dwaipayan Chakraborty, Jeffrey Vetter
#         miniskarnr@ornl.gov
#
# Modification:
#              Baseline code
# Date:        Apr, 2020
# **************************************************************************
###
import os
import numpy as np
from doepy import build
from numpy import random
import pdb
import argparse
import shlex


class DeffeRandomSampling:
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
                 Step() - make a new step of sampling
                       rc = True if successful

                 training_seq:
                 val_seq:       return current training and validation sequences
                 testing_seq:   return remaining entries in not yet sampled as testing

    """

    def __init__(self, framework):
        self.framework = framework
        self.config = framework.config.GetSampling()
        self._seq = np.arange(1)
        self._n_train = 0
        self._n_val = 0
        self._previous_pos = 0
        self._pos = 0
        self._len = 0
        self._step = 0
        self._train_idx = []
        self._val_idx = []
        self._exhausted = False
        self._n_samples = 0
        self._shuffle = True
        self.parser = self.AddArgumentsToParser()
        self.args = self.ReadArguments()

    # Read arguments provided in JSON configuration file
    def ReadArguments(self):
        arg_string = self.config.arguments
        args = self.parser.parse_args(shlex.split(arg_string))
        return args

    # Add command line arguments to parser
    def AddArgumentsToParser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-custom-samples",
            nargs="*",
            action="store",
            dest="custom_samples",
            default="",
        )
        parser.add_argument("-fixed-samples", dest="fixed_samples", default='-1')
        parser.add_argument("-method", dest="method", default='random')
        return parser

    def GenerateSamples(self):
        selected_pruned_params = self.parameters.selected_pruned_params
        param_dict =  { param.name : param_values 
            for (param, param_values, pindex, permutation_index) in 
                selected_pruned_params }
        n_samples = self._n_samples
        max_samples = min(1000000, self._n_samples)
        sampling_method = self.args.method
        if sampling_method == 'random':
            # _n_samples is the permutation count of all parameters, which can be very high
            # Hence, generate samples of maximum 1000000 for training.
            org_seq = np.random.choice(
                n_samples, size=max_samples, replace=False)
            self._seq = org_seq
        else:
            sample_mat = None
            param_dict_actual = param_dict
            #param_dict = { k:np.array(v).astype('float').tolist() for k,v in param_dict_actual.items() }
            #param_dict_hold = { k:np.array(v).astype('float').tolist() for k,v in param_dict_actual.items() }
            param_dict = { k:[index for index in range(len(v))] for k,v in param_dict_actual.items() }
            param_dict_hold = { k:[index for index in range(len(v))] for k,v in param_dict_actual.items() }
            if sampling_method == 'frac_fact_res':
                sample_mat = build.frac_fact_res(param_dict)
            elif sampling_method == 'plackett_burman':
                sample_mat = build.plackett_burman(param_dict)
            elif sampling_method == 'box_behnken':
                sample_mat = build.box_behnken(param_dict)
            elif sampling_method == 'central_composite_ccf':
                sample_mat = build.central_composite(param_dict, face='ccf')
            elif sampling_method == 'central_composite_cci':
                sample_mat = build.central_composite(param_dict, face='cci')
            elif sampling_method == 'central_composite_ccc':
                sample_mat = build.central_composite(param_dict, face='ccc')
            elif sampling_method == 'lhs':
                sample_mat = build.lhs(param_dict, num_samples=n_samples)
            elif sampling_method == 'space_filling_lhs':
                sample_mat = build.space_filling_lhs(param_dict, num_samples=n_samples)
            elif sampling_method == 'random_k_means':
                sample_mat = build.random_k_means(param_dict, num_samples=n_samples)
            elif sampling_method == 'maximin':
                sample_mat = build.maximin(param_dict, num_samples=n_samples)
            elif sampling_method == 'halton':
                sample_mat = build.halton(param_dict, num_samples=n_samples)
            elif sampling_method == 'uniform_random':
                sample_mat = build.uniform_random(param_dict, num_samples=n_samples)
            else:
                print("[Error] Unknown method:{} of sampling!".
                        format(sampling))               
                sys.exit(1)
            np_hdrs = np.char.lower(np.array(list(sample_mat.columns)).astype("str"))
            sample_mat_val = np.round(sample_mat.values)
            sample_mat = np.zeros(sample_mat_val.shape, dtype=int)
            for i in range(sample_mat_val.shape[1]):
                id_keys = np.digitize(sample_mat_val[:, i], param_dict_hold[np_hdrs[i]], right=True)
                sample_mat[:, i] = np.array(param_dict_actual[np_hdrs[i]])[id_keys]
            sample_mat = np.unique(sample_mat, axis=0)
            n_samples = sample_mat.shape[0]
            self._n_samples = n_samples
            np_records = sample_mat.astype("str")
            def GetRecHash(rec):
                sel_param_hash = { 
                    k:rec[index]
                          for index, k in enumerate(np_hdrs) 
                    }
                return sel_param_hash
            self._seq = np.array([ 
                    self.parameters.EncodePermutation( \
                        GetRecHash(rec) \
                    ) \
                    for rec in np_records 
                    ]).astype("int")
        if self._shuffle:
            np.random.shuffle(self._seq)
        return

    def Initialize(self, parameters, n_samples, n_train, n_val, shuffle=True):
        self.custom_samples = []
        self.parameters = parameters
        self.custom_samples_index = 0
        if self.args.custom_samples != "":
            self.custom_samples = [int(s) for s in self.args.custom_samples]
        self._n_samples = n_samples
        self._shuffle = shuffle
        self.GenerateSamples()

        self._n_train = n_train
        self._n_val = n_val
        if self.args.fixed_samples != "-1":
            self._n_val = 0
        self._previous_pos = 0
        self._pos = 0
        self._len = len(self._seq)
        self._step = 0
        self._train_idx = []
        self._val_idx = []
        self._exhausted = False

        if n_val+n_train > self._len:
            n_val = int(self._len * 0.30)
            n_train = self._len - n_val
            self._n_train = n_train
            self._n_val = n_val
        assert n_train >= 1, "Bummer: number of training has to be >1"
        # assert n_val>1, 'Bummer: number of validation has to be >1'
        assert (
            n_val + n_train <= self._len
        ), "Bummer: input sequence is too small: {} + {} < {}".format(
            n_train, n_val, self._len
        )

        self._train_idx = self._seq[0 : self._n_train]
        self._val_idx = self._seq[self._n_train : self._n_train + self._n_val]
        self._pos = self._n_train + self._n_val
        # print("Training: "+str(len(self._train_idx))+" Val: "+str(len(self._val_idx)))

    def IsCompleted(self):
        return self._exhausted

    """
       Take step with increment, generate the sequence of training and validation sets
            if rc = True, sequence has not been exhausted
               rc = False, no more unexplored values in the sequnece
    """

    def StepWithInc(self, inc=1):
        if self._exhausted:
            return False
        if len(self.custom_samples) != 0:
            if self.custom_samples_index >= len(self.custom_samples):
                self._exhausted = True
                return False
            total_count = self.custom_samples[self.custom_samples_index]
            prev_count = len(self._train_idx) + len(self._val_idx)
            train_count = int(total_count * 0.7)
            val_count = total_count - train_count
            all_idx = None
            if total_count > prev_count:
                new_val = self._seq[prev_count:total_count]
                all_idx = np.concatenate(
                    (self._train_idx, self._val_idx, new_val), axis=None
                )
            else:
                all_idx = self._seq[:total_count]
            self._train_idx = all_idx[:train_count]
            self._val_idx = all_idx[train_count:]
            self.custom_samples_index = self.custom_samples_index + 1
            self._previous_pos = self._pos
            self._pos = len(self._train_idx)+len(self._val_idx)
            return True
        if self._pos >= self._len:
            self._exhausted = True
            self._previous_pos = self._pos
            return False
        new_pos = self._pos
        for i in range(inc):
            if int(self.args.fixed_samples) == -1:
                new_pos = new_pos + self._n_val
            else:
                new_pos = new_pos + int(self.args.fixed_samples)
            self._step = self._step + 1
        if new_pos >= self._len:
            new_pos = self._len
        new_val = self._seq[self._pos : new_pos]
        tmp = len(self._val_idx) // 2
        if tmp != 0:
            self._train_idx = np.concatenate(
                (self._train_idx, self._val_idx[range(0, tmp)]), axis=None
            )
            self._val_idx = np.concatenate(
                (self._val_idx[range(tmp, len(self._val_idx))], new_val), axis=None
            )
        else:
            total_count = len(self._train_idx)
            self._val_idx = self._seq[total_count:new_pos]
        self._previous_pos = self._pos
        self._pos = new_pos
        print(
            "Training: "
            + str(len(self._train_idx))
            + " Val: "
            + str(len(self._val_idx))
        )
        return True

    @property
    def training_seq(self):
        return self._train_idx

    @property
    def val_seq(self):
        return self._val_idx

    @property
    def testing_seq(self):
        if len(self._seq) == 0:
            return []
        return np.setdiff1d(
            self._seq,
            np.concatenate((self._train_idx, self._val_idx), axis=None),
            assume_unique=True,
        )

    def GetBatch(self):
        samples = self.training_seq.tolist()+self.val_seq.tolist()
        return samples

    def GetNewBatch(self):
        samples = self.training_seq.tolist()+self.val_seq.tolist()
        return samples[self._previous_pos:]

def run_test1():
    print("Test 1, n_train=2, n_val=4")
    seq = np.array(range(10, 50))
    s1 = SampleSeqGenerator(seq, 2, 4, shuffle=False)
    print("Training set: {}".format(s1.training_seq))
    print("Val set: {}".format(s1.val_seq))
    print("Testing set: {}".format(s1.testing_seq))

    s1.step()

    print("After one step")
    print("Training set: {}".format(s1.training_seq))
    print("Val set: {}".format(s1.val_seq))
    print("Testing set: {}".format(s1.testing_seq))

    pass


def run_test2():
    print("Test 2, n_train=3, n_val=5, exhaustive search, w/ shuffling")
    seq = np.array(range(10, 50))
    s1 = SampleSeqGenerator(seq, 3, 5)
    counter = 0
    print("=== Step {}".format(counter))
    print("Training set: {}".format(s1.training_seq))
    print("Val set: {}".format(s1.val_seq))
    print("Testing set: {}".format(s1.testing_seq))
    counter = counter + 1

    while s1.step():
        print("=== Step {}".format(counter))
        print("Training set: {}".format(s1.training_seq))
        print("Val set: {}".format(s1.val_seq))
        print("Testing set: {}".format(s1.testing_seq))
        counter = counter + 1


def GetObject(framework):
    obj = DeffeRandomSampling(framework)
    return obj


if __name__ == "__main__":
    run_test1()
    run_test2()
    pass
