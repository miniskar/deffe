## Copyright 2020 UT-Battelle, LLC.  See LICENSE.txt for more information.
###
 # @author Narasinga Rao Miniskar, Frank Liu, Dwaipayan Chakraborty, Jeffrey Vetter
 #         miniskarnr@ornl.gov
 # 
 # Modification:
 #              Baseline code
 # Date:        Apr, 2020
 #**************************************************************************
###
import numpy as np
import re
import pdb

checkpoint_dir = "checkpoints"
class BaseMLModel:
    def __init__(self):
        None

    def PreLoadData(self, step, parameters, cost_data, orig_cost_data, train_test_split, validation_split):
        train_count = int(parameters.shape[0]*train_test_split)
        test_count = parameters.shape[0] - train_count
        print("Init Total count:"+str(parameters.shape[0]))
        print("Init Train count:"+str(train_count))
        print("Init Test count:"+str(test_count))
        self.train_count = train_count
        self.val_count = int(train_count * validation_split)
        self.test_count = test_count
        indices = np.random.permutation(parameters.shape[0])
        training_idx = indices[:train_count]
        test_idx = indices[train_count:]
        #print("Tr_indices:"+str(training_idx))
        #print("test_indices:"+str(test_idx))
        #print("Tr_indices count:"+str(training_idx.size))
        #print("test_indices count:"+str(test_idx.size))
        self.training_idx = training_idx
        self.test_idx = test_idx
        x_train = parameters[training_idx,:].astype('float')
        y_train = cost_data[training_idx,:].astype('float')
        z_train = orig_cost_data[training_idx,:].astype('float') 
        x_test = parameters[test_idx,:].astype('float')
        y_test = cost_data[test_idx,:].astype('float')
        z_test = orig_cost_data[test_idx,:].astype('float') 
        self.x_train, self.y_train, self.z_train = x_train, y_train, z_train
        self.x_test, self.y_test, self.z_test = x_test, y_test, z_test

    def save_train_test_data(self, step=-1):
        if step == -1:
            print("Saving train indices: train-indices.npy")
            np.save("train-indices.npy", self.training_idx)
            print("Saving test indices: train-indices.npy")
            np.save("test-indices.npy", self.test_idx)
        else:
            print("Saving train indices: step{}-train-indices.npy".format(step))
            np.save("step{}-train-indices.npy".format(step), self.training_idx)
            print("Saving test indices: step{}-test-indices.npy".format(step))
            np.save("step{}-test-indices.npy".format(step), self.test_idx)

    def load_train_test_data(self, step=-1):
        if step == -1:
            print("Loading train indices: train-indices.npy")
            training_idx = np.load("train-indices.npy")
            print("Loading test indices: test-indices.npy")
            test_idx     = np.load("test-indices.npy")
        else:
            print("Loading train indices: step{}-train-indices.npy".format(step))
            training_idx = np.load("step{}-train-indices.npy".format(step))
            print("Loading test indices: step{}-test-indices.npy".format(step))
            test_idx     = np.load("step{}-test-indices.npy".format(step))
        parameters = self.parameters_data
        cost_data = self.cost_data
        orig_cost_data = self.orig_cost_data
        x_train = parameters[training_idx,:].astype('float')
        y_train = cost_data[training_idx,:].astype('float')
        z_train = orig_cost_data[training_idx,:].astype('float') 
        x_test = parameters[test_idx,:].astype('float')
        y_test = cost_data[test_idx,:].astype('float')
        z_test = orig_cost_data[test_idx,:].astype('float') 
        self.x_train, self.y_train, self.z_train = x_train, y_train, z_train
        self.x_test, self.y_test, self.z_test = x_test, y_test, z_test
        self.train_count = len(training_idx) * (1-self.validation_split)
        self.val_count = len(training_idx) * self.validation_split
        self.test_count = len(test_idx)

    def get_last_cp_model(self, all_files):
        epoch_re = re.compile(r'weights-improvement-([0-9]+)-')
        max_epoch = 0
        last_icp = ''
        for index, icp_file in enumerate(all_files):
            epoch_flag = epoch_re.search(icp_file)
            epoch=0 #loss0.4787-valloss0.4075.hdf5a
            if epoch_flag:
                epoch = int(epoch_flag.group(1)) 
            if epoch > max_epoch:
                max_epoch = epoch
                last_icp = icp_file
        return last_icp

    # Evalaute model results
    def EvaluateModel(self, all_files, outfile="test-output.csv"):
        None



