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

checkpoint_dir = "checkpoints"
class BaseMLModel:
    def __init__(self):
        None

    def preprocess_data(self, parameters, cost_data, orig_cost_data, train_test_split, validation_split):
        train_count = int(parameters.shape[0]*float(train_test_split))
        test_count = parameters.shape[0] - train_count
        print("Total count:"+str(parameters.shape[0]))
        print("Train count:"+str(train_count))
        print("Test count:"+str(test_count))
        self.train_count = train_count
        self.val_count = int(train_count * validation_split)
        self.test_count = test_count
        indices = np.random.permutation(parameters.shape[0])
        training_idx = indices[:train_count]
        test_idx = indices[train_count:]
        np.save("train-indices.npy", training_idx)
        np.save("test-indices.npy", test_idx)
        #print("Tr_indices:"+str(training_idx))
        #print("test_indices:"+str(test_idx))
        #print("Tr_indices count:"+str(training_idx.size))
        #print("test_indices count:"+str(test_idx.size))
        x_train = parameters[training_idx,:].astype('float')
        y_train = cost_data[training_idx,:].astype('float')
        z_train = orig_cost_data[training_idx,:].astype('float') 
        x_test = parameters[test_idx,:].astype('float')
        y_test = cost_data[test_idx,:].astype('float')
        z_test = orig_cost_data[test_idx,:].astype('float') 
        self.x_train, self.y_train, self.z_train = x_train, y_train, z_train
        self.x_test, self.y_test, self.z_test = x_test, y_test, z_test

    def load_train_test_data(self, step=-1):
        if step == -1:
            training_idx = np.load("train-indices.npy")
            test_idx     = np.load("test-indices.npy")
        else:
            training_idx = np.load("step{}-train-indices.npy".format(step))
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



