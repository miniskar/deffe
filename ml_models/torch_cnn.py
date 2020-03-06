from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pdb
import re
import io
from contextlib import redirect_stdout
from tqdm import tqdm, trange
import shlex
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import logging
from torch_wl_cnn import *
from datetime import datetime
from torch_model_eval import *
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


from baseml import *

checkpoint_dir = "checkpoints"
def mean_squared_error(y_actual, y_predicted):
    return torch.mean(torch.square(y_actual - y_predicted), axis=-1)

def mean_squared_error_int(y_actual, y_predicted):
    return torch.mean(torch.square((y_actual) - (y_predicted)), axis=-1)

def custom_mean_abs_loss(y_actual, y_predicted):
    error_sq = torch.abs(y_predicted-y_actual)/y_actual
    return torch.mean(error_sq)

def custom_mean_abs_exp_loss(y_actual, y_predicted):
    error_sq = torch.abs(torch.exp(y_predicted)-y_actual)/y_actual
    return torch.mean(error_sq)

def custom_mean_abs_log_loss(y_actual, y_predicted):
    error_sq = torch.abs(y_predicted-torch.log(y_actual))/torch.log(y_actual)
    return torch.mean(error_sq)

class TorchCNN(BaseMLModel):
    def __init__(self, framework):
        self.framework = framework
        self.args = self.ParseArguments()
        self.step = -1
        self.step_start = framework.args.step_start
        self.step_end= framework.args.step_end
        self.epochs = int(self.args.epochs)
        if framework.args.epochs != '-1':
            self.epochs = int(framework.args.epochs)
        self.nodes = int(self.args.nodes)
        self.convs = int(self.args.convs)
        self.validation_split = float(self.args.validation_split)
        self.tl_freeze_layers = self.args.tl_freeze_layers
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def ParseArguments(self):
        arg_string = self.framework.config.GetModel().arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('-train-test-split', dest='train_test_split', default="1.00")
        parser.add_argument('-validation-split', dest='validation_split', default="0.20")
        parser.add_argument('-icp', dest='icp', default="")
        parser.add_argument('-epochs', dest='epochs', default="50")
        parser.add_argument('-batch-size', dest='batch_size', default="256")
        parser.add_argument('-tl_freeze_layers', dest='tl_freeze_layers', default="-1")
        parser.add_argument('-convs', dest='convs', default="2")
        parser.add_argument('-tl-samples', dest='tl_samples', action='store_true')
        parser.add_argument('-no-run', dest='no_run', action='store_true')
        parser.add_argument('-evaluate-only', dest='evaluate', action='store_true')
        parser.add_argument('-load-train-test', dest='load_train_test', action='store_true')
        parser.add_argument('-plot-loss', dest='plot_loss', action='store_true')
        parser.add_argument('-loss', dest='loss', default='')
        parser.add_argument('-nodes', dest='nodes', default="256")
        parser.add_argument('-last-layer-nodes', dest='last_layer_nodes', default="32")
        parser.add_argument('-real-objective', dest='real_objective', action='store_true')
        args = parser.parse_args(shlex.split(arg_string))
        return args

    def initialize_nn(self):
        """
           Model
        """
        network_topo = [8, 64, 32, 16]
        nn = torch_wl_cnn_gen(network_topo)
        ## convoluted way to capture the network structure to string
        with io.StringIO() as buf, redirect_stdout(buf):
            print(nn)
            nn_struct = buf.getvalue()
        logging.info('Network Structure:')
        print(nn_struct)
        logging.info('{}'.format( nn_struct))
        return nn, network_topo

    def Initialize(self, step, headers, parameters_data, cost_data, name="network"):
        args = self.args
        self.step = step
        print("Headers: "+str(headers))
        self.parameters_data = parameters_data
        self.cost_data = cost_data
        self.orig_cost_data = cost_data 
        self.n_out_fmaps = 1
        #self.nodes = 128
        self.train_count = 0
        self.val_count = 0 
        self.test_count = 0 
        self.n_in_fmaps = parameters_data.shape[1]
        last_layer_nodes = int(args.last_layer_nodes)
        self.name = name
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        log_file = 'train_{}.log'.format(dt_string)
        
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', filename=log_file)
        
        print('Log filename {}'.format( log_file))
        logging.info('Using device {}'.format(self.device))
        self.loss_function = custom_mean_abs_exp_loss
        if args.loss == 'custom_mean_abs_loss':
            self.loss_function = custom_mean_abs_loss
        elif args.loss == 'mean_squared_error':
            self.loss_function = mean_squared_error 
        elif args.loss == 'mean_squared_error_int':
            self.loss_function = mean_squared_error_int
        elif args.loss == 'custom_mean_abs_log_loss':
            self.loss_function = custom_mean_abs_log_loss
        batch_size = int(args.batch_size)
        if self.framework.args.batch_size != '-1':
            batch_size = int(self.framework.args.batch_size)
        self.batch_size = min(self.parameters_data.shape[0], batch_size)
        self.disable_icp = False
        self.nn, self.network_topo = self.initialize_nn()
        summary(self.nn, (1, parameters_data.shape[1]))

    def GetEpochs(self):
        return self.epochs

    def GetBatchSize(self):
        return self.batch_size

    def SetTransferLearning(self, count):
        self.tl_freeze_layers = count

    def DisableICP(self, flag=True):
        self.disable_icp = flag

    def preprocess_data(self):
        BaseMLModel.preprocess_data(self, self.parameters_data, self.cost_data, self.orig_cost_data, self.args.train_test_split, self.validation_split)
        if self.args.tl_samples and self.step != self.step_start:
            all_files = glob.glob(os.path.join(checkpoint_dir, "step{}-*.hdf5".format(self.step-1)))
            last_cp = self.get_last_cp_model(all_files)
            if last_cp != '':
                self.load_model(last_cp)

    def evaluate(self, x_test, y_test, z_test, tags=""):
        if x_test.size == 0:
            return
        print("***********************************************")
        print('\n# Generate predictions for 3 samples of '+tags)
        predictions = self.model.predict(x_test, batch_size=self.GetBatchSize())
        print(predictions[0], np.log(y_test[0]), np.exp(predictions[0]), y_test[0], z_test[0])
        print(predictions[1], np.log(y_test[1]), np.exp(predictions[1]), y_test[1], z_test[1])
        print(predictions[2], np.log(y_test[2]), np.exp(predictions[2]), y_test[2], z_test[2])
        print('predictions shape:', predictions.shape)
        estimation_error = []
        recalc_error = []
        mean_square = False
        for sample_idx in range(len(y_test)):
            nn_out = np.exp(predictions[sample_idx])
            # recalc_val = np.exp(y_test[sample_idx])
            sim_val = y_test[sample_idx]

            cycle_diff = np.float(np.abs(sim_val-nn_out))
            cycle_error = cycle_diff/sim_val
            if mean_square:
                cycle_error = cycle_diff * cycle_diff
            estimation_error.append(cycle_error)

            # recalc_error_val = np.float(np.abs(sim_val-recalc_val))/sim_val
            # recalc_error.append(recalc_error)
        print(tags+" Mean: ", np.mean(estimation_error), "Max: ", np.max(estimation_error), "Min: ", np.min(estimation_error))
        #print(tags+" MeanAct: ", np.mean(estimation_act_error), "MaxAct: ", np.max(estimation_act_error), "MinAct: ", np.min(estimation_act_error))

    def Inference(self):
        return None

    def load_model(self, model_name):
        print("Loading the checkpoint: "+model_name)
        #TODO load model weights
        #keras.losses.custom_loss = self.loss_function
        #self.model.load_weights(model_name)

    def Train(self):
        x_train, y_train, z_train = self.x_train, self.y_train, self.z_train
        x_test, y_test, z_test    = self.x_test, self.y_test, self.z_test
        n_train = x_train.shape[0]
        train_loader = DataLoader(TensorDataset(torch.Tensor(x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))), torch.Tensor(y_train)), batch_size=self.GetBatchSize(), shuffle=True, num_workers=4, pin_memory=True)
        #test_loader = DataLoader(TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test)), batch_size=self.GetBatchSize(), shuffle=True, num_workers=4, pin_memory=True)
        test_loader = None
        val_loader = train_loader
        n_val = x_train.shape[0]
        #x_train_torch = torch.from_numpy(x_train).reshape([x_train.shape[0], 1, [x_train.shape[2]]) 
        tr_comment = '{}_INPUT_{}_PARAM_{}_{}_{}_STEP_{}_SIZE_{}'.format(
            self.name, self.network_topo[0],self.network_topo[1],self.network_topo[2],self.network_topo[3], self.step, x_train.shape[0]+x_test.shape[0])
        print(tr_comment)
        writer = SummaryWriter(comment=tr_comment)
        logging.info('STEP={}, n_train={}, n_val={}, total={}'.format( self.step, n_train, n_val, n_train+n_val))
        #save_indices(dt_string, self.step, train_idx, val_idx, test_idx)
        lr = 0.0010
        momentum = 0.9
        global_step = 0
        g_model = self.nn
        optimizer = optim.SGD(g_model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-8)
        criterion = nn.MSELoss()
        n_epochs= self.epochs
        val_acc = 0.0
        epoch_test = 5
        for epoch in range(n_epochs):
            g_model.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch+1, n_epochs), unit='sample', ascii=True, dynamic_ncols=True) as pbar:
                for batch in train_loader:
                    features = batch[0] #.reshape([x_train.shape[0], 1, x_train.shape[1]])
                    ground_truth = batch[1]
                    features = features.to(device=self.device, dtype=torch.float32)
                    ground_truth = ground_truth.to(device=self.device, dtype=torch.float32)
                    g_model = g_model.to(device=self.device)
                    pred = g_model(features)
                    if self.args.real_objective:
                        loss = custom_mean_abs_loss(y_predicted=pred, y_actual=ground_truth)
                    else:
                        loss = criterion(pred, ground_truth)
                    epoch_loss += loss.item()
                    writer.add_scalar('Loss/train', loss.item(), global_step)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    global_step += 1
                    pbar.set_postfix(**{'batch training loss': loss.item(), 'batch val loss':val_acc})
                    pbar.update( features.shape[0] )
                #print("Loss: "+str(epoch_loss))
                val_acc = 0.0
                if n_val > 0:
                    val_acc = model_eval(self.args, g_model, val_loader, self.device, n_val)
                full_details = False
                #acc = model_eval(g_model, test_loader, device, n_test)
                if full_details and test_loader != None:
                    test_acc = model_eval(self.args, g_model, test_loader, self.device, n_test)
                    val_acc_hash = model_compute_error(self.args, g_model, val_loader, self.device, n_val)
                    test_acc_hash = model_compute_error(self.args, g_model, test_loader, self.device, n_test)
                    logging.info('Step {0:d} Epoch {1:d} Training Loss {2:9.4f} Validation MSE Loss {3:9.4f} Test MSE Loss {4:9.4f} Validation Avg Loss {5:9.4f} Test Avg Loss {6:9.4f} Validation Min Loss {7:9.4f} Test Min Loss {8:9.4f} Validation Max Loss {9:9.4f} Test Max Loss {10:9.4f}'.format( self.step, epoch, epoch_loss, val_acc, test_acc, val_acc_hash['avg'], test_acc_hash['avg'], val_acc_hash['min'], test_acc_hash['min'], val_acc_hash['max'], test_acc_hash['max']))
                else:
                    logging.info('Step {0:d} Epoch {1:d} Training Loss {2:9.4f} Validation Loss {3:9.4f}'.format( self.step, epoch, epoch_loss, val_acc))
                writer.add_scalar('Loss/validation', val_acc, global_step)

                if epoch+1==n_epochs and test_loader != None:
                    acc = model_compute_error(self.args, g_model, test_loader, self.device, n_test)
                    logging.info('Step {0:d} Testing accuracy: max_error(%)={1:.4f}, min_error={2:.4f}, avg_error={3:.4f}, std={4:.4f}, num_val={5}'.
                                format(self.step, acc['max'], acc['min'], acc['avg'], acc['std'], n_test))
                    pass
        weight_file = 'weights_{}.pth'.format( dt_string )
        logging.info('Saving weights to {}'.format( weight_file ))
        torch.save(g_model.state_dict(), weight_file)
        return (0.0, 0.0)

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

    def evaluate_model(self, all_files, outfile="test-output.csv"):
        epoch_re = re.compile(r'weights-improvement-([0-9]+)-')
        loss_re = re.compile(r'-loss([^-]*)-')
        valloss_re = re.compile(r'-valloss(.*)\.hdf5')
        #step2-train350-val350-weights-improvement-610-loss0.1988-valloss0.1341.hdf5
        step_re = re.compile(r'step([^-]*)-')
        traincount_re = re.compile(r'-train([^-]*)-')
        valcount_re = re.compile(r'val([0-9][^-]*)-')
        with open(outfile, "w") as fh:
            hdrs = [ "Epoch", "TrainLoss", "ValLoss", "TestLoss" ]
            print("Calculating test accuracies "+str(len(all_files)))
            for index, icp_file in enumerate(all_files):
                epoch_flag = epoch_re.search(icp_file)
                loss_flag = loss_re.search(icp_file)
                valloss_flag = valloss_re.search(icp_file)
                step_flag = step_re.search(icp_file)
                traincount_flag = traincount_re.search(icp_file)
                valcount_flag = valcount_re.search(icp_file)
                epoch=0 #loss0.4787-valloss0.4075.hdf5a
                train_loss = 0.0
                val_loss = 0.0
                traincount = 0
                valcount = 0
                step = -1 
                if epoch_flag:
                    epoch = int(epoch_flag.group(1)) 
                if loss_flag:
                    train_loss = float(loss_flag.group(1)) 
                if valloss_flag:
                    val_loss = float(valloss_flag.group(1)) 
                if step_flag:
                    step = int(step_flag.group(1))
                if traincount_flag:
                    traincount = int(float(traincount_flag.group(1)))
                if valcount_flag:
                    valcount = int(float(valcount_flag.group(1)))
                self.model.load_weights(icp_file)
                if self.args.load_train_test:
                    self.load_train_test_data(step)
                x_test, y_test, z_test    = self.x_test, self.y_test, self.z_test   
                keras.losses.custom_loss = self.loss_function
                loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
                if fh != None:
                    if index == 0:
                        if step_flag:
                            hdrs.append("Step")
                            hdrs.append("TrainCount")
                            hdrs.append("ValCount")
                        fh.write(", ".join(hdrs)+"\n")
                    data = [str(epoch), str(train_loss), str(val_loss), str(loss)]
                    if step_flag:
                        data.extend([str(step), str(traincount), str(valcount)])
                    fh.write(", ".join(data)+"\n")
                #print('Testing epoch:{} train_loss: {}, val_loss: {}, test_loss: {}\n'.format(epoch, train_loss, val_loss, loss))
            fh.close()


def GetObject(framework):
    obj = TorchCNN(framework)
    return obj

