from __future__ import absolute_import, division, print_function, unicode_literals
from workload_excel import *
import socket
import time
import tensorflow as tf
import pdb
from keras.utils import plot_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import dendrogram, linkage
from classification_network import *
from single_node_network import *
from sam_seq_gen import SampleSeqGenerator
import glob

tf.keras.backend.clear_session()  # For easy reset of notebook state.

from tensorflow import keras
from tensorflow.keras import layers

def SetDevice(args):
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

nepochs = 50
def GetEpochs():
    global nepochs
    return nepochs
batch_size = 256
def GetBatchSize():
    return batch_size

hidden_layer_nodes = 256
def GetHiddenLayerNodes():
    return hidden_layer_nodes


#for i in ['100', '1000', '5000', '10000', '50000']:
#for the below kmeans distributions of number of parameters
# No. of parameters: simulation records count 
#100: 35807
#1000: 35833
#5000: 2464
#10000: 2434
#50000: 1360

#If we exclude #1000 objects from training, the test validation of #1000 results are with ~10% accuracy 
#If we exclude #5000 objects from training, the test validation of #5000 results are with ~81% accuracy

def main(args):
    SetDevice(args)
    workload = Workload()
    samples_count = int(args.n)
    workload.ReadPreEvaluatedWorkloads(args.input, samples_count)
    desired_data = None
    if workload.IsHdrExist('cpu'):
        group_data = workload.GroupsData(['cpu'])
        desired_data = group_data['TimingSimpleCPU']
    else:
        desired_data = workload.GetData()
    cost_name = 'cpu_cycles'
    if not workload.IsHdrExist(cost_name):
        cost_name = 'obj_fn_1'
    sorted_data = workload.SortData(desired_data, workload.GetHdrIndex(cost_name))
    learning_data = sorted_data
    print("Total learning records:"+str(len(learning_data)))
    if args.non_excluded_data or args.excluded_data:
        exclude_hash = {}
        all_excludes = re.split(r',', args.exclude)
        for one_exclude in all_excludes:
            exclude_args = re.split(r'::', one_exclude)
            exclude_hash[exclude_args[0]] = exclude_args[1:]
        excluded_data, non_excluded_data = workload.ExcludeData(exclude_hash, sorted_data)
        print("Total excluded data:"+str(args.exclude)+"->"+str(len(excluded_data)))
        print("Total non-excluded data:"+str(len(non_excluded_data)))
        group_data = workload.GroupsData(['options'])
        for group in group_data.keys():
            print("Group-"+str(group)+": "+str(len(group_data[group])))
        if args.non_excluded_data:
            learning_data = non_excluded_data
        else:
            learning_data = excluded_data
    print("Model data given to network:"+str(len(learning_data)))
    data = np.array(learning_data)
    #np.random.shuffle(data)
    data = data.tolist()
    #parameters_data = np.array(workload.Get2DDataWithIndexing(workload.param_indexes, data, {workload.headers_index_hash['options'] : 1})).astype('int')
    #parameters_data = np.array(workload.Get2DDataWithNormalization(workload.param_indexes, data, {workload.headers_index_hash['options'] : 1})).astype('float')
    parameters_data = np.array(workload.Get2DDataWithNormalization(workload.param_indexes, data, {})).astype('float')
    orig_cost_data = np.array(workload.Get2DData([cost_name], data)).astype('float')
    cost_data =      np.array(workload.Get2DData([cost_name], data)).astype('float')
    if args.cost_mantessa:
        mant_orig = orig_cost_data/pow(2, np.log2(orig_cost_data).astype(int))
        mant = cost_data/pow(2, np.log2(cost_data).astype(int))
        orig_cost_data = mant_orig
        cost_data = mant
    if args.cost_exponent:
        mant_orig = np.log2(orig_cost_data).astype(int)
        mant = np.log2(cost_data).astype(int)
        orig_cost_data = mant_orig
        cost_data = mant
    if args.cost_mantessa_exp:
        mant_orig = orig_cost_data/pow(2, np.log2(orig_cost_data).astype(int))
        mant = np.exp(cost_data/pow(2, np.log2(cost_data).astype(int)))
        orig_cost_data = mant_orig
        cost_data = mant
    print("Parameters shape:"+str(parameters_data.shape))
    print("Cost shape:"+str(cost_data.shape))
    if args.test_dir_evaluate != '':
        network = SingleOutputNetwork(args, parameters_data, cost_data, orig_cost_data)
        if not args.load_train_test:
            network.preprocess_data()
        all_files = glob.glob(os.path.join(args.test_dir_evaluate, "*.hdf5"))
        network.evaluate_model(all_files, args.output)
    elif args.incremental_learning:
        total_samples = parameters_data.shape[0]
        sample_size = 100    ## start small
        n_train = sample_size
        n_val = 2*sample_size
        sam_gen = SampleSeqGenerator(np.arange(total_samples), n_train, n_val)
        inc = int(args.il_step)
        for step in range(0, 30):
            if step != 0:
                sam_gen.step_with_inc(inc)
            if args.step_start != '':
                if step < int(args.step_start):
                    continue
            if args.step_end != '':
                if step > int(args.step_end):
                    continue
            train_idx, val_idx, test_idx = sam_gen.training_seq, sam_gen.val_seq, sam_gen.testing_seq
            print("*********** Step:{} train:{} val:{} test:{} *********".format(step, len(train_idx), len(val_idx), len(test_idx)))
            network = SingleOutputNetwork(args, parameters_data, cost_data, orig_cost_data)
            network.DisableICP()
            network.preprocess_data_incremental(step, train_idx, val_idx, test_idx)
            last_cp = ''
            if step == int(args.step_start) and args.icp != '':
                network.load_model(args.icp)
            if args.take_last_cp and step != int(args.step_start):
                all_files = glob.glob(os.path.join(".", "step{}-*.hdf5".format(step-1)))
                last_cp = network.get_last_cp_model(all_files)
                if last_cp != '':
                    network.load_model(last_cp)
            network.run_model()
            print("Read checkpoints\n")
            all_files = glob.glob(os.path.join(".", "step{}-*.hdf5".format(step)))
            stepoutfile = "step{}-test-output.csv".format(step)
            network.evaluate_model(all_files, stepoutfile)
    else:
        network = SingleOutputNetwork(args, parameters_data, cost_data, orig_cost_data)
        network.preprocess_data()
        network.run_model()
    print("Writing model into file:"+os.path.join(os.getcwd(), args.ocp))


if __name__ == "__main__":
    print("Current directory: "+os.getcwd())
    print("Machine: "+socket.gethostname())
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', dest='n', default="-1")
    parser.add_argument('-input', dest='input', default="output.csv")
    parser.add_argument('-train-test-split', dest='train_test_split', default="0.70")
    parser.add_argument('-validation-split', dest='validation_split', default="0.20")
    parser.add_argument('-icp', dest='icp', default="")
    parser.add_argument('-ocp', dest='ocp', default="weights.best.hdf5")
    parser.add_argument('-exclude', dest='exclude', default="options::5000")
    parser.add_argument('-epochs', dest='epochs', default=str(nepochs))
    parser.add_argument('-batch-size', dest='batch_size', default=str(GetBatchSize()))
    parser.add_argument('-transfer-learning', dest='transfer_learning', default="-1")
    parser.add_argument('-convs', dest='convs', default="2")
    parser.add_argument('-no-run', dest='no_run', action='store_true')
    parser.add_argument('-excluded-data', dest='excluded_data', action='store_true')
    parser.add_argument('-non-excluded-data', dest='non_excluded_data', action='store_true')
    parser.add_argument('-evaluate-only', dest='evaluate', action='store_true')
    parser.add_argument('-incremental-learning', dest='incremental_learning', action='store_true')
    parser.add_argument('-take-last-cp', dest='take_last_cp', action='store_true')
    parser.add_argument('-il-step', dest='il_step', default='1')
    parser.add_argument('-cost-mantessa-exp', dest='cost_mantessa_exp', action='store_true')
    parser.add_argument('-cost-mantessa', dest='cost_mantessa', action='store_true')
    parser.add_argument('-cost-exponent', dest='cost_exponent', action='store_true')
    parser.add_argument('-load-train-test', dest='load_train_test', action='store_true')
    parser.add_argument('-plot-loss', dest='plot_loss', action='store_true')
    parser.add_argument('-cpu', dest='cpu', action='store_true')
    parser.add_argument('-loss', dest='loss', default='')
    parser.add_argument('-step-start', dest='step_start', default='')
    parser.add_argument('-step-end', dest='step_end', default='')
    parser.add_argument('-test-dir-evaluate', dest='test_dir_evaluate', default='')
    parser.add_argument('-gpu', dest='gpu', default="0")
    parser.add_argument('-nodes', dest='nodes', default=str(GetHiddenLayerNodes()))
    parser.add_argument('-output', dest='output', default='test-output.csv')
    parser.add_argument('-last-layer-nodes', dest='last_layer_nodes', default="32")
    args = parser.parse_args()
    nepochs = int(args.epochs)
    start = time.time()
    main(args)
    lapsed_time = "{:.3f} seconds".format(time.time() - start)
    print("Total runtime of script: "+lapsed_time)
