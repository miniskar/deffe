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
import os
import pdb
import re
import sys
import numpy as np
import pandas as pd
from multi_thread_run import *
from deffe_utils import *
import argparse
import shlex

''' DeffeEvaluate class to evaluate the batch of samples with multi-thread execution environment either with/without 
    the help of slurm
    '''
class DeffeEvaluate:
    # Constructor
    def __init__(self, framework):
        self.framework = framework
        self.config = framework.config.GetEvaluate()
        self.batch_size = int(self.config.batch_size)
        self.counter = 0
        self.fr_config = self.framework.fr_config
        self.preload_data = []
        self.preload_header = []
        self.parameters = self.framework.parameters
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
        return parser
    
    # Get valid preloaded data. 
    def GetValidPreloadedData(self):
        trans_data_flag = np.full(shape=trans_data.shape, fill_value=False)
        pruned_list_indexes = []
        for pdata in pruned_param_list:
            (param, param_values, pindex) = pdata
            pruned_list_indexes.append(pindex)
            for pvalue in param_values:
                trans_data_flag[pindex] = trans_data_flag[pindex] | (trans_data[pindex] == pvalue)
        for pdata in pruned_param_list:
            (param, param_values, pindex) = pdata
            if pindex not in pruned_list_indexes:
                trans_data_flag[pindex] = np.full(shape=trans_data_flag[pindex].shape, fill_value=True)
        trans_data_flag = trans_data_flag.transpose()
        valid_indexes = []
        for index, tdata in enumerate(trans_data_flag):
            if tdata.all():
                valid_indexes.append(index)
        #TODO

    # Initialize method should be called for every new instance of new batch of samples.
    # Parameters to be passed: Parameters list, Pruned parameters list, Cost metrics names, and also 
    # if any preload_file (Pre-Evaluated results)
    def Initialize(self, param_list, pruned_param_list, cost_list, preload_file):
        self.param_data_hash = {}
        self.param_extract_indexes = []
        self.unused_params_values = []
        self.param_list = param_list
        self.cost_list = cost_list
        param_hdrs = []
        param_hash = {}
        cost_hash = {}
        for index, cost in enumerate(self.cost_list):
            cost_hash[cost] = index
        for pdata in self.param_list:
            (param, param_values, pindex) = pdata
            param_hdrs.append(param.name.lower())
            param_hash[param.name.lower()] = pdata
            param_hash[param.map.lower()] = pdata
        self.param_hdrs = param_hdrs
        self.param_hash = param_hash
        if preload_file == None:
            return
        pd_data = pd.read_csv(preload_file, dtype='str', delimiter=r'\s*,\s*', engine='python')
        np_data = pd_data.values.astype('str')
        np_hdrs = np.char.lower(np.array(list(pd_data.columns)).astype('str'))
        preload_data = np_data[1:]
        trans_data = preload_data.transpose()
        self.np_param_valid_indexes = []
        self.np_param_hdrs = []
        self.np_cost_valid_indexes = []
        self.np_cost_hdrs = []
        for index, hdr in enumerate(np_hdrs):
            if hdr in param_hash:
                self.np_param_valid_indexes.append(index)
                self.np_param_hdrs.append(hdr)
                param = self.param_hash[hdr][0]
                param_values = list(tuple(trans_data[index]))
                is_numbers = self.framework.parameters.IsParameterNumber(param_values)
                if is_numbers:
                    minp = np.min(trans_data[index].astype('float'))
                    maxp = np.max(trans_data[index].astype('float'))
                    #print("MinP: "+str(minp)+" maxP:"+str(maxp)+" name:"+param.map)
                    self.framework.parameters.UpdateMinMaxRange(param, minp, maxp)
            if hdr in cost_hash:
                self.np_cost_hdrs.append(hdr)
                self.np_cost_valid_indexes.append(index)
        #self.GetValidPreloadedData(trans_data)
        self.param_data = trans_data[self.np_param_valid_indexes,].transpose()
        self.cost_data = trans_data[self.np_cost_valid_indexes,].transpose()
        np_param_hdrs_hash = {}
        for index, hdr in enumerate(self.np_param_hdrs):
            np_param_hdrs_hash[hdr] = index
        unused_params_list = []
        count = 0
        for pdata in self.param_list:
            (param, param_values, pindex) = pdata
            if len(param_values) > 1:
                if param.name in np_param_hdrs_hash:
                    count = count + 1
                elif param.map in np_param_hdrs_hash:
                    count = count + 1
                else:
                    unused_params_list.append(param.name)
                    self.unused_params_values.append(param_values[0])
            else:
                unused_params_list.append(param.name)
                self.unused_params_values.append(param_values[0])
        if count != len(self.np_param_hdrs):
            for (param, param_values, pindex) in self.param_list:
                self.param_extract_indexes.append(pindex)
            return
        self.rev_param_list = self.np_param_hdrs + unused_params_list
        self.param_extract_indexes = [index for index in range(len(self.np_param_hdrs))]
        self.rev_param_extract_indexes = [index for index in range(len(self.param_list))]
        for index, hdr in enumerate(self.rev_param_list):
            self.rev_param_extract_indexes[param_hash[hdr][2]] = index
        for (param, param_values, pindex) in self.param_list:
            if len(param_values) > 1:
                if param.name in np_param_hdrs_hash:
                    self.param_extract_indexes[np_param_hdrs_hash[param.name]] = pindex
                elif param.map in np_param_hdrs_hash:
                    self.param_extract_indexes[np_param_hdrs_hash[param.map]] = pindex
        self.np_param_hdrs_hash = np_param_hdrs_hash
        for index in range(len(self.param_data)):
            self.param_data_hash[tuple(self.param_data[index])] = self.cost_data[index]

    # Get parameters full list which includes the parameters used only for ML model and unused parameters
    def GetParamsFullList(self, np_params):
        return np.append(np_params, self.unused_params_values)[self.rev_param_extract_indexes,]

    # Get pre-evaluated parameters
    def GetPreEvaluatedParameters(self, samples, param_list):
        indexes = samples[0].tolist() + samples[1].tolist()
        out_params = []
        for nd_index in indexes:
            params = self.GetParamsFullList(self.param_data[nd_index])
            out_params.append(params)
        return out_params

    # Get parameter hash for pre-loaded data
    def GetParamHash(self, param_val):
        param_hash = {}
        index = 0
        for (param, param_values, pindex) in self.param_list:
            param_key = "${"+param.name+"}"
            param_hash[param_key] = param_val[index]
            param_key = "${"+param.map+"}"
            if param.name != param.map:
                if param_key in param_hash:
                    print("[Error] Multiple map_name(s):"+param.map+" used in the evaluation")
                param_hash[param_key] = param_val[index]
            index = index + 1
        param_dict = dict((re.escape(k), v) for k, v in param_hash.items())
        param_pattern = re.compile("|".join(param_dict.keys()))
        return (param_pattern, param_hash, param_dict)
     
    # Create environment for evaluating one sample    
    def CreateEvaluateCase(self, param_val):
        (param_pattern, param_hash, param_dict) = self.parameters.GetParamHash(param_val, self.param_list)
        run_dir = self.fr_config.run_directory
        dir_name = os.path.join(run_dir, "evaluate_"+str(self.counter))
        run_dir = dir_name
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        scripts = []
        run_dir = os.path.abspath(run_dir)
        if "${command_file}" in param_hash:
            filename = param_hash["${command_file}"]
            self.parameters.CreateRunScript(filename, run_dir, param_pattern, param_dict)
            scripts.append((run_dir, filename))
        if "${command_option}" in param_hash:
            filename = param_hash["${command_option}"]
            self.parameters.CreateRunScript(filename, run_dir, param_pattern, param_dict)
            scripts.append((run_dir, filename))
        self.parameters.CreateRunScript(self.config.sample_evaluate_script, run_dir, param_pattern, param_dict)
        scripts.append((run_dir, self.config.sample_evaluate_script))
        cmd = ""
        for index, (rdir, filename) in enumerate(scripts):
            redirect_symbol = ">>"
            if index == 0:
                redirect_symbol = ">"
            cmd = cmd + "cd "+rdir+" ; sh "+filename+" "+redirect_symbol+" "+self.config.output_log+" 2>&1 3>&1 ; cd "+os.getcwd()+" ; "
        with open(os.path.join(run_dir, "evaluate.sh"), "w") as fh:
            fh.write(cmd)
            fh.close()
        self.counter = self.counter + 1
        out = ((run_dir, self.config.sample_evaluate_script), cmd)
        if self.config.slurm:
            slurm_script_filename = os.path.join(run_dir, "slurm_evaluate.sh")
            self.framework.slurm.CreateSlurmScript(cmd, slurm_script_filename)
            slurm_script_cmd = self.framework.slurm.GetSlurmJobCommand(slurm_script_filename)
            out = ((run_dir, slurm_script_filename), slurm_script_cmd)
        return out

    # Run method will evaluate the set of parameters
    # ** If it is available in the pre-loaded data, it returns that value
    # ** Else, it evaluate in traditional way
    def Run(self, parameters):
        eval_output = []
        mt = MultiThreadBatchRun(self.batch_size, self.framework)
        for param_val in parameters:
            param_hash_key = tuple(param_val[self.param_extract_indexes].tolist())
            if param_hash_key in self.param_data_hash:
                eval_output.append((self.framework.predicted_flag, self.param_data_hash[param_hash_key]))
            else:
                (output, cmd) = self.CreateEvaluateCase(param_val)
                eval_output.append((self.framework.evaluate_flag, output))
                mt.Run([cmd])
        mt.Close()
        return eval_output

# Get object of evaluate
def GetObject(framework):
    obj = DeffeEvaluate(framework)
    return obj
