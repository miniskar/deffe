import os
import pdb
import shutil
import re
import sys
import numpy as np
import pandas as pd

class DeffeEvaluate:
    def __init__(self, framework):
        self.framework = framework
        self.config = framework.config.GetEvaluate()
        self.counter = 0
        self.fr_config = self.framework.fr_config
        self.preload_data = []
        self.preload_header = []

    def Initialize(self, param_list, cost_list, preload_file):
        self.param_list = param_list
        self.cost_list = cost_list
        if preload_file == None:
            return
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
        pd_data = pd.read_csv(preload_file, dtype='str', delimiter=r'\s*,\s*', engine='python')
        np_data = pd_data.values.astype('str')
        np_hdrs = np.char.lower(np.array(list(pd_data.columns)).astype('str'))
        preload_data = np_data[1:]
        self.np_param_valid_indexes = []
        self.np_param_hdrs = []
        self.np_cost_valid_indexes = []
        self.np_cost_hdrs = []
        for index, hdr in enumerate(np_hdrs):
            if hdr in param_hash:
                self.np_param_valid_indexes.append(index)
                self.np_param_hdrs.append(hdr)
            if hdr in cost_hash:
                self.np_cost_hdrs.append(hdr)
                self.np_cost_valid_indexes.append(index)
        trans_data = preload_data.transpose()
        self.param_data = trans_data[self.np_param_valid_indexes,].transpose()
        self.cost_data = trans_data[self.np_cost_valid_indexes,].transpose()
        self.param_data_hash = {}
        np_param_hdrs_hash = {}
        for index, hdr in enumerate(self.np_param_hdrs):
            np_param_hdrs_hash[hdr] = index
        self.param_extract_indexes = []
        unused_params_list = []
        self.unused_params_values = []
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
                elif param.name in np_param_hdrs_hash:
                    self.param_extract_indexes[np_param_hdrs_hash[param.map]] = pindex
        for index in range(len(self.param_data)):
            self.param_data_hash[tuple(self.param_data[index])] = self.cost_data[index]

    def GetParamsFullList(self, np_params):
        return np.append(np_params, self.unused_params_values)[self.rev_param_extract_indexes,]

    def GetPreEvaluatedParameters(self, samples, param_list):
        indexes = samples[0].tolist() + samples[1].tolist()
        out_params = []
        for nd_index in indexes:
            params = self.GetParamsFullList(self.param_data[nd_index])
            out_params.append(params)
        return out_params

    def InitializeParser(self, parser):
        None

    def SetArgs(self, args):
        self.args = args

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
            
    def CreateRunScript(self, script, run_dir, param_pattern, param_dict):
        with open(script, "r") as rfh, \
             open(os.path.join(run_dir, os.path.basename(script)), "w") as wfh:
            lines = rfh.readlines()
            for line in lines:
                wline = param_pattern.sub(lambda m: param_dict[re.escape(m.group(0))], line)
                wfh.write(wline)
            rfh.close()
            wfh.close()
            

    def CreateEvaluateCase(self, param_val):
        (param_pattern, param_hash, param_dict) = self.GetParamHash(param_val)
        run_dir = self.fr_config.run_directory
        dir_name = os.path.join(run_dir, "evaluate_"+str(self.counter))
        run_dir = dir_name
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        self.CreateRunScript(self.config.sample_evaluate_script, run_dir, param_pattern, param_dict)
        if "${command_file}" in param_hash:
            filename = param_hash["${command_file}"]
            self.CreateRunScript(filename, run_dir, param_pattern, param_dict)
        if "${command_option}" in param_hash:
            filename = param_hash["${command_option}"]
            self.CreateRunScript(filename, run_dir, param_pattern, param_dict)
        self.counter = self.counter + 1

    def Run(self, parameters):
        eval_output = []
        for param_val in parameters:
            param_hash_key = tuple(param_val[self.param_extract_indexes].tolist())
            if param_hash_key in self.param_data_hash:
                eval_output.append((self.framework.predicted_flag, self.param_data_hash[param_hash_key]))
            else:
                output = self.CreateEvaluateCase(param_val)
                eval_output.append((self.framework.evaluate_flag, output))
        return eval_output

def GetObject(framework):
    obj = DeffeEvaluate(framework)
    return obj
