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
        np_hdrs_hash = {}
        param_hash = {}
        cost_hash = {}
        for index, cost in enumerate(self.cost_list):
            cost_hash[cost] = index
        for pdata in self.param_list:
            (param, param_values, pindex) = pdata
            param_hdrs.append(param.name.lower())
            param_hash[param.name.lower()] = pdata
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
            np_hdrs_hash[hdr] = index
        trans_data = preload_data.transpose()
        self.param_data = trans_data[self.np_param_valid_indexes,].transpose()
        self.cost_data = trans_data[self.np_cost_valid_indexes,].transpose()
        self.param_data_hash = {}
        for index in range(len(self.param_data)):
            self.param_data_hash[tuple(self.param_data[index])] = self.cost_data[index]
        None

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

    def Run(self, parameters):
        for param_val in parameters:
            self.CreateEvaluateCase(param_val)
        sys.exit(0)
        eval_output = np.zeros(shape=(parameters.shape[0], 1))
        return eval_output

def GetObject(framework):
    obj = DeffeEvaluate(framework)
    return obj
