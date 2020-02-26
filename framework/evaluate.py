import os
import numpy as np
import pdb
import shutil
import re
import sys

class DeffeRISCVEvaluate:
    def __init__(self, framework):
        self.framework = framework
        self.config = framework.config.GetEvaluate()
        self.counter = 0
        self.fr_config = self.framework.fr_config

    def InitializeParser(parser):
        None

    def SetArgs(args):
        self.args = args

    def GetParamHash(self, param_val, param_list):
        param_hash = {}
        index = 0
        for (param, param_values, pindex) in param_list:
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
            

    def CreateEvaluateCase(self, param_val, param_list):
        (param_pattern, param_hash, param_dict) = self.GetParamHash(param_val, param_list)
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

    def Run(self, param_list, parameters):
        for param_val in parameters:
            self.CreateEvaluateCase(param_val, param_list)
        sys.exit(0)
        eval_output = np.zeros(shape=(parameters.shape[0], 1))
        return eval_output

def GetObject(framework):
    obj = DeffeRISCVEvaluate(framework)
    return obj
