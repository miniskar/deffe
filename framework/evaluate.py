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
import pdb
import re
import sys
import shlex
import argparse
import numpy as np
import pandas as pd
from multi_thread_run import *
from deffe_utils import *
from read_config import *

""" DeffeEvaluate class to evaluate the batch of samples with 
    multi-thread execution environment either with/without 
    the help of slurm
    """
class DeffeEvaluate:
    # Constructor
    def __init__(self, framework):
        self.framework = framework
        self.config = framework.config.GetEvaluate()
        self.slurm_flag = self.config.slurm
        self.slurm = LoadPyModule(self.config.GetSlurm().pyscript, 
                self, self.config.GetSlurm())
        if self.framework.args.no_slurm:
            self.slurm_flag = False
        self.sample_evaluate_script = self.config.sample_evaluate_script
        if not os.path.exists(self.sample_evaluate_script):
            self.sample_evaluate_script = os.path.join(
                self.framework.config_dir, self.sample_evaluate_script
            )
        self.batch_size = self.config.batch_size
        self.counter = 0
        self.fr_config = self.framework.fr_config
        if self.framework.args.batch_size!= -1:
            self.batch_size = self.framework.args.batch_size
        if self.framework.args.evaluate_batch_size != -1:
            self.batch_size = self.framework.args.evaluate_batch_size
        self.output_flow = self.batch_size
        if self.config.output_flow != -1:
            self.output_flow = self.config.output_flow
        if self.framework.args.evaluate_out_flow != -1:
            self.output_flow = self.framework.args.evaluate_out_flow
        self.preload_data = []
        self.preload_header = []
        self.parameters = self.framework.parameters
        self.parser = self.AddArgumentsToParser()
        self.args = self.ReadArguments()
        self.mt = None

    def GetBatchSize(self):
        return self.batch_size

    def GetOutputFlow(self):
        return self.output_flow

    # Read arguments provided in JSON configuration file
    def ReadArguments(self):
        arg_string = self.config.arguments
        args = self.parser.parse_args(shlex.split(arg_string))
        return args

    # Add command line arguments to parser
    def AddArgumentsToParser(self):
        parser = argparse.ArgumentParser()
        return parser

    # Initialize method should be called for every 
    # new instance of new batch of samples.
    # Parameters to be passed: Parameters list, 
    #          Pruned parameters list, Parameters preloaded data
    # if any preload_file (Pre-Evaluated results)
    def Initialize(self, param_list, pruned_param_list, param_data):
        self.param_list = param_list
        self.pruned_param_list = pruned_param_list
        self.param_data = param_data

    # Create environment for evaluating one sample
    def CreateEvaluateCase(self, param_val):
        (param_pattern, 
         param_val_hash, param_val_with_escapechar_hash,
         bash_param_val_hash, bash_param_val_with_escapechar_hash) = \
            self.parameters.GetParamHash(
                param_val, self.param_list
            )
        run_dir = self.fr_config.run_directory
        #print(f"param_val_hash: {param_val_hash}")
        dir_name = os.path.join(run_dir, "explore_" + str(self.counter))
        run_dir = dir_name
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        scripts = []
        run_dir = os.path.abspath(run_dir)
        param_json = DeffeConfig()
        param_json_data = param_val_hash.copy()
        param_json_data['dir'] = f'explore_{self.counter}'
        param_json_data['index'] = f'{self.counter}'
        param_json.WriteFile(os.path.join(run_dir, "param.json"), param_json_data)
        bash_evaluate_replacements_hash = GetHashCopy(
                bash_param_val_with_escapechar_hash)
        evaluate_replacements_hash = GetHashCopy(
                param_val_with_escapechar_hash)
        AddBashKeyValue(bash_evaluate_replacements_hash, 
                "evaluate_index", str(self.counter), True)
        evaluate_replacements_hash['evaluate_index'] = str(self.counter)
        if "${command_file}" in bash_param_val_hash:
            filename = bash_param_val_hash["${command_file}"]
            if not os.path.exists(filename):
                filename = os.path.join(self.framework.config_dir, filename)
            filename = self.parameters.CreateRunScript(
                filename, "", "", run_dir, 
                param_pattern, 
                evaluate_replacements_hash,
                bash_evaluate_replacements_hash
            )
            scripts.append((run_dir, filename))
        if "${command_option}" in bash_param_val_hash:
            filename = bash_param_val_hash["${command_option}"]
            if not os.path.exists(filename):
                filename = os.path.join(self.framework.config_dir, filename)
            filename = self.parameters.CreateRunScript(
                filename, "", "", run_dir, 
                param_pattern, 
                evaluate_replacements_hash,
                bash_evaluate_replacements_hash
            )
            scripts.append((run_dir, filename))
        sample_evaluate_script = self.parameters.CreateRunScript(
            self.sample_evaluate_script, 
            self.config.sample_evaluate_arguments, 
            self.config.sample_evaluate_excludes, 
            run_dir, 
            param_pattern, 
            evaluate_replacements_hash,
            bash_evaluate_replacements_hash, "evaluate_"
        )
        scripts.append((run_dir, 
                    os.path.basename(sample_evaluate_script)))
        cmd = ""
        for index, (rdir, filename) in enumerate(scripts):
            redirect_symbol = ">>"
            if index == 0:
                redirect_symbol = ">"
            cmd = (
                cmd
                + "cd "
                + rdir
                + " ; sh "
                + filename
                + " "
                + redirect_symbol
                + " "
                + self.config.output_log
                + " 2>&1 3>&1 ; cd "
                + os.getcwd()
                + " ; "
            )
        #with open(os.path.join(run_dir, "_sample_evaluate.sh"), "w") as fh:
        #    fh.write(cmd)
        #    fh.close()
        out = ((run_dir, self.counter, self.sample_evaluate_script), cmd)
        if self.slurm_flag:
            slurm_script_filename = os.path.join(run_dir, 
                    f"_slurm_evaluate_{self.counter}.sh")
            self.slurm.CreateSlurmScript(cmd, 
                    slurm_script_filename)
            slurm_script_cmd = self.slurm.GetSlurmJobCommand(
                slurm_script_filename)
            out = ((run_dir, self.counter, slurm_script_filename), slurm_script_cmd)
        self.counter = self.counter + 1
        return out

    # Run method will evaluate the set of parameters
    # ** If it is available in the pre-loaded data, it returns that value
    # ** Else, it evaluate in traditional way
    def Run(self, parameters, callback=None, use_global_thread_queue=False):
        eval_output_list = []
        mt = None
        if use_global_thread_queue and self.mt == None:
            self.mt = MultiThreadBatchRun(self.batch_size, self.framework)
            mt = self.mt
        elif use_global_thread_queue:
            mt = self.mt
        elif not use_global_thread_queue:
            mt = MultiThreadBatchRun(self.batch_size, self.framework)
            callback = None
        for pindex, param_val in enumerate(parameters):
            # Check if data already existing 
            param_cost = self.param_data.GetParamterCost(param_val)
            if type(param_cost) == np.ndarray and len(param_cost) > 0:
                eval_output = (self.framework.pre_evaluated_flag, param_cost)
                eval_output_list.append(eval_output)
                if callback != None:
                    out_args = callback[0:1]+callback[2:]+(pindex, eval_output)
                    callback[1](*out_args)
            else:
                (output, cmd) = self.CreateEvaluateCase(param_val)
                eval_output = (self.framework.evaluate_flag, output)
                eval_output_list.append(eval_output)
                mt.Run([(cmd, pindex, eval_output)], callback=callback)
        if not use_global_thread_queue:
            self.WaitForThreads(mt)
        return eval_output_list

    def WaitForThreads(self, mt=None):
        if mt == None:
            mt = self.mt
        mt.Close()

# Get object of evaluate
def GetObject(*args):
    obj = DeffeEvaluate(*args)
    return obj
