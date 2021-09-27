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
import numpy as np
import pandas as pd
from multi_thread_run import *
from deffe_utils import *
import argparse
import shlex

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
        if self.framework.args.no_slurm:
            self.slurm_flag = False
        if not os.path.exists(self.config.sample_evaluate_script):
            self.config.sample_evaluate_script = os.path.join(
                self.framework.config_dir, self.config.sample_evaluate_script
            )
        self.batch_size = int(self.config.batch_size)
        self.counter = 0
        self.fr_config = self.framework.fr_config
        if self.framework.args.batch_size!= -1:
            self.batch_size = self.framework.args.batch_size
        if self.framework.args.evaluate_batch_size != -1:
            self.batch_size = self.framework.args.evaluate_batch_size
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

    def WriteSamplesToCSV(self):
        n_samples = self.param_data.shape[0]
        org_seq = np.random.choice(n_samples, 
                size=min(20, n_samples), replace=False)
        x_train = self.param_data[
            org_seq,
        ]
        y_train = self.cost_data[
            org_seq,
        ]
        y_train = y_train.reshape((y_train.shape[0],))
        out_data_hash = {}
        x_train_tr = x_train.transpose()
        for index, hdr in enumerate(self.param_hdrs):
            out_data_hash[hdr] = x_train_tr[index].tolist()
        out_data_hash["cpu_cycles"] = y_train.tolist()
        df = pd.DataFrame(out_data_hash)
        df.to_csv("random-samples.csv", index=False, sep=",", encoding="utf-8")

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
        (param_pattern, param_val_hash, 
         param_val_with_escapechar_hash) = self.parameters.GetParamHash(
            param_val, self.param_list
        )
        run_dir = self.fr_config.run_directory
        dir_name = os.path.join(run_dir, "evaluate_" + str(self.counter))
        run_dir = dir_name
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        scripts = []
        run_dir = os.path.abspath(run_dir)
        evaluate_replacements_hash = { 
            k:v for k,v in param_val_with_escapechar_hash.items() 
        }
        evaluate_replacements_hash[re.escape("${evaluate_index}")] = str(self.counter)
        evaluate_replacements_hash[re.escape("$evaluate_index")] = str(self.counter)
        if "${command_file}" in param_val_hash:
            filename = param_val_hash["${command_file}"]
            if not os.path.exists(filename):
                filename = os.path.join(self.framework.config_dir, filename)
            self.parameters.CreateRunScript(
                filename, run_dir, 
                param_pattern, evaluate_replacements_hash
            )
            scripts.append((run_dir, filename))
        if "${command_option}" in param_val_hash:
            filename = param_val_hash["${command_option}"]
            if not os.path.exists(filename):
                filename = os.path.join(self.framework.config_dir, filename)
            self.parameters.CreateRunScript(
                filename, run_dir, 
                param_pattern, evaluate_replacements_hash
            )
            scripts.append((run_dir, filename))
        self.parameters.CreateRunScript(
            self.config.sample_evaluate_script, run_dir, 
            param_pattern, evaluate_replacements_hash
        )
        scripts.append((run_dir, 
                    os.path.basename(self.config.sample_evaluate_script)))
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
        with open(os.path.join(run_dir, "_sample_evaluate.sh"), "w") as fh:
            fh.write(cmd)
            fh.close()
        self.counter = self.counter + 1
        out = ((run_dir, self.config.sample_evaluate_script), cmd)
        if self.slurm_flag:
            slurm_script_filename = os.path.join(run_dir, "_slurm_evaluate.sh")
            self.framework.slurm.CreateSlurmScript(cmd, slurm_script_filename)
            slurm_script_cmd = self.framework.slurm.GetSlurmJobCommand(
                slurm_script_filename
            )
            out = ((run_dir, slurm_script_filename), slurm_script_cmd)
        return out

    # Run method will evaluate the set of parameters
    # ** If it is available in the pre-loaded data, it returns that value
    # ** Else, it evaluate in traditional way
    def Run(self, parameters, callback=None):
        eval_output = []
        mt = MultiThreadBatchRun(self.batch_size, self.framework)
        for pindex, param_val in enumerate(parameters):
            # Check if data already existing 
            param_cost = self.param_data.GetParamterCost(param_val)
            if type(param_cost) == np.ndarray and len(param_cost) > 0:
                eval_output.append(
                    (
                        self.framework.pre_evaluated_flag,
                        param_cost,
                    )
                )
                if callback != None:
                    callback[0].callback[1](pindex, self.framework.pre_evaluated_flag, param_val, param_cost)
            else:
                (output, cmd) = self.CreateEvaluateCase(param_val)
                eval_output.append((self.framework.evaluate_flag, output))
                mt.Run([(cmd, pindex, self.framework.pre_evaluated_flag, param_val, None)], callback=callback)
        mt.Close()
        return eval_output


# Get object of evaluate
def GetObject(framework):
    obj = DeffeEvaluate(framework)
    return obj
