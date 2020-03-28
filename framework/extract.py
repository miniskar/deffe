import os
from multi_thread_run import *
from deffe_utils import *
import numpy as np
import argparse
import shlex

class DeffeExtract:
    def __init__(self, framework):
        self.framework = framework
        self.config = framework.config.GetExtract()
        self.batch_size = int(self.config.batch_size)
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
    
    # Initialize the class with parameters list and to be extracted cost metrics
    def Initialize(self, param_list, cost_list):
        self.param_list = param_list
        self.cost_list = cost_list

    def GetExtractCommand(self, output, param_pattern, param_dict):
        (run_dir, sample_evaluate_script) = output
        extract_script = self.config.sample_extract_script
        self.parameters.CreateRunScript(extract_script, run_dir, param_pattern, param_dict)
        cmd = "cd "+run_dir+" ; sh "+extract_script+" > "+self.config.output_log+" 2>&1 3>&1 ; cd "+os.getcwd()
        return cmd

    def GetResult(self, flag, eval_output):
        (run_dir, sample_evaluate_script) = eval_output
        file_path = os.path.join(run_dir, self.config.cost_output)
        if os.path.exists(file_path):
            with open(file_path, "r") as fh:
                lines = fh.readlines()
                return (self.framework.valid_flag, np.array([RemoveWhiteSpaces(lines[0]),]).astype('str'))
        return (self.framework.not_valid_flag, flag, np.array([0,]).astype('str'))

    # Run the extraction
    def Run(self, param_val, param_list, eval_output):
        batch_output = []
        mt = MultiThreadBatchRun(self.batch_size, self.framework)

        for index, (flag, output) in enumerate(eval_output):
            (param_pattern, param_hash, param_dict) = self.parameters.GetParamHash(param_val[index], self.param_list)
            if flag == self.framework.predicted_flag:
                batch_output.append((self.framework.valid_flag, flag, output))
            elif flag == self.framework.evaluate_flag:
                cmd = self.GetExtractCommand(output, param_pattern, param_dict)
                (run_dir, eval_script_filename) = output
                if self.config.slurm:
                    slurm_script_filename = os.path.join(run_dir, "slurm_extract.sh")
                    self.framework.slurm.CreateSlurmScript(cmd, slurm_script_filename)
                    cmd = self.framework.slurm.GetSlurmJobCommand(slurm_script_filename)
                mt.Run([cmd])
                batch_output.append((self.framework.not_valid_flag, flag, np.array([0,]).astype('str')))
            else:
                print("[Error] Unknow flag received in DeffeExtract::Run")
                batch_output.append((self.framework.not_valid_flag, flag, np.array([0,]).astype('str')))
        mt.Close()
        for index, (flag, output) in enumerate(eval_output):
            if flag == self.framework.evaluate_flag:
                batch_output[index] = self.GetResult(flag, output)
        return batch_output

def GetObject(framework):
    obj = DeffeExtract(framework)
    return obj
