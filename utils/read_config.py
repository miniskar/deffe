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
import re
import os
import sys
import subprocess
import pdb
import itertools
import shlex
import tempfile
import argparse
import time
import json


class DeffeConfigValues:
    def __init__(self, values):
        self.values = []
        if type(values) == list:
            self.values = []
            for i in values:
                self.values.extend(self.ExtractValues(str(i)))
        else:
            self.values.extend(self.ExtractValues(values))

    def ExtractValues(self, values):
        values_list = re.split("\s*,\s*", values)
        values_extract = []
        for value in values_list:
            if re.search(r"^([0-9]+)\s*-\s*([0-9]+)", value):
                fields = re.split("\s*-\s*", value)
                start = int(fields[0])
                end = int(fields[1])
                inc = 1
                if len(fields) > 2:
                    inc = int(fields[2])
                sub_values = [str(i) for i in range(start, end, inc)]
                values_extract.extend(sub_values)
            else:
                values_extract.append(os.path.expandvars(value))
        return values_extract


class DeffeConfigKnob:
    def __init__(self, data):
        self.data = data
        self.name = data["name"]
        self.values = []
        self.groups = []
        self.map = self.name
        if "map" in data:
            self.map = data["map"]
        if "values" in data:
            self.values = DeffeConfigValues(data["values"])
        if "groups" in data:
            self.groups = re.split("\s*,\s*", data["groups"])
        self.groups.append("all")


class DeffeConfigScenarios:
    def __init__(self, data):
        self.data = data
        self.name = data["name"]
        self.values = []
        self.groups = []
        self.map = self.name
        if "map" in data:
            self.map = data["map"]
        if "values" in data:
            self.values = DeffeConfigValues(data["values"])
        if "groups" in data:
            self.groups = re.split("\s*,\s*", data["groups"])
        self.groups.append("all")


class DeffeConfigModel:
    def __init__(self, data):
        self.data = data
        self.ml_model_script = "keras_cnn.py"
        if data != None and "ml_model_script" in data:
            self.ml_model_script = data["ml_model_script"]
        self.output_log = "ml_model.log"
        if data != None and "output_log" in data:
            self.output_log = data["output_log"]
        self.pyscript = "ml_model.py"
        if data != None and "pyscript" in data:
            self.pyscript = data["pyscript"]
        self.arguments = ""
        if data != None and "arguments" in data:
            self.arguments = data["arguments"]
        self.ml_arguments = ""
        if data != None and "ml_arguments" in data:
            self.ml_arguments = data["ml_arguments"]


class DeffeConfigSingleExploration:
    def __init__(self, data, i):
        self.name = "explore_" + str(i)
        self.pre_evaluated_data = None
        self.groups = []
        if data == None:
            self.groups.append("all")
        if data != None and "pre_evaluated_data" in data:
            self.pre_evaluated_data = data["pre_evaluated_data"]
        if data != None and "groups" in data:
            self.groups = re.split("\s*,\s*", data["groups"])
        self.exploration_table = "deffe_exploration.csv"
        if data != None and "exploration_table" in data:
            self.exploration_table = data["exploration_table"]
        self.evaluation_table = "deffe_evaluation.csv"
        if data != None and "evaluation_table" in data:
            self.evaluation_table = data["evaluation_table"]
        self.ml_predict_table = "deffe_prediction.csv"
        if data != None and "ml_predict_table" in data:
            self.ml_predict_table = data["ml_predict_table"]
        self.evaluation_predict_table = "deffe_eval_predict.csv"
        if data != None and "evaluation_predict_table" in data:
            self.evaluation_predict_table = data["evaluation_predict_table"]


class DeffeConfigExploration:
    def __init__(self, data):
        self.data = data
        self.pyscript = "exploration.py"
        if data != None and "pyscript" in data:
            self.pyscript = data["pyscript"]
        self.arguments = ""
        if data != None and "arguments" in data:
            self.arguments = data["arguments"]
        self.output_log = "exploration.log"
        if data != None and "output_log" in data:
            self.output_log = data["output_log"]
        self.exploration_list = []
        if data != None and "explore" in data:
            self.exploration_list = [
                DeffeConfigSingleExploration(exp, index)
                for index, exp in enumerate(self.data["explore"])
            ]
        if len(self.exploration_list) == 0:
            self.exploration_list = [DeffeConfigSingleExploration(None, 0)]


class DeffeConfigSampling:
    def __init__(self, data):
        self.data = data
        self.output_log = "sampling.log"
        if data != None and "output_log" in data:
            self.output_log = data["output_log"]
        self.pyscript = "random_sampling.py"
        if data != None and "pyscript" in data:
            self.pyscript = data["pyscript"]
        self.arguments = ""
        if data != None and "arguments" in data:
            self.arguments = data["arguments"]


class DeffeConfigEvaluate:
    def __init__(self, data):
        self.data = data
        self.pyscript = "evaluate.py"
        if data != None and "pyscript" in data:
            self.pyscript = data["pyscript"]
        self.arguments = ""
        if data != None and "arguments" in data:
            self.arguments = data["arguments"]
        self.batch_size = "20"
        if data != None and "batch_size" in data:
            self.batch_size = data["batch_size"]
        self.output_log = "evaluate.log"
        if data != None and "output_log" in data:
            self.output_log = data["output_log"]
        self.slurm = False
        if data != None and "slurm" in data and data["slurm"].lower() == "true":
            self.slurm = True
        self.batch = 40
        if data != None and "batch" in data:
            self.batch = int(data["batch"])
        self.sample_evaluate_script = "evaluate.sh"
        if data != None and "sample_evaluate_script" in data:
            self.sample_evaluate_script = data["sample_evaluate_script"]


class DeffeConfigExtract:
    def __init__(self, data):
        self.data = data
        self.pyscript = "extract.py"
        if data != None and "pyscript" in data:
            self.pyscript = data["pyscript"]
        self.arguments = ""
        if data != None and "arguments" in data:
            self.arguments = data["arguments"]
        self.batch_size = "20"
        if data != None and "batch_size" in data:
            self.batch_size = data["batch_size"]
        self.output_log = "extract.log"
        if data != None and "output_log" in data:
            self.output_log = data["output_log"]
        self.sample_extract_script = "extract.sh"
        if data != None and "sample_extract_script" in data:
            self.sample_extract_script = data["sample_extract_script"]
        self.slurm = False
        if data != None and "slurm" in data and data["slurm"].lower() == "true":
            self.slurm = True
        self.cost_output = "results.out"
        if data != None and "cost_output" in data:
            self.cost_output = data["cost_output"]


class DeffeConfigFramework:
    def __init__(self, data):
        self.data = data
        self.output_log = "framework.log"
        if data != None and "output_log" in data:
            self.output_log = data["output_log"]
        self.run_directory = "run"
        if data != None and "run_directory" in data:
            self.run_directory = data["run_directory"]


class DeffeConfigSlurm:
    def __init__(self, data):
        self.data = data
        self.nodes = "1"
        if data != None and "nodes" in data:
            self.nodes = str(data["nodes"])
        self.cpus_per_task = "1"
        if data != None and "cpus_per_task" in data:
            self.cpus_per_task = str(data["cpus_per_task"])
        self.constriant = "x86_64,centos"
        if data != None and "constraint" in data:
            self.constraint = data["constraint"]
        self.pyscript = "deffe_slurm.py"
        if data != None and "pyscript" in data:
            self.pyscript = data["pyscript"]
        self.user_script_configured = False
        if (
            data != None
            and "user_script_configured" in data
            and data["user_script_configured"].lower() == "true"
        ):
            self.user_script_configured = True


class DeffeConfig:
    def __init__(self, file_name=None):
        self.data = None
        self.json_file = file_name
        if file_name != None:
            self.ReadFile(file_name)

    def ReadFile(self, filename):
        self.file_name = filename
        with open(filename) as infile:
            self.data = json.load(infile)
            return self.data
        return None

    def WriteFile(self, filename, data):
        self.file_name = filename
        with open(filename, "w") as outfile:
            json.dump(data, outfile, indent=2)

    def GetPythonPaths(self):
        if self.data != None and "python_path" in self.data:
            return self.data["python_path"]
        return []

    def GetKnobs(self):
        if self.data != None and "knobs" in self.data:
            return [DeffeConfigKnob(knob) for knob in self.data["knobs"]]
        return []

    def GetScenarios(self):
        if self.data != None and "scenarios" in self.data:
            return [DeffeConfigScenarios(scn) for scn in self.data["scenarios"]]
        return []

    def GetCosts(self):
        if self.data != None and "costs" in self.data:
            return self.data["costs"]
        return []

    def GetModel(self):
        if self.data != None and "model" in self.data:
            return DeffeConfigModel(self.data["model"])
        return DeffeConfigModel(None)

    def GetExploration(self):
        if self.data != None and "exploration" in self.data:
            return DeffeConfigExploration(self.data["exploration"])
        return DeffeConfigExploration(None)

    def GetSampling(self):
        if self.data != None and "sampling" in self.data:
            return DeffeConfigSampling(self.data["sampling"])
        return DeffeConfigSampling(None)

    def GetEvaluate(self):
        if self.data != None and "evaluate" in self.data:
            return DeffeConfigEvaluate(self.data["evaluate"])
        return DeffeConfigEvaluate(None)

    def GetExtract(self):
        if self.data != None and "extract" in self.data:
            return DeffeConfigExtract(self.data["extract"])
        return DeffeConfigExtract(None)

    def GetFramework(self):
        if self.data != None and "framework" in self.data:
            return DeffeConfigFramework(self.data["framework"])
        return DeffeConfigFramework(None)

    def GetSlurm(self):
        if self.data != None and "slurm" in self.data:
            return DeffeConfigSlurm(self.data["slurm"])
        return DeffeConfigSlurm(None)


def main():
    name = "config.json"
    config = DeffeConfig(name)
    config.WriteFile("config_write.json", config.data)


if __name__ == "__main__":
    main()
