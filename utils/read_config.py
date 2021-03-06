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
import commentjson
import jsoncomment 

# Values: Scalar / List
#   List: String with values seperated with ','        : Example: "1,2,3,4"
#         String with range of values representated '-': Example: "1-10"
#         represented as a list itself                 : Example: [1,2,3,4]
#         combination of all above        Example: [1, 2, "10-20", "30-40"]
# Note:
#         One can use different delimiter (instead of comma ',') in string 
#         This feature is useful to use it as list of list in a string itself
#                   Example: groups: "riscv, app_size::12;15;20-22, mode"
#                            In the above example, the second list is provided 
#                            app_size with list of values [12, 15, 20, 21, 22]
class DeffeConfigValues:
    def __init__(self, values, delim=','):
        self.values = []
        if type(values) == list:
            self.values = []
            for i in values:
                self.values.extend(self.ExtractValues(str(i), delim))
        else:
            self.values.extend(self.ExtractValues(values, delim))

    def GetRangeOfValues(self, value, values_extract, prefix='', postfix=''):
        if re.search(r"^([0-9]+)\s*-\s*([0-9]+)", value):
            fields = re.split("\s*-\s*", value)
            start = int(fields[0])
            end = int(fields[1])
            inc = 1
            if len(fields) > 2:
                inc = int(fields[2])
            sub_values = [prefix+str(i)+postfix for i in range(start, end, inc)]
            values_extract.extend(sub_values)
        else:
            #if type(value) == str and value.lower() == 'true':
            #    values_extract.append(True)
            #elif type(value) == str and value.lower() == 'false':
            #    values_extract.append(False)
            #else:
            values_extract.append(os.path.expandvars(value))

    def ExtractValues(self, values, delim=','):
        if type(values)!=str:
            values = str(values)
        values_list = re.split("\s*"+delim+"\s*", values)
        values_extract = []
        for value in values_list:
            range_value = value
            prefix = ''
            postfix = ''
            match = re.match(r'(.*)\[(.*)\](.*)', value)
            if match:
                prefix = match[1]
                range_value = match[2]
                postfix = match[3]
            self.GetRangeOfValues(range_value, values_extract, prefix, postfix)
        return values_extract

def MergeTwoJsons(json1, json2):
    def merge_two_json_lists(l1, l2):
        is_dict_list = False
        if len(l1) > 0:
            if isinstance(l1[0], dict):
                is_dict_list = True
        if len(l2) > 0: 
            if isinstance(l2[0], dict):
                is_dict_list = True
        if not is_dict_list:
            return l1+l2
        new_list = [] + l2
        new_list_hash = { edict['name'] : (edict, index) for index, edict in enumerate(new_list) }
        for edict in l1:
            if edict['name'] in new_list_hash:
                (olddict, index) = new_list_hash[edict['name']]
                new_list[index] = dict(merge_two_jsons(edict, olddict))
            else:
                new_list.append(edict)
        return new_list
    def merge_two_jsons(dict1, dict2):
        for k in set(dict1.keys()).union(dict2.keys()):
            if k in dict1 and k in dict2:
                if isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
                    yield (k, dict(merge_two_jsons(dict1[k], dict2[k])))
                elif type(dict1[k]) == list and type(dict2[k]) == list:
                    yield(k, merge_two_json_lists(dict1[k], dict2[k]))
                else:
                    # If one of the values is not a dict, you can't continue merging it.
                    # Value from first dict overrides one in first and we move on.
                    yield (k, dict1[k])
                # Alternatively, replace this with exception raiser to alert you of value conflicts
            elif k in dict1:
                yield (k, dict1[k])
            else:
                yield (k, dict2[k])
    return dict(merge_two_jsons(json1, json2))


class DeffeConfigKnob:
    def __init__(self, data):
        self.data = data
        self.name = data["name"]
        self.values = []
        self.groups = []
        self.map = self.name
        self.base = None
        self.groups_configured = False
        self.onedim_combination = False
        if "map" in data:
            self.map = data["map"]
        if "values" in data:
            self.values = DeffeConfigValues(data["values"])
        if "base" in data:
            self.base =data['base']
        if "onedim_combination" in data:
            onedim_combination = data['onedim_combination']
            if type(onedim_combination) == str and onedim_combination.lower() == 'true':
                self.onedim_combination = True
            else:
                self.onedim_combination = data['onedim_combination']
        if "groups" in data:
            self.groups = DeffeConfigValues(data["groups"]).values
            self.groups_configured = True
        if self.name not in self.groups:
            self.groups.append(self.name)
        self.groups.append("all")


class DeffeConfigScenarios:
    def __init__(self, data):
        self.data = data
        self.name = data["name"]
        self.values = []
        self.groups = []
        self.base = None
        self.map = self.name
        self.groups_configured = False
        if "map" in data:
            self.map = data["map"]
        if "values" in data:
            self.values = DeffeConfigValues(data["values"])
        if "groups" in data:
            self.groups = DeffeConfigValues(data["groups"]).values
            self.groups_configured = True
        if self.name not in self.groups:
            self.groups.append(self.name)
        self.groups.append("all")


class DeffeConfigModel:
    def __init__(self, data):
        self.data = data
        self.ml_model_type = "keras_cnn"
        if data != None and "ml_model_type" in data:
            self.ml_model_type = os.path.expandvars(data["ml_model_type"])
        self.ml_model_script = ""
        if data != None and "ml_model_script" in data:
            self.ml_model_script = os.path.expandvars(data["ml_model_script"])
        if self.ml_model_script == '':
            if self.ml_model_type == 'keras_cnn':
                self.ml_model_script = os.path.expandvars("keras_cnn.py")
            elif self.ml_model_type == 'torch_cnn':
                self.ml_model_script = os.path.expandvars("torch_cnn.py")
            elif self.ml_model_type == 'sklearn':
                self.ml_model_script = os.path.expandvars("sklearn_rf.py")
        self.output_log = "ml_model.log"
        if data != None and "output_log" in data:
            self.output_log = os.path.expandvars(data["output_log"])
        self.batch_size = -1
        if data != None and "batch_size" in data:
            self.batch_size = int(data["batch_size"])
        self.pyscript = "_ml_model.py"
        if data != None and "pyscript" in data:
            self.pyscript = os.path.expandvars(data["pyscript"])
        self.arguments = ""
        if data != None and "arguments" in data:
            self.arguments = os.path.expandvars(data["arguments"])
        self.exclude_costs = []
        if data != None and "exclude_costs" in data:
            self.exclude_costs = DeffeConfigValues(data["exclude_costs"]).values
        self.ml_arguments = ""
        if data != None and "ml_arguments" in data:
            self.ml_arguments = os.path.expandvars(data["ml_arguments"])


class DeffeConfigSingleExploration:
    def __init__(self, data, i):
        self.name = "explore_" + str(i)
        self.pre_evaluated_data = None
        self.groups = []
        if data == None:
            self.groups.append("all")
        self.valid_costs = []
        if data != None and "valid_costs" in data:
            self.valid_costs = DeffeConfigValues(data["valid_costs"]).values
        if data != None and "pre_evaluated_data" in data:
            self.pre_evaluated_data = os.path.expandvars(data["pre_evaluated_data"])
        if data != None and "groups" in data:
            self.groups = DeffeConfigValues(data["groups"]).values
        self.all_table = "deffe_output.csv"
        if data != None and "all_table" in data:
            self.all_table = os.path.expandvars(data["all_table"])
        self.exploration_table = "deffe_exploration.csv"
        if data != None and "exploration_table" in data:
            self.exploration_table = os.path.expandvars(data["exploration_table"])
        self.evaluation_table = "deffe_evaluation.csv"
        if data != None and "evaluation_table" in data:
            self.evaluation_table = os.path.expandvars(data["evaluation_table"])
        self.ml_predict_table = "deffe_prediction.csv"
        if data != None and "ml_predict_table" in data:
            self.ml_predict_table = os.path.expandvars(data["ml_predict_table"])
        self.evaluation_predict_table = "deffe_eval_predict.csv"
        if data != None and "evaluation_predict_table" in data:
            self.evaluation_predict_table = os.path.expandvars(data["evaluation_predict_table"])


class DeffeConfigExploration:
    def __init__(self, data):
        self.data = data
        self.pyscript = "_exploration.py"
        if data != None and "pyscript" in data:
            self.pyscript = os.path.expandvars(data["pyscript"])
        self.arguments = ""
        if data != None and "arguments" in data:
            self.arguments = os.path.expandvars(data["arguments"])
        self.output_log = "exploration.log"
        if data != None and "output_log" in data:
            self.output_log = os.path.expandvars(data["output_log"])
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
            self.output_log = os.path.expandvars(data["output_log"])
        self.pyscript = "_sampling.py"
        if data != None and "pyscript" in data:
            self.pyscript = os.path.expandvars(data["pyscript"])
        self.arguments = ""
        if data != None and "arguments" in data:
            self.arguments = os.path.expandvars(data["arguments"])
        self.validate_module = ""
        if data != None and "validate_module" in data:
            self.validate_module = os.path.expandvars(data["validate_module"])
        self.optimize_sample_sequence = ""
        if data != None and "optimize_sample_sequence_module" in data:
            self.optimize_sample_sequence = os.path.expandvars(data["optimize_sample_sequence_module"])
        self.cost_objective = []
        if data != None and "cost_objective" in data:
            self.cost_objective = [x.lower() for x in DeffeConfigValues(data["cost_objective"]).values]


class DeffeConfigEvaluate:
    def __init__(self, data, parent=None):
        self.data = data
        self.pyscript = "_evaluate.py"
        if data != None and "pyscript" in data:
            self.pyscript = os.path.expandvars(data["pyscript"])
        self.arguments = ""
        if data != None and "arguments" in data:
            self.arguments = os.path.expandvars(data["arguments"])
        self.batch_size = 20
        if data != None and "batch_size" in data:
            self.batch_size = int(data["batch_size"])
        self.output_flow = -1
        if data != None and "output_flow" in data:
            self.output_flow = int(data["output_flow"])
        self.output_log = "evaluate.log"
        if data != None and "output_log" in data:
            self.output_log = os.path.expandvars(data["output_log"])
        self.slurm = False
        self.slurm_config = {}
        if data != None and "slurm" in data:
            slurm_data = data['slurm']
            if isinstance(slurm_data, dict):
                self.slurm = True
                self.slurm_config = slurm_data
            elif type(slurm_data) == bool:
                self.slurm  =  slurm_data
            elif slurm_data.lower()=="true":
                self.slurm = True
        if parent != None and 'slurm' in parent:
            self.slurm_config = MergeTwoJsons(self.slurm_config, parent['slurm'])
        self.batch = 40
        if data != None and "batch" in data:
            self.batch = int(data["batch"])
        self.sample_evaluate_script = "evaluate.sh"
        if data != None and "sample_evaluate_script" in data:
            self.sample_evaluate_script = os.path.expandvars(data["sample_evaluate_script"])
        self.sample_evaluate_arguments = ""
        if data != None and "sample_evaluate_arguments" in data:
            self.sample_evaluate_arguments = os.path.expandvars(data["sample_evaluate_arguments"])
        self.sample_evaluate_excludes = ""
        if data != None and "sample_evaluate_excludes" in data:
            self.sample_evaluate_excludes = os.path.expandvars(data["sample_evaluate_excludes"])

    def GetSlurm(self):
        if self.slurm:
            return DeffeConfigSlurm(self.slurm_config)
        return DeffeConfigSlurm(None)

class DeffeConfigExtract:
    def __init__(self, data, parent=None):
        self.data = data
        self.pyscript = "_extract.py"
        if data != None and "pyscript" in data:
            self.pyscript = os.path.expandvars(data["pyscript"])
        self.arguments = ""
        if data != None and "arguments" in data:
            self.arguments = os.path.expandvars(data["arguments"])
        self.batch_size = 20
        if data != None and "batch_size" in data:
            self.batch_size = int(data["batch_size"])
        self.output_flow = -1
        if data != None and "output_flow" in data:
            self.output_flow = int(data["output_flow"])
        self.hold_evaluated_data = False
        if data != None and "hold_evaluated_data" in data and \
                   data["hold_evaluated_data"].lower() == "true":
            self.hold_evaluated_data = True
        self.output_log = "extract.log"
        if data != None and "output_log" in data:
            self.output_log = os.path.expandvars(data["output_log"])
        self.sample_extract_script = "extract.sh"
        if data != None and "sample_extract_script" in data:
            self.sample_extract_script = os.path.expandvars(data["sample_extract_script"])
        self.sample_extract_arguments = ""
        if data != None and "sample_extract_arguments" in data:
            self.sample_extract_arguments = os.path.expandvars(data["sample_extract_arguments"])
        self.sample_extract_excludes = ""
        if data != None and "sample_extract_excludes" in data:
            self.sample_extract_excludes = os.path.expandvars(data["sample_extract_excludes"])
        self.slurm = False
        self.slurm_config = {}
        if data != None and "slurm" in data:
            slurm_data = data['slurm']
            if isinstance(slurm_data, dict):
                self.slurm_config = slurm_data
                self.slurm = True
            elif type(slurm_data) == bool:
                self.slurm  =  slurm_data
            elif slurm_data.lower()=="true":
                self.slurm = True
        if parent != None and 'slurm' in parent:
            self.slurm_config = MergeTwoJsons(self.slurm_config, parent['slurm'])
        self.cost_output = "results.out"
        if data != None and "cost_output" in data:
            self.cost_output = os.path.expandvars(data["cost_output"])

    def GetSlurm(self):
        if self.slurm:
            return DeffeConfigSlurm(self.slurm_config)
        return DeffeConfigSlurm(None)


class DeffeConfigFramework:
    def __init__(self, data):
        self.data = data
        self.output_log = "framework.log"
        if data != None and "output_log" in data:
            self.output_log = os.path.expandvars(data["output_log"])
        self.run_directory = "run"
        if data != None and "run_directory" in data:
            self.run_directory = os.path.expandvars(data["run_directory"])


class DeffeConfigSlurm:
    def __init__(self, data):
        self.data = data
        self.nodes = "1"
        if data != None and "nodes" in data:
            self.nodes = str(data["nodes"])
        self.cpus_per_task = "1"
        if data != None and "cpus_per_task" in data:
            self.cpus_per_task = str(data["cpus_per_task"])
        self.exclude = ""
        if data != None and "exclude" in data:
            self.exclude = data["exclude"]
        self.mem = ""
        if data != None and "mem" in data:
            self.mem= data["mem"]
        self.mail_type = ""
        if data != None and "mail_type" in data:
            self.mail_type = str(data["mail_type"])
        self.mail_user = ""
        if data != None and "mail_user" in data:
            self.mail_user = str(data["mail_user"])
        self.constraint = ""
        if data != None and "constraint" in data:
            self.constraint = data["constraint"]
        self.partition = ""
        if data != None and "partition" in data:
            self.partition = data["partition"]
        self.source_scripts = []
        if data != None and "source_scripts" in data:
            self.source_scripts = DeffeConfigValues(data["source_scripts"]).values
        self.pyscript = "deffe_slurm.py"
        if data != None and "pyscript" in data:
            self.pyscript = os.path.expandvars(data["pyscript"])
        self.user_script_configured = False
        if (data != None
            and "user_script_configured" in data
            and data["user_script_configured"].lower() == "true"):
            self.user_script_configured = True

class DeffeConfig:
    def __init__(self, file_name=None, data=None):
        self.data = data 
        self.json_file = file_name
        if file_name != None:
            self.data = self.ReadFile(file_name)

    def ReadFile(self, filename):
        self.file_name = os.path.expandvars(filename)
        if not os.path.exists(self.file_name):
            print("[Error] Json file:{} not available!".format(self.file_name))
            return None
        with open(self.file_name) as infile:
            #data = commentjson.load(infile)
            data = jsoncomment.JsonComment().load(infile)
            if data != None and 'include' in data:
                includes = data['include']
                if type(includes) == list:
                    for inc_file in includes:
                        inc_data = self.ReadFile(inc_file)
                        if inc_data != None:
                            data = MergeTwoJsons(data, inc_data)
                else:
                    inc_data = self.ReadFile(includes)
                    if inc_data != None:
                        data = MergeTwoJsons(data, inc_data)
            return data
        return None

    def WriteFile(self, filename, data):
        filename = os.path.expandvars(filename)
        with open(filename, "w") as outfile:
            json.dump(data, outfile, indent=2)

    def GetPythonPaths(self):
        if self.data != None and "python_path" in self.data:
            py_list = self.data["python_path"]
            if type(py_list) == list:
                py_list = [os.path.expandvars(v) for v in py_list]
            else:
                py_list = os.path.expandvars(py_list)
            return py_list
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
            return [x.lower() for x in DeffeConfigValues(self.data["costs"]).values]
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
            return DeffeConfigEvaluate(self.data["evaluate"], self.data)
        return DeffeConfigEvaluate(None)

    def GetExtract(self):
        if self.data != None and "extract" in self.data:
            return DeffeConfigExtract(self.data["extract"], self.data)
        return DeffeConfigExtract(None)

    def GetFramework(self):
        if self.data != None and "framework" in self.data:
            return DeffeConfigFramework(self.data["framework"])
        return DeffeConfigFramework(None)

    def GetSlurm(self):
        if self.data != None and "slurm" in self.data:
            return DeffeConfigSlurm(self.data["slurm"])
        return DeffeConfigSlurm(None)


def test1():
    name = "config.json"
    config = DeffeConfigValues("hello[0-10]")
    print(config.values)
    config = DeffeConfigValues("[0-9]hello")
    print(config.values)
    config = DeffeConfigValues("hi[0-9]hello")
    print(config.values)

def test2():
    name = "config.json"
    config = DeffeConfig(name)
    config.WriteFile("config_write.json", config.data)
    print(config.GetCosts())
    #print(config.GetKnobs())

def main():
    test1()
    test2()

if __name__ == "__main__":
    main()
