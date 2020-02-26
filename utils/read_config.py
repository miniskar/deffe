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
            if re.search(r'^([0-9]+)\s*-\s*([0-9]+)', value):
                fields = re.split('\s*-\s*', value)
                start = int(fields[0])
                end = int(fields[1])
                inc = 1
                if len(fields) > 2:
                    inc = int(fields[2])
                sub_values = [str(i) for i in range(start, end, inc)]
                values_extract.extend(sub_values)
            else:
                values_extract.append(value)
        return values_extract

class DeffeConfigKnob:
    def __init__(self, data):
        self.data = data
        self.name = data['name']
        self.values = []
        self.groups = []
        self.map = self.name
        if 'map' in data:
            self.map = data['map']
        if 'values' in data:
            self.values = DeffeConfigValues(data['values'])
        if 'groups' in data:
            self.groups = re.split('\s*,\s*', data['groups'])
        self.groups.append("all")

class DeffeConfigScenarios:
    def __init__(self, data):
        self.data = data
        self.name = data['name']
        self.values = []
        self.groups = []
        self.map = self.name
        if 'map' in data:
            self.map = data['map']
        if 'values' in data:
            self.values = DeffeConfigValues(data['values'])
        if 'groups' in data:
            self.groups = re.split('\s*,\s*', data['groups'])
        self.groups.append("all")

class DeffeConfigModel:
    def __init__(self, data):
        self.data = data
        self.script = data['script']
        self.ml_model_script = data['ml_model_script']
        self.arguments = data['arguments']
        self.output = data['output']
        self.parameters = []
        if 'parameters' in data: 
            self.parameters = re.split('\s*,\s*', data['parameters'])
        
class DeffeConfigExploration:
    def __init__(self, data):
        self.data = data
        self.script = data['script']
        self.arguments = data['arguments']
        self.output = data['output']
        self.exploration_list = []
        if 'explore' in data:
            self.exploration_list = data['explore']
        if len(self.exploration_list) == 0:
            self.exploration_list = ["all"]
        
class DeffeConfigSampling:
    def __init__(self, data):
        self.data = data
        self.script = data['script']
        self.arguments = data['arguments']
        self.output = data['output']
        
class DeffeConfigEvaluate:
    def __init__(self, data):
        self.data = data
        self.script = data['script']
        self.arguments = data['arguments']
        self.output = data['output']
        self.slurm = False
        if 'slurm' in data and data['slurm'].lower() == 'true':
            self.slurm = True
        self.batch = 40
        if 'batch' in data:
            self.batch = int(data['batch']) 
        self.sample_evaluate_script = "run_sample.sh"
        if 'sample_evaluate_script' in data:
            self.sample_evaluate_script = data['sample_evaluate_script'] 
        self.parameters = []
        if 'parameters' in data: 
            self.parameters = re.split('\s*,\s*', data['parameters'])
        
class DeffeConfigExtract:
    def __init__(self, data):
        self.data = data
        self.script = data['script']
        self.arguments = data['arguments']
        self.output = data['output']

class DeffeConfigFramework:
    def __init__(self, data):
        self.data = data
        self.output = data['output']
        self.run_directory = "run"
        if 'run_directory' in data:
            self.run_directory = data['run_directory']
        self.exploration_table = "deffe_exploration.hdf5"
        if 'exploration_table' in data:
            self.exploration_table = data['exploration_table']
        self.evaluation_table = "deffe_evaluation.hdf5"
        if 'evaluation_table' in data:
            self.evaluation_table = data['evaluation_table']
        self.ml_predict_table = "deffe_prediction.hdf5"
        if 'ml_predict_table' in data:
            self.ml_predict_table = data['ml_predict_table']

class DeffeConfig:
    def __init__(self, file_name=None):
        self.data = None
        if file_name != None:
            self.ReadFile(file_name)

    def ReadFile(self, filename):
        self.file_name = filename
        with open(filename) as infile:
            self.data = json.load(infile)
            return self.data
        return None

    def GetPythonPaths(self):
        if self.data != None and 'python_path' in self.data:
            return self.data['python_path']
        return []

    def GetKnobs(self):
        if self.data != None and 'knobs' in self.data:
            return [DeffeConfigKnob(knob) for knob in self.data['knobs']]
        return []

    def GetScenarios(self):
        if self.data != None and 'scenarios' in self.data:
            return [DeffeConfigScenarios(scn) for scn in self.data['scenarios']]
        return []

    def GetCosts(self):
        if self.data != None and 'costs' in self.data:
            return self.data['costs']
        return []

    def GetModel(self):
        if self.data != None and 'model' in self.data:
            return DeffeConfigModel(self.data['model'])
        return None

    def GetExploration(self):
        if self.data != None and 'exploration' in self.data:
            return DeffeConfigExploration(self.data['exploration'])
        return None

    def GetSampling(self):
        if self.data != None and 'sampling' in self.data:
            return DeffeConfigSampling(self.data['sampling'])
        return None

    def GetEvaluate(self):
        if self.data != None and 'evaluate' in self.data:
            return DeffeConfigEvaluate(self.data['evaluate'])
        return None

    def GetExtract(self):
        if self.data != None and 'extract' in self.data:
            return DeffeConfigExtract(self.data['extract'])
        return None

    def GetFramework(self):
        if self.data != None and 'framework' in self.data:
            return DeffeConfigFramework(self.data['framework'])
        return None

def main():
    name = 'config.json'
    config = DeffeConfig(name)

if __name__ == "__main__":
    main()
