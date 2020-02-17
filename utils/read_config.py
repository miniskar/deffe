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
        values_list = re.split("\s*,\s*", values)
        for value in values_list:
            if re.search(r'\s*-\s*', value):
                fields = re.split('\s*-\s*', value)
                start = int(fields[0])
                end = int(fields[1])
                inc = 1
                if len(fields) > 2:
                    inc = int(fields[2])
                sub_values = [str(i) for i in range(start, end, inc)]
                self.values.extend(sub_values)
            else:
                self.values.append(value)

class DeffeConfigKnob:
    def __init__(self, data):
        self.data = data
        self.name = data['name']
        self.values = []
        if 'values' in data:
            self.values = DeffeConfigValues(data['values'])

class DeffeConfigParameter:
    def __init__(self, data):
        self.data = data
        self.name = data['name']
        self.values = []
        if 'values' in data:
            self.values = DeffeConfigValues(data['values'])

class DeffeConfigSystem:
    def __init__(self, data):
        self.data = data
        self.name = data['name']
        self.knobs = []
        if 'knobs' in data:
            self.knobs = [DeffeConfigKnob(knob) for knob in data['knobs']]

class DeffeConfigWorkload:
    def __init__(self, data):
        self.data = data
        self.name = data['name']
        self.parameters = []
        if 'parameters' in data:
            self.parameters = [DeffeConfigParameter(wk) for wk in data['parameters']]
        self.knobs = []
        if 'knobs' in data:
            self.knobs = [DeffeConfigKnob(wk) for wk in data['knobs']]

class DeffeConfigModel:
    def __init__(self, data):
        self.data = data
        self.script = data['script']
        self.ml_model_script = data['ml_model_script']
        self.arguments = data['arguments']
        self.output = data['output']
        
class DeffeConfigExploration:
    def __init__(self, data):
        self.data = data
        self.script = data['script']
        self.arguments = data['arguments']
        self.output = data['output']
        
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
        self.evaluation_table = data['evaluation_table']
        self.nn_predict_table = data['nn_predict_table']

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

    def GetSystems(self):
        if self.data != None and 'systems' in self.data:
            return [ DeffeConfigSystem(sys) for sys in self.data['systems'] ]
        return []

    def GetWorkloads(self):
        if self.data != None and 'workloads' in self.data:
            return [ DeffeConfigWorkload(wk) for wk in self.data['workloads'] ]
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
