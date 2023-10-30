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
import re
import importlib, pathlib
import sys
import pdb
import inspect
import numpy as np

def IsNumber(x):
    allowed_types = [
        float,
        int,
        np.float64,
        np.float32,
        np.float16,
        np.int64,
        np.int32,
        np.int16,
        np.int8,
        np.uint64,
        np.uint32,
        np.uint16,
        np.uint8,
    ]
    if type(x) in allowed_types:
        return True
    return False

debug_flag = False
def EnableDebugFlag():
    global debug_flag
    debug_flag = True
# Generic loading of python module
def LoadModule(parent, py_file):
    py_mod_name = pathlib.Path(py_file).stem
    py_mod = importlib.import_module(py_mod_name)
    return py_mod.GetObject(parent)

# Generic loading of python module
def LoadPyModule(*args):
    py_file = args[0]
    rargs = args[1:]
    py_mod_name = pathlib.Path(py_file).stem
    py_mod = importlib.import_module(py_mod_name)
    return py_mod.GetObject(*rargs)

# Generic loading of python module
def LoadModuleNoParent(py_file):
    py_mod_name = pathlib.Path(py_file).stem
    py_mod = importlib.import_module(py_mod_name)
    return py_mod.GetObject()

# Remove whitespaces in line
def RemoveWhiteSpaces(line):
    line = re.sub(r"\r", "", line)
    line = re.sub(r"\n", "", line)
    line = re.sub(r"^\s*$", "", line)
    line = re.sub(r"^\s*", "", line)
    line = re.sub(r"\s*$", "", line)
    return line

def GetFmtMsg(message):
    framework_path = os.getenv("DEFFE_DIR")
    message = re.sub(re.escape(framework_path), "$DEFFE_DIR", message)
    return message

def Log(message, type='Info'):
    message = GetFmtMsg(message)
    print("["+type+"] "+message, flush=True)

def LogCmd(message, type='Cmd'):
    Log(message, type)

def LogModule(message, type='Info', caller_name=None):
    #called_name = sys._getframe().f_code.co_name
    if type == 'Debug' and not debug_flag:
        return
    if caller_name == None:
        stack = inspect.stack()
        caller_class = stack[1][0].f_locals["self"].__class__.__name__
        caller_name = sys._getframe().f_back.f_code.co_name
    message = GetFmtMsg(message)
    print("["+type+"] ("+caller_class+"."+caller_name+"): "+message)
    sys.stdout.flush()
    
def ReshapeCosts(cost_data):
    out_cost_data = []
    for i in range(cost_data.shape[1]):
        one_train = cost_data.transpose()[i].reshape(cost_data.shape[0], 1)
        out_cost_data.append(one_train)
    return np.array(out_cost_data)

def DebugLogModule(message, caller_name=None):
    #called_name = sys._getframe().f_code.co_name
    if not debug_flag:
        return
    caller_class = ''
    if caller_name == None:
        stack = inspect.stack()
        caller_class = stack[1][0].f_locals["self"].__class__.__name__+"."
        caller_name = sys._getframe().f_back.f_code.co_name
    message = GetFmtMsg(message)
    print("[Debug] ("+caller_class+caller_name+"): "+message)
    sys.stdout.flush()
    
def AddBashKeyValue(hash_obj, key, val, escape=False):
    if escape:
        hash_obj[re.escape("${"+key+"}")] = val
        hash_obj[re.escape("$"+key)] = val 
    else:
        hash_obj["${"+key+"}"] = val
        hash_obj["$"+key] = val 
    
def GetHashCopy(obj, escape_key=False):
    if escape_key:
        new_obj = { 
            re.escape(k):v for k,v in obj.items() 
        }
    else:
        new_obj = { 
            k:v for k,v in obj.items() 
        }
    return new_obj
if __name__ == "__main__":
    class A:
        def __init__(self):
            None
        def Message(self):
            LogModule("Hello")
    a = A()
    a.Message()

num_pattern= r'^\s*([-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?)\s*$'
num_rx = re.compile(num_pattern, re.VERBOSE)
def IsStringNumber(number):
    global num_rx
    if num_rx.search(number):
        return True
    return False

numeric_const_pattern = r'\breal\b\s*([-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?)'
def GetScriptExecutionTime(fname):
    global numeric_const_pattern
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    if os.path.isfile(fname):
        with open(fname, 'r') as fh:
            lines = fh.readlines()
            data = "---".join(lines)
            data = data.replace("\n", '')
            fields = rx.search(data)
            if fields:
                return fields.group(1)
    return '0.0'

if __name__ == "__main__":
    print("Current directory: " + os.getcwd())
    data = GetScriptExecutionTime('evaluate.log')
    pdb.set_trace()
    print(data)
