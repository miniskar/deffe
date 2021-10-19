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
    
def DebugLogModule(message, caller_name=None):
    #called_name = sys._getframe().f_code.co_name
    if not debug_flag:
        return
    if caller_name == None:
        stack = inspect.stack()
        caller_class = stack[1][0].f_locals["self"].__class__.__name__
        caller_name = sys._getframe().f_back.f_code.co_name
    message = GetFmtMsg(message)
    print("[Debug] ("+caller_class+"."+caller_name+"): "+message)
    sys.stdout.flush()
    
if __name__ == "__main__":
    class A:
        def __init__(self):
            None
        def Message(self):
            LogModule("Hello")
    a = A()
    a.Message()
