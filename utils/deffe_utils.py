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


# Generic loading of python module
def LoadModule(parent, py_file):
    py_mod_name = pathlib.Path(py_file).stem
    py_mod = importlib.import_module(py_mod_name)
    return py_mod.GetObject(parent)

# Remove whitespaces in line
def RemoveWhiteSpaces(line):
    line = re.sub(r"\r", "", line)
    line = re.sub(r"\n", "", line)
    line = re.sub(r"^\s*$", "", line)
    line = re.sub(r"^\s*", "", line)
    line = re.sub(r"\s*$", "", line)
    return line
