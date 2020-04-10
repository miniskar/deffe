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


def RemoveWhiteSpaces(line):
    line = re.sub(r"\r", "", line)
    line = re.sub(r"\n", "", line)
    line = re.sub(r"^\s*$", "", line)
    line = re.sub(r"^\s*", "", line)
    line = re.sub(r"\s*$", "", line)
    return line
