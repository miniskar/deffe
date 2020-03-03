import os
import re

def RemoveWhiteSpaces(line):
    line = re.sub(r'\r', '', line)
    line = re.sub(r'\n', '', line)
    line = re.sub(r'^\s*$', '', line)
    line = re.sub(r'^\s*', '', line)
    line = re.sub(r'\s*$', '', line)
    return line



