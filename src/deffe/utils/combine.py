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
import pandas as pd
import jsoncomment 

def getFileSuffix(fname):
    return ''.join(pathlib.Path(fname).suffixes)
def isFileExists(fname, suffix=''):
    if suffix == '':
        return os.path.exists(fname)
    else:
        return re.compile(r'.*'+re.sub(r'\.', '\\.', suffix)).search(fname) and os.path.exists(fname) and pathlib.Path(fname)

def read_results_file(filename):
    file_name = os.path.expandvars(filename)
    if not os.path.exists(file_name):
        print("[Error] Json file:{} not available!".format(file_name))
        return None
    data = {}
    with open(file_name) as infile:
        data = jsoncomment.JsonComment().load(infile)
        infile.close()
    return data

def init_parser(argv):
    parser = argparse.ArgumentParser(description="""Read and print a test sample.""")
    parser.add_argument('-files', '--files', dest='files', help='Read files.', nargs='+', default=[])
    parser.add_argument('-output', '--output', dest='output', type=str, default="combine_output.csv")
    args = parser.parse_args(argv[1:])
    return args

def process_files(args):
    df = pd.DataFrame()
    for f in args.files:
        data = read_results_file(f)
        if len(data) > 0:
            df = df.append(data, ignore_index=True)
    return df

def generate_output(args, df):
    print(f"[Info] Generating output in {args.output}")
    df.to_csv(args.output, index=False, sep=",", na_rep='null', encoding="utf-8")

def main(argv):
    args = init_parser(argv)
    df = process_files(args)
    generate_output(args, df)

if __name__ == "__main__":
    main(sys.argv)
