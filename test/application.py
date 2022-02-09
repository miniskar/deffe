import sys
import os
import pdb
import argparse
import re
import time

error_re = re.compile(r'\berror\b|\bfault\b|\bfatal\b', re.IGNORECASE)
def mean(t):
    return sum(t)/len(t)

def RunCommand(cmd):
    print("[Cmd] "+cmd)
    os.system(cmd)

def InitParser(parser):
    parser.add_argument('-cwd', dest='cwd', default=".")
    parser.add_argument('-app', dest='app', default="backprop")
    parser.add_argument('-evaluate-index', '--evaluate-index', dest='evaluate_index', default="1")
    parser.add_argument('-param1', '--param1', dest='param1', default="1")
    parser.add_argument('-param2', '--param2', dest='param2', default="1")
    parser.add_argument('-options', '--options', dest='options', default="16")
    parser.add_argument('-args', dest='args', nargs='*', default='')
    parser.add_argument('-iter', dest='iter', default="1")

def RemoveWhiteSpaces(line):
    line = re.sub(r'\r', '', line)
    line = re.sub(r'\n', '', line)
    line = re.sub(r'^\s*$', '', line)
    line = re.sub(r'^\s*', '', line)
    line = re.sub(r'\s*$', '', line)
    return line

def RunBench(args, out_dir):
    pwd = os.getcwd()
    print("Current working directory:"+pwd)
    os.chdir(pwd)
    timings = []
    val = int(args.param1)*int(args.param2)*float(args.options)
    results_file = os.path.join(out_dir, "results.out")
    RunCommand('echo "'+str(val)+'" > '+results_file)
    time.sleep(int(args.param1)*10)
    with open(results_file, 'r') as fh:
            t = fh.readline()
            t = RemoveWhiteSpaces(t)
            if t == '':
                timings.append(float("0"))
            else:
                timings.append(float(t))
    return mean(timings)

def main():
    parser = argparse.ArgumentParser()
    InitParser(parser)
    args = parser.parse_args()
    out_dir = os.getcwd()
    if args.cwd != '.':
        os.chdir(args.cwd)
    timing = RunBench(args, out_dir)
    print(timing)
    
if __name__ == "__main__":
    main()

