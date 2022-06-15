import json
import csv
import os
import re
import natsort
import pandas as pd
import argparse
import sys

def get_explore_dirs(args):
    explore_dirs = natsort.natsorted([f for f in os.listdir(args.dirname) if os.path.isdir(os.path.join(args.dirname, f))])
    #print(explore_dirs)
    return explore_dirs

def init_parser(argv):
    parser = argparse.ArgumentParser(description="""Read and print a test sample.""")
    parser.add_argument('-dirname', '--dirname', dest='dirname', type=str, default="sd_exploration")
    parser.add_argument('-output', '--output', dest='output', type=str, default="deffe_extract_output.csv")
    args = parser.parse_args(argv[1:])
    return args

def process_explore_dirs(args, explore_dirs):
    data = []
    df = pd.DataFrame()
    idx_re = re.compile(r"explore_(\d*)")
    for explore in explore_dirs:
        try:
            match = idx_re.match(explore)
            if match:
                i = int(match[1])
    
            else:
                print(f"Skipping {explore}")
                continue
            
            #print(explore)
            d = json.load(open(f'sd_exploration/{explore}/results.json', 'r'))
            p = json.load(open(f'sd_exploration/{explore}/param.json', 'r'))
            d.update(p)
            d['explore'] = i
            data.append(d)
            if len(d) > 0:
                df = df.append(d, ignore_index=True)
            
        except Exception as e:
            print(e)
            print(f"*** Missing Data for {explore} ***")


    print(f"[Info] Generating output in {args.output}")
    df.to_csv(args.output, index=False, sep=",", na_rep='null', encoding="utf-8")

def main(argv):
    args = init_parser(argv)
    explore_dirs = get_explore_dirs(args)
    df = process_explore_dirs(args, explore_dirs)

if __name__ == "__main__":
    main(sys.argv)

