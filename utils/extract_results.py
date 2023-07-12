import json
import csv
import os
import re
import natsort
import pandas as pd
import argparse
import sys
import pdb

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
    idx_re = re.compile(r"explore_(\d*)")
    new_df_data = []
    for explore in explore_dirs:
        try:
            match = idx_re.match(explore)
            if match:
                i = int(match[1])
    
            else:
                print(f"Skipping {explore}")
                continue
            
            #print(explore)
            results_json = f'sd_exploration/{explore}/results.json'
            params_json = f'sd_exploration/{explore}/param.json'
            if os.path.exists(results_json) and os.path.exists(params_json):
                d = json.load(open(results_json, 'r'))
                p = json.load(open(params_json, 'r'))
                d.update(p)
                d['explore'] = i
                data.append(d)
                if len(d) > 0:
                    new_df_data.append(d)
        except Exception as e:
            print(e)
            print(f"*** Missing Data for {explore} ***")

    new_df = pd.DataFrame(new_df_data)
    print(f"[Info] Generating output in {args.output}")
    new_df.to_csv(args.output, index=False, sep=",", na_rep='null', encoding="utf-8")

def main(argv):
    args = init_parser(argv)
    explore_dirs = get_explore_dirs(args)
    df = process_explore_dirs(args, explore_dirs)

if __name__ == "__main__":
    main(sys.argv)

