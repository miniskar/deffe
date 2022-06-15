import json
import csv
import os
import re
import natsort
import pandas as pd

explore_dirs = natsort.natsorted([f for f in os.listdir('sd_exploration') if os.path.isdir(os.path.join('sd_exploration', f))])

idx_re = re.compile(r"explore_(\d*)")

print(explore_dirs)

data = []
df = pd.DataFrame()
for explore in explore_dirs:
    try:
        match = idx_re.match(explore)
        if match:
            i = int(match[1])

        else:
            print(f"Skipping {explore}")
            continue
        
        print(explore)
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


print(f"[Info] Generating output in combine_all.csv")
df.to_csv("combine_all.csv", index=False, sep=",", na_rep='null', encoding="utf-8")

