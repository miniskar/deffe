import sys
import os
import pdb
import re
import numpy as np
import pandas as pd

class Deffe1DSampler:
    def __init__(self):
        None

    # Mandatory function to find next optimal set of samples for evaluation
    def Run(self, parameters, cost, count):
        if cost.size == 0:
            return np.array([])
        #pdb.set_trace()
        single_cost = cost.columns[0]
        for hdr in cost.columns:
            if not cost[hdr].isnull().values.all():
                single_cost = hdr
                break
        if cost[single_cost].isnull().values.all():
            return np.array([])
        cost_sorted = cost.sort_values(single_cost)
        return cost_sorted['Sample'].values[:count]

# Mandatory function to return class object
def GetObject(*args):
    return Deffe1DSampler(*args)

def main():
    opt1d = Deffe1DSampler()

if __name__ == "__main__":
    main()
