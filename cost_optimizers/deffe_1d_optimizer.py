import sys
import os
import pdb
import re
import numpy as np
import pandas as pd

class Deffe1DSampler:
    def __init__(self, cost_objective=[]):
        min_max = [re.sub(r'.*::', '', x) if re.search(r'::', x) else 'min' for x in cost_objective]
        self.min_max = [True if x == 'min' else False for x in min_max]
        if len(self.min_max) == 0:
            self.min_max.append(True)
        self.cost_names = [re.sub(r'::.*', '', x) for x in cost_objective]
        #pdb.set_trace()
        self.deviation = 0.0

    # Mandatory function to find next optimal set of samples for evaluation
    def Run(self, parameters, cost, count):
        if cost.size == 0:
            return np.array([])
        #pdb.set_trace()
        if count > cost.shape[0]:
            count = cost.shape[0]
        single_cost = cost.columns[0]
        cost_sorted = cost.sort_values(single_cost, ascending=self.min_max[0])
        return cost_sorted['Sample'].values[:count]

# Mandatory function to return class object
def GetObject(*args):
    return Deffe1DSampler(*args)

def main():
    ds = Deffe1DSampler()
    s = [[1,9],[2,6],[3,10],[3,4],[4,7],[4,2],[5,6],[6,1],[6,4],[7,5],[8,2],[8,5],[9,8]]
    data = pd.DataFrame(s, columns=['Quantity', 'Price'])
    data = data[['Quantity', 'Price']]
    data['Sample'] = list(range(data.shape[0]))
    #print(data)
    data = data.sort_values('Price')
    out = ds.Run(None, data, 100)
    print(out)

if __name__ == "__main__":
    main()
