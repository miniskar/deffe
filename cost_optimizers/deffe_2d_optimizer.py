import sys
import os
import pdb
import re
import numpy as np
import pandas as pd

class Deffe2DSampler:
    def __init__(self, cost_objective=[]):
        min_max = [re.sub(r'.*::', '', x) if re.search(r'::', x) else 'min' for x in cost_objective]
        self.min_max = [True if x == 'min' else False for x in min_max]
        for x in range(len(self.min_max), 2, 1):
            self.min_max.append(True)
        self.cost_names = [re.sub(r'::.*', '', x) for x in cost_objective]
        self.deviation = 0.0

    def GetParetoDataCore(self, xydata):
        best_point = 0
        prev_best_point = 0 
        first_point = True
        pareto_points = []
        for index in range(xydata.shape[0]):
            if xydata[index][0] == xydata[best_point][0]:
                if xydata[index][1] < xydata[best_point][1]:
                    best_point = index
            else:
                is_best = False
                if first_point or xydata[best_point][1] < xydata[prev_best_point][1]:
                    is_best = True
                if is_best:
                    pareto_points.append(best_point)
                    first_point = False
                    prev_best_point = best_point
                best_point = index
        is_best = False
        if first_point or xydata[best_point][1] < xydata[prev_best_point][1]:
            is_best = True
        if is_best:
            pareto_points.append(best_point)
        return pareto_points

    def GetParetoData(self, xydata, anndata, count, levels=sys.maxsize):
        oxydata = xydata.copy()
        all_pareto_points = np.array([]).astype('int')
        if count > xydata.shape[0]:
            count = xydata.shape[0]
        xyindexes = np.arange(xydata.shape[0]).astype('int')
        while len(all_pareto_points) < count and levels>0:
            #print("-------")
            pareto_points = self.GetParetoDataCore(xydata)
            all_pareto_points = np.append(all_pareto_points, xyindexes[pareto_points], 0)
            #print(xydata)
            #print(pareto_points)
            #print(all_pareto_points)
            xydata = np.delete(xydata, pareto_points, axis=0)
            xyindexes = np.delete(xyindexes, pareto_points)
            levels = levels - 1
        annout = all_pareto_points
        if len(anndata)>0:
            annout = anndata[all_pareto_points.tolist()]
        return oxydata[all_pareto_points], annout

    # Mandatory function to find next optimal set of samples for evaluation
    # Return pareto points 
    def Run(self, parameters, cost, count):
        if cost.size == 0:
            return np.array([])
        columns = cost.columns.tolist()
        total_rows = cost.shape[0]
        columns.remove('Sample')
        cost = cost.sort_values(columns, ascending=self.min_max)
        #print(cost)
        #xydata = cost[columns[0:2]].values.transpose().tolist()
        xydata = cost[columns[0:2]].values
        anndata = cost['Sample'].values # list(range(total_rows)) 
        xydata = xydata.astype("float")
        pareto_xydata, pareto_anndata = self.GetParetoData(xydata, anndata, count)
        #print(pareto_xydata)
        return pareto_anndata

# Mandatory function to return class object
def GetObject(*args):
    ds = Deffe2DSampler(*args)
    return ds
    

def main():
    ds = Deffe2DSampler()
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
