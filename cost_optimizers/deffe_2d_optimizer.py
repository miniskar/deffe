import sys
import os
import pdb
import re
import numpy as np
import pandas as pd

class Deffe2DSampler:
    def __init__(self):
        self.deviation = 0.0

    def GetParetoData(self, xydata, anndata, deviation=0.0):
        xdata = np.array(xydata[0]).astype("float")
        ydata = np.array(xydata[1]).astype("float")
        best_point = [xdata[0], ydata[0], 0]
        prev_best_point = best_point.copy()
        pareto_point = [[], [], []]
        non_pareto_point = [[], [], []]
        for index in range(xdata.size):
            if xdata[index] == best_point[0]:
                if ydata[index] < best_point[1]:
                    best_point = [xdata[index], ydata[index], index]
            else:
                is_best = False
                is_second_best = False
                if len(pareto_point[1]) == 0 or best_point[1] < prev_best_point[1]:
                    is_best = True
                if len(pareto_point[1]) == 0 or best_point[1] < (
                    prev_best_point[1] + prev_best_point[1] * deviation
                ):
                    is_second_best = True
                if is_best or is_second_best:
                    pareto_point[0].append(best_point[0])
                    pareto_point[1].append(best_point[1])
                    pareto_point[2].append(best_point[2])
                else:
                    non_pareto_point[0].append(best_point[0])
                    non_pareto_point[1].append(best_point[1])
                    non_pareto_point[2].append(best_point[2])
                if is_best:
                    prev_best_point = best_point.copy()
                best_point = [xdata[index], ydata[index], index]
        is_best = False
        is_second_best = False
        if len(pareto_point[1]) == 0 or best_point[1] < prev_best_point[1]:
            is_best = True
        if len(pareto_point[1]) == 0 or best_point[1] < (
            prev_best_point[1] + prev_best_point[1] * deviation
        ):
            is_second_best = True
        if is_best or is_second_best:
            pareto_point[0].append(best_point[0])
            pareto_point[1].append(best_point[1])
            pareto_point[2].append(best_point[2])
        else:
            non_pareto_point[0].append(best_point[0])
            non_pareto_point[1].append(best_point[1])
            non_pareto_point[2].append(best_point[2])
        annout = pareto_point[2]
        if len(anndata)>0:
            annout = np.array(anndata)[pareto_point[2]].tolist()
        return [pareto_point[0], pareto_point[1]], annout, [non_pareto_point[0], non_pareto_point[1]], non_pareto_point[2]

    # Mandatory function to find next optimal set of samples for evaluation
    # Return pareto points 
    def Run(self, parameters, cost, count):
        if cost.size == 0:
            return np.array([])
        columns = cost.columns.tolist()
        total_rows = cost.shape[0]
        columns.remove('Sample')
        cost = cost.sort_values(columns[0])
        xydata = cost[columns[0:2]].values.transpose().tolist()
        anndata = cost['Sample'].values.tolist() # list(range(total_rows)) 
        pareto_xydata, pareto_anndata, non_pareto_xydata, non_pareto_anndata = self.GetParetoData(xydata, anndata, float(self.deviation))
        #print(pareto_xydata)
        #pdb.set_trace()
        return np.array(pareto_anndata)

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
    print(data)
    data = data.sort_values('Price')
    out = ds.Run(None, data, 100)
    print(out)

if __name__ == "__main__":
    main()
