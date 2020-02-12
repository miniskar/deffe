import numpy as np
import pdb

def dist(a, b):
    return b-a

def clusterupdate(a, b):
    return max(a, b)

class Hierarchical:
    def __init__(self, n_clusters=1, init=None, distcore=dist, clusterupdate_core=clusterupdate):
        self.n_clusters = n_clusters
        self.distcore = dist
        self.clusterupdate_core = clusterupdate_core

    def fit(self, data):
        cdata = np.array(data)
        maxval = max(cdata)+1
        current_clusters = np.frompyfunc(list,0,1)(np.empty(cdata.size, dtype=object))
        for index in range(cdata.size):
            current_clusters[index].append(index)
        dist_vec = np.array([ self.distcore(cdata[index-1], cdata[index]) if index >=1 else maxval for index in range(cdata.size)]).astype('int')
        #print(current_clusters)
        prev_len = len(cdata)
        #print("***************************************")
        #print("data: "+str(cdata))
        #print("cluster: "+str(current_clusters))
        #print("dist: "+str(dist_vec))
        while len(cdata) > self.n_clusters: 
            #print("***************************************")
            #print("data: "+str(cdata))
            #print("cluster: "+str(current_clusters))
            #print("dist: "+str(dist_vec))
            if len(dist_vec) == 0:
                pdb.set_trace()
                None
            mindist_arg = np.argmin(dist_vec)
            # Update cluster data, 
            cdata[mindist_arg] = self.clusterupdate_core(cdata[mindist_arg-1], cdata[mindist_arg])
            current_clusters[mindist_arg].extend(current_clusters[mindist_arg-1])
            dist_vec[mindist_arg] = self.distcore(cdata[mindist_arg-2], cdata[mindist_arg]) if mindist_arg >= 2 else maxval
            # Delete 
            cdata = np.delete(cdata, mindist_arg-1)
            current_clusters = np.delete(current_clusters, mindist_arg-1)
            dist_vec = np.delete(dist_vec, mindist_arg-1)
        #print("***************************************")
        #print("out data: "+str(cdata))
        #print("out cluster: "+str(current_clusters))
        self.labels_ = np.zeros(len(data))
        self.clusters = current_clusters
        self.cluster_centers_ = cdata
        for cindex, cluster in enumerate(current_clusters):
            for obj in cluster:
                self.labels_[obj] = cindex
        self.labels_ = self.labels_.astype('int')

def main():
    hierarchical = Hierarchical(n_clusters=3)
    s = np.array([100, 101, 105, 106, 110, 120, 130, 150])
    hierarchical.fit(s)
    print(hierarchical.labels_)
    print(hierarchical.cluster_centers_)

if __name__ == "__main__":
    main()
