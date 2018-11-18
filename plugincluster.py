#coding=utf-8
import scipy.cluster.hierarchy as sch
#import coloredlogs
#import logging
import numpy as np
#l=logging.getLogger("NEUZZ")
#l.setLevel("INFO")
#fmt = "%(asctime)-15s %(filename)s:%(lineno)d %(process)d %(levelname)s %(message)s"
#coloredlogs.install(fmt=fmt)

linkage_method={"single","average","complete","weighted","centroid","median","ward"}
linkage_method={"complete"}

            
def hierarchy_cluster(dismatrix):
    
    for item in linkage_method:
        #l.warn("using %s method", item)
        Z = sch.linkage(dismatrix, method=item)
        cluster_assignments = sch.fcluster(Z,t=2, criterion="maxclust")
        #l.info("cluster to %d ", cluster_assignments.max())
        #r= sch.cophenet(Z, dismatrix)
        #r2= sch.inconsistent(Z)
        #l.info(r[1])
    print "ok"

def get_cluster_indices(cluster_assignments):
    '''映射每一类至原数据索引
    
    Arguments:
        cluster_assignments 层次聚类后的结果
    
    Returns:
        [[idx1, idx2,..], [idx3]] -- 每一类下的索引
    '''
    n = cluster_assignments.max()
    indices = []
    for cluster_number in range(1, n + 1):
        indices.append(np.where(cluster_assignments == cluster_number)[0])
    return indices

def main( a):	
    print a.shape
    arr = [[0., 21.6, 22.6, 63.9, 65.1, 17.7, 99.2],
	    [21.6, 0., 1., 42.3, 43.5, 3.9, 77.6],
	    [22.6, 1., 0, 41.3, 42.5, 4.9, 76.6],
	    [63.9, 42.3, 41.3, 0., 1.2, 46.2, 35.3],
	    [65.1, 43.5, 42.5, 1.2, 0., 47.4, 34.1],
	    [17.7, 3.9, 4.9, 46.2, 47.4, 0, 81.5],
	    [99.2, 77.6, 76.6, 35.3, 34.1, 81.5, 0.]]
    arr= np.array(arr)
    hierarchy_cluster(arr)
    return arr.astype(np.float)
    
if __name__ == "__main__":
    main()

