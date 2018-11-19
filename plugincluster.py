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
    print "first" 
    for item in linkage_method:
        #l.warn("using %s method", item)
        Z = sch.linkage(dismatrix, method=item)
        cluster_assignments = sch.fcluster(Z,t=2, criterion="maxclust")
        #l.info("cluster to %d ", cluster_assignments.max())
        #r= sch.cophenet(Z, dismatrix)
        #r2= sch.inconsistent(Z)
        #l.info(r[1])
    print "ok"
    return dismatrix

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

def main():	
    hierarchy_cluster(arr)
    
if __name__ == "__main__":
    main()

