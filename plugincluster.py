#coding=utf-8
import scipy
import scipy.cluster.hierarchy as sch
#import coloredlogs
#import logging
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

#l=logging.getLogger("NEUZZ")
#l.setLevel("INFO")
#fmt = "%(asctime)-15s %(filename)s:%(lineno)d %(process)d %(levelname)s %(message)s"
#coloredlogs.install(fmt=fmt)

#linkage_method={"single","average","complete","weighted","centroid","median","ward"}
linkage_method={"complete"}

num=0;            
def getcluster(dismatrix):
    #print "start" 
    #print dismatrix
    #l.warn("using %s method", item)
    dismatrix = scipy.spatial.distance.squareform(dismatrix)
    Z = sch.linkage(dismatrix, method="average")
    P=sch.dendrogram(Z, p=12, truncate_mode="lastp",leaf_rotation=90.,show_leaf_counts=True, leaf_font_size=8., show_contracted=True, labels=None) 
    plt.savefig("test"+str(num)+".png")
    global num
    num=num+1
    
    cluster = sch.fcluster(Z,t=0.8, criterion="inconsistent")
    #l.info("\ncluster1 to %d, t is 0.8", cluster.max())
    #indices = get_cluster_indices(cluster)
    #for k, ind in enumerate(indices):
    #    print( "cluster %d is %s"%( k + 1,  ind))

    #r= sch.cophenet(Z, dismatrix)
    #r2= sch.inconsistent(Z)
        
    result= get_selected_indices(cluster)
    #print "result"
    #print result
    return result 

def get_cluster_indices(cluster):
    '''映射每一类至原数据索引
    Arguments:
        cluster_assignments 层次聚类后的结果
    Returns:
        [[idx1, idx2,..], [idx3]] -- 每一类下的索引
    '''
    n = cluster.max()
    indices = []
    for cluster_number in range(1, n + 1):
        indices.append(np.where(cluster == cluster_number)[0])
    return indices

def get_selected_indices(cluster):
    n = cluster.max()
    selected=list()
    for cluster_number in range(1, n + 1):
        curcluster= np.where(cluster == cluster_number)[0]
        input = np.random.choice(curcluster, 1, replace=False)[0]
        selected.append( input)
    return np.array(selected)

def main():	
    points=scipy.randn(20,1)  
    disMat = sch.distance.pdist(points,'euclidean')
    getcluster(disMat)
    
if __name__ == "__main__":
    main()

