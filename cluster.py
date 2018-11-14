#coding=utf-8
import os
import Levenshtein
import scipy
import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import coloredlogs
import logging
l=logging.getLogger("NEUZZ")
l.setLevel("INFO")
fmt = "%(asctime)-15s %(filename)s:%(lineno)d %(process)d %(levelname)s %(message)s"
coloredlogs.install(fmt=fmt)

linkage_method={"single","average","complete","weighted","centroid","median","ward"}
linkage_method={"complete"}

class AFLCluster():
    def __init__(self):
        # set the data dir and get an ordered inputs list
        self.datadir="/dev/shm/collect-fuzzed/fuzzed"
        self.inputs=None
        self._get_inputs()
        self.inputs_num=len(self.inputs)
        #self.inputs_num=20
        self.labels=self.inputs[0:self.inputs_num]

        # init distance matrix
        self.use_old_distance =False
        self.distance_matrix=scipy.zeros([self.inputs_num,self.inputs_num])
        self.distancefile="/home/xiaosatianyu/learning/distance.npy"

    def _get_inputs(self):
        inputs = os.listdir(self.datadir)
        inputs.sort()
        try:
            inputs.remove(".state")
        except:
            pass
        self.inputs=inputs

    def get_distance_matrix(self):
        l.info("ready to calculate the distance")
        if os.path.exists(self.distancefile) and self.use_old_distance:
            self._load_distance_matrix()
            return
        for i in range(0, self.inputs_num):
            #l.info("process %d", i)
            for j in range(i+1, self.inputs_num):
                file1=os.path.join(self.datadir,self.inputs[i])
                file2=os.path.join(self.datadir,self.inputs[j])
                with open(file1, "rb") as f1:
                    content1 = f1.read()
                    #fsize1 = os.path.getsize(file1)
                with open(file2, "rb") as f2:
                    content2 = f2.read()
                    #fsize2 = os.path.getsize(file2)
                distance = Levenshtein.distance(content1, content2)
                self.distance_matrix[i][j] = distance
                self.distance_matrix[j][i] = distance
                #tag1=inputs[i][0:10]
                #tag2=inputs[j][0:10]
                #print "the distance between %s (length: %d) and %s (length: %d) is %d"%(tag1,fsize1,tag2, fsize2, d)
        l.info("end calcualting the distance matrix")
        self.distance_matrix=scipy.spatial.distance.squareform(self.distance_matrix)

        #l.info(self.distance_matrix)
        #print self.distance_matrix
        # save the distance matrix
        self._save_distance_matrix()

    def _save_distance_matrix(self):
        np.save(self.distancefile, self.distance_matrix)
        l.info("save the distance matrix")
    def _load_distance_matrix(self):
        self.distance_matrix=np.load(self.distancefile)
        l.info("load distance matrix from old")

    def hierarchy_cluster(self):
        #get the distance_matrix
        self.get_distance_matrix()
        l.info("the max distance is %d", self.distance_matrix.max())
        
        for item in linkage_method:
            l.warn("using %s method", item)
            Z = sch.linkage(self.distance_matrix, method=item)
            cluster_assignments = sch.fcluster(Z,t=self.inputs_num/5, criterion="maxclust")
            l.info("cluster to %d ", cluster_assignments.max())
            #indices = self.get_cluster_indices(cluster_assignments)
            #for k, ind in enumerate(indices):
            #    l.info( "cluster %d is %s", k + 1,  ind)
            self.get_plot(Z, item)
            r= sch.cophenet(Z, self.distance_matrix)
            r2= sch.inconsistent(Z)
            l.info(r[1])

    def get_cluster_indices(self,cluster_assignments):
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

    def get_plot(self, Z, name):
        plt.figure(figsize=(15, 10))
        P=sch.dendrogram(Z, p=12, truncate_mode="lastp", leaf_rotation=90.,show_leaf_counts=True, leaf_font_size=8., show_contracted=True, labels=None)
        plt.savefig("result"+name+".png")


def main():
    aflcluster=AFLCluster()
    aflcluster.hierarchy_cluster()

if __name__ == "__main__":
    main()

