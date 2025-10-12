# implementacion de la clase AgglomerativeClustering
import pandas
from utils import *
import numpy as np
import copy

class AgglomerativeClustering:
    def __init__(self, linkage="average", metric="euclidean", minkowski_p=3):
        self.linkage = linkage
        self.metric = metric
        self.p = minkowski_p
        self.clusters = list()
        self.clusters_history = list()
        self.labels_ = None

    def fit(self, data_set: pandas.DataFrame):
        # contruye el dendograma a partir del dataset
        self.clusters = [ClusterNode(datos=[dato], id=i) for i, dato in enumerate(data_set.values)]
        self.clusters_history = copy.deepcopy(self.clusters)
        
        while len(self.clusters) > 1:
            distancias = calcular_distancias(self.clusters, self.linkage, self.metric, self.p)
            min_dist, (i, j) = get_min_dist(distancias)
            clusterA = self.clusters[i]
            clusterB = self.clusters[j]
            self.update_clusters_list(clusterA, clusterB, min_dist)
            
        return None
            
            
    def update_clusters_list(self, clusterA, clusterB, min_dist):
        new_cluster = ClusterNode(clusterA, clusterB, min_dist, clusterA.datos + clusterB.datos, len(self.clusters_history))
        self.clusters.remove(clusterA)
        self.clusters.remove(clusterB)
        self.clusters.append(new_cluster)
        self.clusters_history.append(new_cluster)
        
    def get_clusters(self, dist_to_cut):
        
        return None


class ClusterNode:
    def __init__(self, left=None, right=None, distance=0.0, datos=None, id=None):
        self.left = left          
        self.right = right        
        self.distance = distance  
        self.datos = datos    
        self.id = id              