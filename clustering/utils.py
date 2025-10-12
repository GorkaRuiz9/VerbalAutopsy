# Para funciones complementarias
import numpy as np
import pandas
from linkage import *
import math

def calcular_distancias(clusters, linkage, metric, p):
    distancias = [[math.inf if i == j else 0.0 for i in range(len(clusters))] for j in range(len(clusters))]
    # calcular matriz de distancias
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            distancias[i][j] = distancias[j][i] = inter_group_distance(clusters[i].datos, clusters[j].datos, linkage, metric, p)
            
    return distancias
            
def get_min_dist(distancias):
    arr = np.array(distancias)
    i, j = np.unravel_index(np.argmin(arr), arr.shape)
    min_dist = arr[i, j]
    return min_dist, (i, j)