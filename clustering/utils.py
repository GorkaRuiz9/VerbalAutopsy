# Para funciones complementarias
import numpy as np
import pandas as pd
from clustering.distances import inter_group_distance
import math
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt


def calcular_distancias(clusters, linkage, metric, p, dist_prev, i, j):
    n = len(clusters)
    distancias = [[math.inf for _ in range(n)] for _ in range(n)]

    # Primera iteración: calcular todas las distancias desde cero
    if dist_prev is None or i is None or j is None:
        for a in range(n):
            for b in range(a + 1, n):
                d = inter_group_distance(clusters[a].datos, clusters[b].datos, linkage, metric, p)
                distancias[a][b] = distancias[b][a] = d
        return distancias

    # --- Actualización incremental ---
    # dist_prev tiene tamaño (n+1) x (n+1) porque antes había un cluster más
    # i y j fueron eliminados, y se añadió uno nuevo al final

    # Copiamos todas las distancias que siguen siendo válidas
    idx_antiguos = [k for k in range(len(dist_prev)) if k not in (i, j)]
    for a_new, a_old in enumerate(idx_antiguos[:-1]):  # el último índice es el nuevo cluster
        for b_new, b_old in enumerate(idx_antiguos[:-1]):
            if a_new != b_new:
                distancias[a_new][b_new] = dist_prev[a_old][b_old]

    # Recalcular distancias del nuevo cluster (último índice)
    new_idx = n - 1
    for k in range(n - 1):
        d = inter_group_distance(clusters[new_idx].datos, clusters[k].datos, linkage, metric, p)
        distancias[new_idx][k] = distancias[k][new_idx] = d

    return distancias
            
def get_min_dist(distancias):
    arr = np.array(distancias)
    i, j = np.unravel_index(np.argmin(arr), arr.shape)
    min_dist = arr[i, j]
    return min_dist, (i, j)

def get_results_df(clusters, data_set: pd.DataFrame):
    filas = []

    for cluster in clusters:
        for instancia in cluster.datos:
            fila = list(instancia) + [cluster.id]
            filas.append(fila)

    columnas = list(data_set.columns) + ["cluster"]
    df = pd.DataFrame(filas, columns=columnas)
    
    return df

def plt_dendrogram(clusters_history):
    d, n = build_linkage_matrix(clusters_history)
    dendrogram(d, labels=list(range(n)))
    plt.xlabel("Instancias originales")
    plt.ylabel("Distancia")
    plt.show()

def build_linkage_matrix(clusters_history):
    linkage_matrix = []
    n = next((i for i, c in enumerate(clusters_history) if len(c.datos) > 1), None)
    cluster_sizes = {}

    for cluster in clusters_history:
        if cluster.left is not None and cluster.right is not None:
            i = cluster.left.id
            j = cluster.right.id
            d = cluster.distance
            size = len(cluster.datos)
            linkage_matrix.append([i, j, d, size])
            cluster_sizes[cluster.id] = size

    return np.array(linkage_matrix), n
