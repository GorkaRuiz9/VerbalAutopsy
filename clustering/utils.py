# Para funciones complementarias
import numpy as np
import pandas as pd
from distances import inter_group_distance
import math
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt


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

def get_results_df(clusters, data_set: pd.DataFrame):
    filas = []

    for cluster in clusters:
        for instancia in cluster.datos:
            fila = list(instancia) + [cluster.id]
            filas.append(fila)

    columnas = list(data_set.columns) + ["cluster"]
    df = pd.DataFrame(filas, columns=columnas)

    df.insert(0, "id", None)
    for i, row in df.iterrows():
        # buscamos la instancia original en data_set
        mask = (data_set == row[data_set.columns]).all(axis=1)
        match = data_set[mask]
        if not match.empty:
            df.at[i, "id"] = match.index[0]
        else:
            df.at[i, "id"] = -1  # por si no se encuentra coincidencia exacta

    df = df.sort_values(by="id").reset_index(drop=True)
    
    print(df)
    return df

def plt_dendrogram(clusters_history):
    d, n = build_linkage_matrix(clusters_history)
    print(list(range(n)))
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
