# Para funciones complementarias
import numpy as np
import pandas as pd
from distances import inter_group_distance
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

def get_results_df(clusters):
    filas = []

    for cluster in clusters:
        for instancia in cluster.datos:
            fila = [None] + list(instancia) + [cluster.id]  # None temporal para id
            filas.append(fila)

    # crear dataframe
    num_atributos = len(clusters[0].datos[0])
    columnas = ["id"] + [f"atrib{i+1}" for i in range(num_atributos)] + ["id_cluster"]
    df = pd.DataFrame(filas, columns=columnas)

    # asignar id único por fila
    df["id"] = range(len(df))
    print(df)
    return df
