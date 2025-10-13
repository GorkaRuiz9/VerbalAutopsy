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
