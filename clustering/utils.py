# Para funciones complementarias
import numpy as np
import pandas as pd
from clustering.distances import inter_group_distance
import math
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt


def calcular_distancias(clusters, linkage, metric, p, distancias_prev: dict, i, j):
    
    distancias = {}
    n = len(clusters)
    
    if distancias_prev == None:
        for i in range(n):
            datos_i = extraer_datos(clusters[i].datos)
            for j in range(i+1, n):
                datos_j = extraer_datos(clusters[j].datos)
                distancias[(clusters[i].id, clusters[j].id)] = inter_group_distance(datos_i, datos_j, linkage, metric, p)
                distancias[(clusters[j].id, clusters[i].id)] = distancias[(clusters[i].id, clusters[j].id)]

    else:
        distancias = distancias_prev.copy()
        for key in distancias_prev.keys():
            if i in key or j in key:
                del distancias[key]
                
        datos_ultimo = extraer_datos(clusters[-1].datos)
        for k in range(n-1):
            datos_k = extraer_datos(clusters[k].datos)
            distancias[(clusters[k].id, clusters[-1].id)] = inter_group_distance(datos_ultimo, datos_k, linkage, metric, p)
            distancias[(clusters[-1].id, clusters[k].id)] = distancias[(clusters[k].id, clusters[-1].id)]
            
    return distancias


def extraer_datos(d):
    return [fila[1] for fila in d]

            
def get_min_dist(distancias):
    min_key = min(distancias, key=distancias.get)  # clave con valor mínimo
    min_val = distancias[min_key]
    return min_val, min_key

def get_results(clusters, data_set: pd.DataFrame):
    filas = []

    for cluster in clusters:
        for instancia in cluster.datos:
            fila = [instancia[0]] + list(instancia[1]) + [cluster.id]
            filas.append(fila)

    columnas = list(data_set.columns) + ["cluster"]
    df = pd.DataFrame(filas, columns=columnas)
    
    return df

def plt_dendrogram(clusters_history, show, linkage, metric, f_name):
    d, n = build_linkage_matrix(clusters_history)
    dendrogram(d, labels=list(range(n)))
    plt.xlabel("Instancias originales")
    plt.ylabel("Distancia")
    plt.title(f"Dendograma con {linkage} y {metric}")
    plt.savefig(f"./output/{f_name}.png")
    if show:
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
            size = len(cluster.datos[1])
            linkage_matrix.append((i, j, d, size))
            cluster_sizes[cluster.id] = size
    
    return np.array(linkage_matrix), n
