# para las funciones de distancia: Heuclidean, Manhattan, Minkowski, Sentence similarity

import numpy as np
from itertools import product
from sklearn.metrics.pairwise import cosine_similarity


# Distancia entre instancias

def heuclidean_distance(x, y):
    x, y = np.array(x), np.array(y)
    return np.sqrt(np.sum((x - y) ** 2))


def manhattan_distance(x, y):
    x, y = np.array(x), np.array(y)
    return np.sum(np.abs(x - y))


def minkowski_distance(x, y, p=2):
    x, y = np.array(x), np.array(y)
    return np.sum(np.abs(x - y) ** p) ** (1 / p)


def sentence_similarity(embedding_a, embedding_b):
    sim = cosine_similarity([embedding_a], [embedding_b])[0][0]
    return sim


def sentence_distance(embedding_a, embedding_b):
    return 1 - sentence_similarity(embedding_a, embedding_b)


# Distancia entre clusters

def inter_group_distance(cluster_a, cluster_b, linkage='single', metric='euclidean', p=2):
    #Parámetros:
    #- cluster_a, cluster_b: arrays o listas con las instancias de cada cluster
    #- linkage: 'single', 'complete', 'average', 'mean'
    #- metric: 'euclidean', 'manhattan', 'minkowski', 'sentence'
    #- p: parámetro para Minkowski (solo si metric='minkowski')
    

    # Selección de la métrica base
    if metric == 'euclidean':
        base_distance = heuclidean_distance
    elif metric == 'manhattan':
        base_distance = manhattan_distance
    elif metric == 'minkowski':
        base_distance = lambda a, b: minkowski_distance(a, b, p)
    elif metric == 'sentence':
        base_distance = sentence_distance
    else:
        raise ValueError("metric debe ser: 'euclidean', 'manhattan', 'minkowski' o 'sentence'")

    # Calcular distancias entre todos los pares (cartesiano)
    distances = [base_distance(a, b) for a, b in product(cluster_a, cluster_b)]

    # Aplicar tipo de enlace
    if linkage == 'single':
        return np.min(distances)
    elif linkage == 'complete':
        return np.max(distances)
    elif linkage == 'average':
        return np.mean(distances)
    elif linkage == 'mean':
        centroid_a = np.mean(cluster_a, axis=0)
        centroid_b = np.mean(cluster_b, axis=0)
        return base_distance(centroid_a, centroid_b)
    else:
        raise ValueError("linkage debe ser: 'single', 'complete', 'average' o 'mean'")


# Pruebas

if __name__ == "__main__":
    # Datos numéricos
    cluster1 = np.array([[1, 2], [2, 3], [3, 3]])
    cluster2 = np.array([[6, 5], [7, 8], [8, 6]])

    print("Distancias numéricas")
    print("Euclidean:", heuclidean_distance([1, 2], [4, 6]))
    print("Manhattan:", manhattan_distance([1, 2], [4, 6]))
    print("Minkowski (p=3):", minkowski_distance([1, 2], [4, 6], p=3))

    for link in ['single', 'complete', 'average', 'mean']:
        d = inter_group_distance(cluster1, cluster2, linkage=link, metric='euclidean')
        print(f"Inter-grupal ({link}, euclidean): {d:.4f}")

    # Datos de texto (embeddings simulados)
    print("\nDistancias semánticas")
    emb1 = np.array([0.2, 0.3, 0.9])
    emb2 = np.array([0.25, 0.4, 0.85])
    emb3 = np.array([-0.1, 0.0, 0.1])

    clusterA = np.array([emb1, emb2])
    clusterB = np.array([emb3])

    for link in ['single', 'complete', 'average', 'mean']:
        d = inter_group_distance(clusterA, clusterB, linkage=link, metric='sentence')
        print(f"Inter-grupal ({link}, sentence): {d:.4f}")
