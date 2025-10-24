import pandas as pd
import numpy as np
from itertools import combinations
from math import isclose


from clustering.distances import heuclidean_distance, manhattan_distance, minkowski_distance, sentence_distance, inter_group_distance, intra_group_distance


def load_dataset(path, id_col=None, cluster_col=None):
    """
    Carga un dataset con embeddings y etiquetas de cluster.

    Parámetros
    ----------
    path : str
        Ruta del archivo CSV.
    id_col : str o int, opcional
        Nombre o índice de la columna que contiene el ID de la instancia.
    cluster_col : str o int, opcional
        Nombre o índice de la columna que contiene el número de cluster.

    Retorna
    -------
    ids : np.ndarray
        Vector con los IDs de instancia.
    labels : np.ndarray
        Vector con los labels de cluster.
    embeddings : np.ndarray
        Matriz con los embeddings (solo columnas numéricas relevantes).
    """

    df = pd.read_csv(path)

    # Convertir nombres de columna a índices si hace falta
    if isinstance(id_col, str):
        id_col = df.columns.get_loc(id_col)
    if isinstance(cluster_col, str):
        cluster_col = df.columns.get_loc(cluster_col)

    # IDs
    ids = df.iloc[:, id_col].values if id_col is not None else np.arange(len(df))

    # Labels (número de cluster)
    labels = df.iloc[:, cluster_col].values if cluster_col is not None else None

    # Filtrar columnas de embeddings (todas excepto id y cluster)
    drop_cols = []
    if id_col is not None:
        drop_cols.append(df.columns[id_col])
    if cluster_col is not None:
        drop_cols.append(df.columns[cluster_col])

    emb_df = df.drop(columns=drop_cols)
    embeddings = emb_df.values.astype(float)

    return ids, labels, embeddings


def compute_cohesion(embeddings, labels, metric='euclidean', p=2, mode='mean'):
    """
    Calcula la cohesión de cada cluster y la cohesión global usando la función intra_group_distance().

    Parámetros
    ----------
    embeddings : np.ndarray
        Matriz (n_samples, n_features) con los embeddings.
    labels : np.ndarray
        Vector (n_samples,) con el número de cluster asignado a cada instancia.
    metric : str
        'euclidean', 'manhattan', 'minkowski', 'sentence'
    p : int
        Parámetro de Minkowski (si metric='minkowski').
    mode : str
        'mean'  -> promedio de distancias al centroide (default)
        'pairs' -> promedio entre todas las parejas del cluster
        'max'   -> distancia máxima interna (diámetro del cluster)
    """
    # Se identifican los clusters únicos y se inicializan estructuras para guardar resultados
    unique_clusters = np.unique(labels)
    per_cluster = {}
    total_weighted_sum = 0.0
    total_points = embeddings.shape[0]
    # Se recorre cada cluster para calcular su cohesión individual
    for cluster in unique_clusters:
        mask = (labels == cluster)
        cluster_points = embeddings[mask]

        if len(cluster_points) == 0:
            continue
        # Cálculo de la cohesión del cluster mediante la función intra_group_distance()
        cohesion_value = intra_group_distance(cluster_points, metric=metric, p=p, mode=mode)
        # Se almacena el tamaño del cluster y su cohesión
        per_cluster[cluster] = {
            "size": len(cluster_points),
            "cohesion": float(cohesion_value)
        }
        # Acumulación ponderada para el cálculo de cohesión global
        total_weighted_sum += cohesion_value * len(cluster_points)
    # Se obtiene la cohesión promedio global
    overall = total_weighted_sum / total_points if total_points > 0 else 0.0

    return per_cluster, overall




def compute_separability(embeddings, labels, metric='euclidean', p=2, linkage='mean'):
    """
    Calcula la separabilidad entre clusters según la métrica elegida,
    utilizando inter_group_distance() para garantizar consistencia.

    Parámetros
    ----------
    embeddings : np.ndarray (n_samples, n_features)
        Matriz con todos los embeddings.
    labels : np.ndarray (n_samples,)
        Vector con la asignación de cluster para cada instancia.
    metric : str
        'euclidean', 'manhattan', 'minkowski', 'sentence'
    p : int
        Parámetro de Minkowski si aplica.
    linkage : str
        Tipo de enlace para inter_group_distance ('mean', 'single', 'complete', 'average').

    Retorna
    -------
    result : dict
        {
            "pairwise_matrix": np.ndarray(k, k),
            "mean_pairwise": float,
            "min_pairwise": float,
            "max_pairwise": float,
            "k": int
        }
    """
    unique_clusters = np.unique(labels)
    k = len(unique_clusters)
    if k < 2:
        return {
            "pairwise_matrix": None,
            "mean_pairwise": None,
            "min_pairwise": None,
            "max_pairwise": None,
            "k": k
        }

    # Extraer puntos de cada cluster
    clusters = {c: embeddings[labels == c] for c in unique_clusters}

    # Crear matriz de distancias
    pairwise = np.zeros((k, k))

    for i, ci in enumerate(unique_clusters):
        for j, cj in enumerate(unique_clusters):
            if i == j:
                pairwise[i, j] = 0.0
            else:
                pairwise[i, j] = inter_group_distance(
                    clusters[ci],
                    clusters[cj],
                    linkage=linkage,
                    metric=metric,
                    p=p
                )

    # Extraer valores únicos de la parte superior para estadísticas
    upper_vals = np.array([pairwise[i, j] for i, j in combinations(range(k), 2)])

    return {
        "pairwise_matrix": pairwise,
        "mean_pairwise": float(upper_vals.mean()),
        "min_pairwise": float(upper_vals.min()),
        "max_pairwise": float(upper_vals.max()),
        "k": k
    }



def compute_silhouette(embeddings, labels, metric='euclidean', p=2):
    """
    Calcula el coeficiente silhouette medio y por cluster con distintas métricas.
    """
    if metric == 'euclidean':
        base_distance = heuclidean_distance
    elif metric == 'manhattan':
        base_distance = manhattan_distance
    elif metric == 'minkowski':
        base_distance = lambda a, b: minkowski_distance(a, b, p=p)
    elif metric == 'sentence':
        base_distance = sentence_distance
    else:
        raise ValueError("metric debe ser 'euclidean', 'manhattan', 'minkowski' o 'sentence'")

    n = embeddings.shape[0]
    unique_clusters = np.unique(labels)
    silhouettes = np.zeros(n)

    # Calcular matriz de distancias
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = base_distance(embeddings[i], embeddings[j])

    for i in range(n):
        lbl = labels[i]
        same_mask = (labels == lbl)
        same_mask[i] = False
        a = D[i, same_mask].mean() if same_mask.sum() > 0 else 0.0

        b_vals = [
            D[i, labels == other].mean()
            for other in unique_clusters if other != lbl and np.sum(labels == other) > 0
        ]
        b = min(b_vals) if b_vals else 0.0

        denom = max(a, b)
        silhouettes[i] = (b - a) / denom if not isclose(denom, 0.0) else 0.0

    silhouette_per_cluster = {
        int(lbl): float(silhouettes[labels == lbl].mean()) for lbl in unique_clusters
    }

    return float(silhouettes.mean()), silhouette_per_cluster

def get_metrics(path, metric, p, mode="mean"):
    # ---------------- Configuración ----------------
    config = {
        "instances_path": path,   # ruta de dataset con embeddings
        "id_col": "newid",                      # columna de ID (nombre o índice)
        "cluster_col": "cluster"             # columna de cluster (nombre o índice)
    }

    # ---------------- Carga de datos ----------------
    ids, labels, embeddings = load_dataset(
        path=config["instances_path"],
        id_col=config["id_col"],
        cluster_col=config["cluster_col"]
    )

    print("\n Datos cargados correctamente")
    print(f"Instancias: {len(ids)}, Dimensión embeddings: {embeddings.shape[1]}")

    # ---------------- Cohesión ----------------
    per_cluster_cohesion, cohesion_global = compute_cohesion(
        embeddings=embeddings,
        labels=labels,
        metric=metric,
        p=p,
        mode=mode
    )

    print("\n=== Cohesión por cluster ===")
    for cluster, stats in per_cluster_cohesion.items():
        print(f"Cluster {cluster}: size={stats['size']}, cohesion={stats['cohesion']:.4f}")

    print(f"\nCohesión global: {cohesion_global:.4f}")

    # ---------------- Separabilidad ----------------
    separability = compute_separability(embeddings, labels, metric="euclidean", linkage='mean')
    print("\n=== Separabilidad entre clusters ===")
    print(f"n_clusters = {separability['k']}, mean pairwise distance = {separability['mean_pairwise']:.4f}")
    print(f"min = {separability['min_pairwise']:.4f}, max = {separability['max_pairwise']:.4f}")


    # ---------------- Silhouette ----------------
    sil_global, sil_per_cluster = compute_silhouette(embeddings, labels, metric="euclidean")
    print("\n=== Silhouette ===")
    print(f"Silhouette global: {sil_global:.4f}")
    for cluster, val in sil_per_cluster.items():
        print(f"Cluster {cluster}: {val:.4f}")
        
    metrics = {
        "n_instancias": len(ids),
        "dimensiones": embeddings.shape[1],
        "cohesion": cohesion_global,
        "mean_pairwise_distance": separability['mean_pairwise'],
        "min_pairwise_distance": separability['min_pairwise'],
        "max_pairwise_distance": separability['max_pairwise'],
        "silhouette_global": sil_global
    }

    return metrics

if __name__ == "__main__":
    main()
