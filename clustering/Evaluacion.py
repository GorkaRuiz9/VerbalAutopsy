import pandas as pd
import numpy as np
from itertools import combinations
from math import isclose


from distances import heuclidean_distance, manhattan_distance, minkowski_distance, sentence_distance, inter_group_distance


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

def load_centroids(path, cluster_col=None):
    """
    Carga un dataset de centroides.

    Parámetros
    ----------
    path : str
        Ruta del archivo CSV.
    cluster_col : str o int, opcional
        Nombre o índice de la columna que contiene el número de cluster.

    Retorna
    -------
    centroids : dict
        Diccionario {cluster_label: np.ndarray(embedding)}.
    """

    df = pd.read_csv(path)

    # Convertir nombre de columna a índice si hace falta
    if isinstance(cluster_col, str):
        cluster_col = df.columns.get_loc(cluster_col)

    # Extraer labels
    labels = df.iloc[:, cluster_col].values.astype(int)

    # Quitar columna del cluster para quedarnos solo con embeddings
    emb_df = df.drop(columns=[df.columns[cluster_col]])
    embeddings = emb_df.values.astype(float)

    # Construir diccionario {cluster_label: vector}
    centroids = {int(lbl): embeddings[i] for i, lbl in enumerate(labels)}

    return centroids

def compute_cohesion(embeddings, labels, centroids=None, squared=False, metric='euclidean', p=2):
    """
    Calcula la cohesión de cada cluster y global con distintas métricas de distancia.

    Parámetros
    ----------
    embeddings : np.ndarray
        Matriz (n_samples, n_features) con los embeddings.
    labels : np.ndarray
        Vector (n_samples,) con el número de cluster asignado a cada instancia.
    centroids : dict, opcional
        Diccionario {cluster_label: embedding_vector}. Si no se pasa, se calcula.
    squared : bool, opcional
        Si True, devuelve distancias al cuadrado (solo válido para métricas numéricas).
    metric : str
        'euclidean', 'manhattan', 'minkowski', 'sentence'
    p : int
        Parámetro de Minkowski.
    """
    # Selección de función de distancia
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

    unique_clusters = np.unique(labels)
    n = embeddings.shape[0]
    per_cluster = {}
    total_sum = 0.0

    for cluster in unique_clusters:
        mask = (labels == cluster)
        cluster_points = embeddings[mask]
        if cluster_points.shape[0] == 0:
            continue

        # Calcular centroid si no se proporciona
        centroid = centroids[cluster] if centroids and cluster in centroids else cluster_points.mean(axis=0)

        # Calcular distancias del cluster a su centroide
        dists = np.array([base_distance(x, centroid) for x in cluster_points])
        dists_sq = dists**2

        per_cluster[cluster] = {
            "size": cluster_points.shape[0],
            "mean_distance": float(dists.mean()),
            "mean_squared_distance": float(dists_sq.mean())
        }

        total_sum += (dists_sq.mean() if squared else dists.mean()) * cluster_points.shape[0]

    overall = total_sum / n
    return per_cluster, overall



def compute_separability(centroids, metric='euclidean', p=2, linkage='mean'):
    """
    Calcula la separabilidad entre centroides según la métrica elegida,
    utilizando inter_group_distance() para garantizar consistencia.

    Parámetros
    ----------
    centroids : dict
        Diccionario {cluster_label: embedding_vector}.
    metric : str
        'euclidean', 'manhattan', 'minkowski', 'sentence'
    p : int
        Parámetro de Minkowski (si aplica).
    linkage : str
        Tipo de enlace a usar en inter_group_distance ('mean', 'single', 'complete', 'average').

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
    labels = sorted(centroids.keys())
    k = len(labels)
    if k < 2:
        return {
            "pairwise_matrix": None,
            "mean_pairwise": None,
            "min_pairwise": None,
            "max_pairwise": None,
            "k": k
        }

    # Crear matriz de distancias
    pairwise = np.zeros((k, k))

    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i == j:
                pairwise[i, j] = 0.0
            else:
                # inter_group_distance espera arrays/lists de instancias → pasamos cada centroide como [vector]
                pairwise[i, j] = inter_group_distance(
                    [np.asarray(centroids[a])],
                    [np.asarray(centroids[b])],
                    linkage=linkage,
                    metric=metric,
                    p=p
                )

    # Extraer los valores únicos de la parte superior (sin duplicar)
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

def main():
    # ---------------- Configuración ----------------
    config = {
        "instances_path": "instances.csv",   # ruta de dataset con embeddings
        "centroids_path": "centroids.csv",   # ruta de centroides
        "id_col": "id",                      # columna de ID (nombre o índice)
        "cluster_col": "cluster"             # columna de cluster (nombre o índice)
    }

    # ---------------- Carga de datos ----------------
    ids, labels, embeddings = load_dataset(
        path=config["instances_path"],
        id_col=config["id_col"],
        cluster_col=config["cluster_col"]
    )

    centroids = load_centroids(
        path=config["centroids_path"],
        cluster_col=config["cluster_col"]
    )

    print("\n Datos cargados correctamente")
    print(f"Instancias: {len(ids)}, Dimensión embeddings: {embeddings.shape[1]}, Clusters: {len(centroids)}")

    # ---------------- Cohesión ----------------
    per_cluster_cohesion, cohesion_global = compute_cohesion(
        embeddings, labels, centroids=centroids, squared=False, metric="euclidean"
    )
    print("\n=== Cohesión por cluster ===")
    for cluster, stats in per_cluster_cohesion.items():
        print(f"Cluster {cluster}: size={stats['size']}, mean_distance={stats['mean_distance']:.4f}, mean_squared_distance={stats['mean_squared_distance']:.4f}")
    print(f"Cohesión global: {cohesion_global:.4f}")

    # ---------------- Separabilidad ----------------
    separability = compute_separability(centroids, metric="euclidean")
    print("\n=== Separabilidad entre centroides ===")
    print(f"n_clusters = {separability['k']}, mean pairwise centroid distance = {separability['mean_pairwise']:.4f}")
    print(f"min = {separability['min_pairwise']:.4f}, max = {separability['max_pairwise']:.4f}")

    # ---------------- Silhouette ----------------
    sil_global, sil_per_cluster = compute_silhouette(embeddings, labels, metric="euclidean")
    print("\n=== Silhouette ===")
    print(f"Silhouette global: {sil_global:.4f}")
    for cluster, val in sil_per_cluster.items():
        print(f"Cluster {cluster}: {val:.4f}")

if __name__ == "__main__":
    main()
