import pandas as pd
import numpy as np

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

def compute_cohesion(embeddings, labels, centroids=None, squared=False):
    """
    Calcula la cohesión de cada cluster y global.

    Parámetros
    ----------
    embeddings : np.ndarray
        Matriz (n_samples, n_features) con los embeddings de las instancias.
    labels : np.ndarray
        Vector (n_samples,) con el número de cluster asignado a cada instancia.
    centroids : dict, opcional
        Diccionario {cluster_label: embedding_vector}. Si no se pasa, se calcula.
    squared : bool, opcional
        Si True devuelve distancias al cuadrado (MSE). Si False, distancias euclidianas.

    Retorna
    -------
    per_cluster : dict
        {cluster_label: {"size": int, "mean_distance": float, "mean_squared_distance": float}}
    overall : float
        Media global ponderada por el tamaño de cada cluster.
    """

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
        if centroids is None or cluster not in centroids:
            centroid = cluster_points.mean(axis=0)
        else:
            centroid = centroids[cluster]

        # Distancias
        diffs = cluster_points - centroid
        dist_sq = np.sum(diffs**2, axis=1)
        dist = np.sqrt(dist_sq)

        per_cluster[cluster] = {
            "size": cluster_points.shape[0],
            "mean_distance": float(dist.mean()),
            "mean_squared_distance": float(dist_sq.mean())
        }

        total_sum += (dist_sq.mean() if squared else dist.mean()) * cluster_points.shape[0]

    overall = total_sum / n
    return per_cluster, overall

from itertools import combinations
import numpy as np

def compute_separability(centroids):
    """
    Calcula la separabilidad entre clusters (distancias entre centroides).

    Parámetros
    ----------
    centroids : dict
        Diccionario {cluster_label: embedding_vector}.

    Retorna
    -------
    result : dict
        {
            "pairwise_matrix": np.ndarray(k,k),
            "mean_pairwise": float,
            "min_pairwise": float,
            "max_pairwise": float,
            "k": int
        }
    """
    labels = sorted(centroids.keys())
    k = len(labels)

    if k < 2:
        return {"pairwise_matrix": None, "mean_pairwise": None, "min_pairwise": None, "max_pairwise": None, "k": k}

    pairwise = np.zeros((k, k))

    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i == j:
                pairwise[i, j] = 0.0
            else:
                pairwise[i, j] = np.linalg.norm(centroids[a] - centroids[b])

    # Extraer upper triangle para estadísticas
    upper_vals = [pairwise[i,j] for i,j in combinations(range(k), 2)]
    upper_vals = np.array(upper_vals)

    return {
        "pairwise_matrix": pairwise,
        "mean_pairwise": float(upper_vals.mean()),
        "min_pairwise": float(upper_vals.min()),
        "max_pairwise": float(upper_vals.max()),
        "k": k
    }

from math import isclose
from itertools import combinations

def compute_silhouette(embeddings, labels):
    """
    Calcula el coeficiente silhouette medio y por cluster.

    Parámetros
    ----------
    embeddings : np.ndarray
        Matriz (n_samples, n_features) con embeddings.
    labels : np.ndarray
        Vector con número de cluster por instancia.

    Retorna
    -------
    silhouette_mean : float
        Silhouette global promedio.
    silhouette_per_cluster : dict
        {cluster_label: silhouette promedio de ese cluster}
    """
    n = embeddings.shape[0]
    unique_clusters = np.unique(labels)
    silhouettes = np.zeros(n)

    # Precalcular matriz de distancias completa
    D = np.linalg.norm(embeddings[:, None, :] - embeddings[None, :, :], axis=2)

    for i in range(n):
        lbl = labels[i]

        # a(i): media de distancias a su cluster (excluyendo self)
        same_mask = (labels == lbl)
        same_mask[i] = False
        if same_mask.sum() == 0:
            a = 0.0  # cluster singleton
        else:
            a = D[i, same_mask].mean()

        # b(i): mínima media de distancias a otros clusters
        b_vals = []
        for other in unique_clusters:
            if other == lbl:
                continue
            other_mask = (labels == other)
            if other_mask.sum() == 0:
                continue
            b_vals.append(D[i, other_mask].mean())
        b = min(b_vals) if b_vals else 0.0

        denom = max(a, b)
        silhouettes[i] = (b - a) / denom if not isclose(denom, 0.0) else 0.0

    # Silhouette por cluster
    silhouette_per_cluster = {}
    for lbl in unique_clusters:
        mask = (labels == lbl)
        silhouette_per_cluster[int(lbl)] = float(silhouettes[mask].mean() if mask.sum() > 0 else 0.0)

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
        embeddings, labels, centroids=centroids, squared=False
    )
    print("\n=== Cohesión por cluster ===")
    for cluster, stats in per_cluster_cohesion.items():
        print(f"Cluster {cluster}: size={stats['size']}, mean_distance={stats['mean_distance']:.4f}, mean_squared_distance={stats['mean_squared_distance']:.4f}")
    print(f"Cohesión global: {cohesion_global:.4f}")

    # ---------------- Separabilidad ----------------
    separability = compute_separability(centroids)
    print("\n=== Separabilidad entre centroides ===")
    print(f"n_clusters = {separability['k']}, mean pairwise centroid distance = {separability['mean_pairwise']:.4f}")
    print(f"min = {separability['min_pairwise']:.4f}, max = {separability['max_pairwise']:.4f}")

    # ---------------- Silhouette ----------------
    sil_global, sil_per_cluster = compute_silhouette(embeddings, labels)
    print("\n=== Silhouette ===")
    print(f"Silhouette global: {sil_global:.4f}")
    for cluster, val in sil_per_cluster.items():
        print(f"Cluster {cluster}: {val:.4f}")

if __name__ == "__main__":
    main()
