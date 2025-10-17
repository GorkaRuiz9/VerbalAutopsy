from contextlib import nullcontext
import numpy as np
import pandas as pd


def load_embeddings(path, id_col=None, cluster_col=None):
    import pandas as pd
    df = pd.read_csv(path)

    posibles_ids = ["id", "newid", "ID", "Id"]
    if id_col is None or id_col not in df.columns:
        for posible in posibles_ids:
            if posible in df.columns:
                id_col = posible
                break
        else:
            raise KeyError(f"No se encontró ninguna columna de ID entre {posibles_ids} en {path}")

    if cluster_col and cluster_col not in df.columns:
        cluster_col = None

    ids = np.array(df[id_col].tolist())
    clusters = np.array(df[cluster_col].tolist()) if cluster_col else None
    embeddings = df.drop(columns=[id_col] + ([cluster_col] if cluster_col else []), errors="ignore").values
    return ids, clusters, embeddings


def load_centroids(path, cluster_col=None):
    df = pd.read_csv(path)
    if isinstance(cluster_col, str):
        cluster_col = df.columns.get_loc(cluster_col)
    labels = df.iloc[:, cluster_col].values.astype(int)
    emb = df.drop(columns=[df.columns[cluster_col]]).values.astype(float)
    return {int(lbl): emb[i] for i, lbl in enumerate(labels)}


def assign_cluster(instance_vec, centroids):
    """Devuelve el cluster más cercano a la instancia."""
    min_dist = float("inf")
    assigned_cluster = None
    for c_label, c_vec in centroids.items():
        dist = np.linalg.norm(instance_vec - c_vec)
        if dist < min_dist:
            min_dist = dist
            assigned_cluster = c_label
    return assigned_cluster, min_dist


def find_nearest_instances(instance_vec, cluster_points, cluster_ids, top_k=5):
    """Devuelve las instancias más cercanas dentro del mismo cluster."""
    dists = np.linalg.norm(cluster_points - instance_vec, axis=1)
    idx_sorted = np.argsort(dists)
    nearest_ids = cluster_ids[idx_sorted[:top_k]]
    nearest_dists = dists[idx_sorted[:top_k]]
    return list(zip(nearest_ids, nearest_dists))



def main():
    config = {
        "instances_path": "instances.csv",      # embeddings de entrenamiento
        "centroids_path": "centroids.csv",      # centroides del modelo
        "new_instances_path": "instances_embeddings.csv",       # nuevas instancias (dataset de test)
        "id_col": nullcontext,
        "cluster_col": "cluster",
        "top_k": 5,
        "output_path": "classified.csv"
    }

    # Cargar datos
    ids_train, clusters_train, emb_train = load_embeddings(
        config["instances_path"], id_col=config["id_col"], cluster_col=config["cluster_col"]
    )
    centroids = load_centroids(config["centroids_path"], cluster_col=config["cluster_col"])
    ids_new, _, emb_new = load_embeddings(config["new_instances_path"], id_col=config["id_col"])

    print(f"\nClasificando {len(ids_new)} nuevas instancias...")

    results = []
    for i, vec in enumerate(emb_new):
        new_id = ids_new[i]
        assigned_cluster, dist = assign_cluster(vec, centroids)

        # Buscar vecinos del mismo cluster
        mask = clusters_train == assigned_cluster
        nearest = find_nearest_instances(vec, emb_train[mask], ids_train[mask], top_k=config["top_k"])

        nearest_ids_str = ";".join(str(nid) for nid, _ in nearest)
        nearest_dists_str = ";".join(f"{d:.4f}" for _, d in nearest)

        results.append({
            "id": new_id,
            "assigned_cluster": assigned_cluster,
            "distance_to_centroid": round(dist, 4),
            "nearest_instance_ids": nearest_ids_str,
            "nearest_instance_dists": nearest_dists_str
        })

    # Guardar resultados en CSV
    df_out = pd.DataFrame(results)
    df_out.to_csv(config["output_path"], index=False)
    print(f"\n Clasificación completada. Resultados guardados en '{config['output_path']}'.")


if __name__ == "__main__":
    main()
