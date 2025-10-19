import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score, precision_recall_fscore_support
from scipy.optimize import linear_sum_assignment

# --- Módulo 1: Merge ---
def merge_labels_and_clusters(path_labels, path_clusters,
                              id_col='newid',
                              true_col='gs_text34',
                              cluster_col='cluster'):
    df_labels = pd.read_csv(path_labels)
    df_clusters = pd.read_csv(path_clusters)
    
    for col in [id_col, true_col]:
        if col not in df_labels.columns:
            raise ValueError(f"Columna '{col}' no encontrada en {path_labels}")
    if cluster_col not in df_clusters.columns:
        raise ValueError(f"Columna '{cluster_col}' no encontrada en {path_clusters}")
    
    merged_df = pd.merge(df_labels[[id_col, true_col]],
                         df_clusters[[id_col, cluster_col]],
                         on=id_col, how='inner')
    
    if merged_df.empty:
        raise ValueError("El merge resultó vacío. Verifica que los IDs coincidan en ambos CSVs.")
    
    true_labels = merged_df[true_col].values
    cluster_labels = merged_df[cluster_col].values
    
    return true_labels, cluster_labels, merged_df

# --- Módulo 2: Evaluación externa ---
def contingency_table(true_labels, cluster_labels):
    labels_true = np.unique(true_labels)
    labels_pred = np.unique(cluster_labels)
    true_index = {lab:i for i, lab in enumerate(labels_true)}
    pred_index = {lab:i for i, lab in enumerate(labels_pred)}
    cont = np.zeros((len(labels_true), len(labels_pred)), dtype=int)
    for t, p in zip(true_labels, cluster_labels):
        cont[true_index[t], pred_index[p]] += 1
    return pd.DataFrame(cont, index=labels_true, columns=labels_pred)

def purity_score(true_labels, cluster_labels):
    cont = contingency_table(true_labels, cluster_labels)
    return cont.max(axis=0).sum() / cont.values.sum()

def hungarian_accuracy(true_labels, cluster_labels):
    cont = contingency_table(true_labels, cluster_labels).values
    row_ind, col_ind = linear_sum_assignment(-cont)
    matched = cont[row_ind, col_ind].sum()
    return matched / cont.sum(), row_ind, col_ind

def external_evaluation(true_labels, cluster_labels):
    true = np.asarray(true_labels)
    pred = np.asarray(cluster_labels)
    
    results = {}
    results['ARI'] = adjusted_rand_score(true, pred)
    results['NMI'] = normalized_mutual_info_score(true, pred)
    results['FMI'] = fowlkes_mallows_score(true, pred)
    results['Purity'] = purity_score(true, pred)
    
    acc, row_ind, col_ind = hungarian_accuracy(true, pred)
    labels_true = np.unique(true)
    labels_pred = np.unique(pred)
    hungarian_map = {labels_pred[c]: labels_true[r] for r, c in zip(row_ind, col_ind)}
    
    cont_df = contingency_table(true, pred)
    full_map = {cluster: hungarian_map.get(cluster, cont_df[cluster].idxmax()) for cluster in cont_df.columns}
    
    results['Hungarian_accuracy'] = acc
    results['mapping_cluster_to_true'] = full_map
    
    mapped_pred = np.array([full_map[p] for p in pred])
    
    labels = np.unique(true)
    p, r, f, sup = precision_recall_fscore_support(true, mapped_pred, labels=labels, zero_division=0)
    prf_df = pd.DataFrame({'precision': p, 'recall': r, 'f1': f, 'support': sup}, index=labels)
    results['per_class_prf'] = prf_df
    
    cont = contingency_table(true, pred)
    results['contingency_table'] = cont
    
    return results

# --- Módulo 3: Visualización ---
def plot_contingency_heatmap(contingency_df, figsize=(10,6), cmap="Blues"):
    plt.figure(figsize=figsize)
    sns.heatmap(contingency_df, annot=True, fmt="d", cmap=cmap)
    plt.title("Tabla de contingencia: Clusters vs Etiquetas reales")
    plt.ylabel("Etiqueta real")
    plt.xlabel("Cluster")
    plt.show()

def plot_prf_metrics(per_class_prf_df, figsize=(10,6)):
    metrics = per_class_prf_df[['precision', 'recall', 'f1']].copy()
    metrics.plot(kind='bar', figsize=figsize)
    plt.title("Métricas por clase (Precision, Recall, F1)")
    plt.ylabel("Valor")
    plt.xlabel("Clase")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# --- Main ---
def main(path_labels, path_clusters,
         id_col='newid', true_col='gs_text34', cluster_col='cluster'):
    true_labels, cluster_labels, merged_df = merge_labels_and_clusters(
        path_labels, path_clusters, id_col, true_col, cluster_col
    )
    print(f"Se han alineado {len(true_labels)} instancias.")
    display(merged_df.head())
    
    res = external_evaluation(true_labels, cluster_labels)
    
    print("\n--- Métricas de evaluación externa ---")
    print(f"ARI: {res['ARI']:.4f}")
    print(f"NMI: {res['NMI']:.4f}")
    print(f"Fowlkes-Mallows (FMI): {res['FMI']:.4f}")
    print(f"Purity: {res['Purity']:.4f}")
    print(f"Hungarian best-match accuracy: {res['Hungarian_accuracy']:.4f}")
    
    print("\nMapping cluster -> etiqueta real:")
    for c, t in res['mapping_cluster_to_true'].items():
        print(f"  Cluster {c} -> {t}")
    
    print("\nTabla de contingencia (primeras filas):")
    display(res['contingency_table'].head())
    
    # Visualizaciones
    plot_contingency_heatmap(res['contingency_table'])
    plot_prf_metrics(res['per_class_prf'])
    
    return res

if __name__ == "__main__":
    main(
        path_labels='CSV_Original.csv',
        path_clusters='clusters_prueba.csv',
        id_col='newid',
        true_col='gs_text34',
        cluster_col='cluster'
    )
