from clustering.embeddings import *
from clustering.hierarchical import *
import pandas as pd
from clustering.Evaluacion import *
from clustering.EvaluacionExterna import *
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import traceback
import joblib

# para la reproducibilidad
SEED = 42
# tamaño del set que se va a procesar
SUB_SET = 100

embeddings_file = "./output/cleaned_PHMRC_VAI_redacted_free_text.train_embeddings.csv"
file_exists = os.path.isfile(embeddings_file)
if not file_exists:
    embeddings("./dataset/cleaned_PHMRC_VAI_redacted_free_text.train.csv")

df = pd.read_csv(embeddings_file)
df_reduced = df.sample(n=SUB_SET, random_state=SEED)

x = df_reduced.drop("gs_text34", axis=1)
y = df_reduced["gs_text34"]
#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
#y_train["id"] = X_train["id"]
#y_test["id"] = X_test["id"]

X_train = x.reset_index(drop=True)
y_train = y.reset_index(drop=True)
y_train["id"] = X_train["id"]


# definimos el dominio de los experimentos
metrics_list = ['euclidean']
linkage_list = ['single']
p_list = [1]
pca_list = [50]
poda_list = [4]

output_file = "./output/metrics.csv"
file_exists = os.path.isfile(output_file)
        
for metric, linkage, p, n_pca, poda in product(metrics_list, linkage_list, p_list, pca_list, poda_list):
    print(f"\n[INFO] Ejecutando experimento con linkage={linkage}, metric={metric}, p={p}, pca={n_pca}, poda={poda}")

    try:

        #Separar ids y embeddings originales
        ids = X_train["id"].reset_index(drop=True)
        embeddings_df = X_train.drop(columns=["id"])  #DataFrame con sólo atributos

        #Aplicar PCA si procede
        if n_pca is not None and int(n_pca) > 0:
            print(f"[INFO] Aplicando PCA con {n_pca} componentes")
            pca = PCA(n_components=int(n_pca))
            X_pca_array = pca.fit_transform(embeddings_df.values) 

            joblib.dump(pca, f"./output/pca_model_pca{n_pca}.pkl")


            #Reconstruir DataFrame con ids + componentes principales
            comp_cols = [f"pc_{i}" for i in range(X_pca_array.shape[1])]
            X_reduced = pd.DataFrame(X_pca_array, columns=comp_cols)
            X_reduced.insert(0, "id", ids)
        else:
            #no aplicar PCA: conservar formato (id + embeddings)
            X_reduced = pd.concat([ids, embeddings_df.reset_index(drop=True)], axis=1)
        
        cluster = AgglomerativeClustering(
            linkage=linkage,
            metric=metric,
            minkowski_p=p
        )

        cluster.fit(X_reduced)
        cluster.view_dendrogram(show=False, n=len(X_reduced))
        clusters_result = cluster.cut_tree(poda)

        asign_path = f"./output/asignacion_{linkage}_{metric}_p{p}_pca{n_pca}_poda{poda}.csv"

        clusters_result = clusters_result.rename(columns={"id": "newid"})
        clusters_result.to_csv(asign_path, index=False)

        metrics = get_metrics(asign_path, metric, p, mode="mean")
        metrics["linkage"] = linkage
        metrics["metric"] = metric
        metrics["p"] = p
        metrics["pca"] = n_pca
        metrics["poda"] = poda
        # falta añadir metricas externas con y_train ["id", "label"]
        # clusters_result tiene formato ["id", "embeddings", "cluster"]
        # el id de y_train y de clusters_result es el mismo

        

        # path al CSV original con etiquetas reales
        original_labels_file = "clustering/CSV_Original.csv"

        # merge: obtener true_labels y cluster_labels
        true_labels, cluster_labels, merged_df = merge_labels_and_clusters(
            path_labels=original_labels_file,
            path_clusters=asign_path,
            id_col='newid',
            true_col='gs_text34',
            cluster_col='cluster'
)

        external_res = external_evaluation(true_labels, cluster_labels)

        # extraer métricas resumidas (sin los DataFrames internos)
        external_metrics = {
            "ARI": external_res["ARI"],
            "NMI": external_res["NMI"],
            "FMI": external_res["FMI"],
            "Purity": external_res["Purity"],
            "Hungarian_accuracy": external_res["Hungarian_accuracy"]
        }

        metrics.update(external_metrics)


        pd.DataFrame([metrics]).to_csv(output_file, index=False, mode="a", header=not file_exists)
        file_exists = True
        
        # falta añadir evaluación de test con x_test e y_test

    except Exception as e:
        print(f"[ERROR] Falló el experimento con linkage={linkage}, metric={metric}, p={p}, pca={n_pca}, poda={poda}")
        print("         →", e)
        traceback.print_exc()
        continue

print("\n✅ Todos los experimentos completados.")
