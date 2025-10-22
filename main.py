from clustering.embeddings import *
from clustering.hierarchical import *
import pandas as pd
from clustering.Evaluacion import *
from itertools import product

# para la reproducibilidad
SEED = 42
# tamaño del set que se va a procesar
SUB_SET = 1000

embeddings_file = "./output/cleaned_PHMRC_VAI_redacted_free_text.train_embeddings.csv"
file_exists = os.path.isfile(embeddings_file)
if not file_exists:
    embeddings("./dataset/cleaned_PHMRC_VAI_redacted_free_text.train.csv")

df = pd.read_csv(embeddings_file)
df_reduced = df.sample(n=SUB_SET, random_state=SEED)

# definimos el dominio de los experimentos
metrics_list = ['euclidean']
linkage_list = ['single']
p_list = [1]
pca_list = [50]
poda_list = [6]

output_file = "./output/metrics.csv"
file_exists = os.path.isfile(output_file)
        
for metric, linkage, p, n_pca, poda in product(metrics_list, linkage_list, p_list, pca_list, poda_list):
    print(f"\n[INFO] Ejecutando experimento con linkage={linkage}, metric={metric}, p={p}, pca={n_pca}, poda={poda}")

    try:
        
        cluster = AgglomerativeClustering(
            linkage=linkage,
            metric=metric,
            minkowski_p=p
        )

        cluster.fit(df_reduced)
        cluster.view_dendrogram(show=False, n=SUB_SET)
        clusters_result = cluster.cut_tree(poda)

        asign_path = f"./output/asignacion_{linkage}_{metric}_p{p}_pca{n_pca}_poda{poda}.csv"
        clusters_result.to_csv(asign_path, index=False)

        metrics = get_metrics(asign_path, metric, p, mode="mean")
        metrics["linkage"] = linkage
        metrics["metric"] = metric
        metrics["p"] = p
        metrics["pca"] = n_pca
        metrics["poda"] = poda

        pd.DataFrame([metrics]).to_csv(output_file, index=False, mode="a", header=not file_exists)
        file_exists = True

    except Exception as e:
        print(f"[ERROR] Falló el experimento con linkage={linkage}, metric={metric}, p={p}, pca={n_pca}, poda={poda}")
        print("         →", e)
        continue

print("\n✅ Todos los experimentos completados.")
