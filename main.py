from clustering.embeddings import *
from clustering.hierarchical import *
import pandas as pd
from clustering.Evaluacion import *

SEED = 42
SUB_SET = 1500

#embedding = embeddings("./dataset/cleaned_PHMRC_VAI_redacted_free_text.train.csv")
#print(embedding)

df = pd.read_csv("./output/cleaned_PHMRC_VAI_redacted_free_text.train_embeddings.csv")
df_reduced = df.sample(n=SUB_SET, random_state=SEED)
cluster = AgglomerativeClustering()
cluster.fit(df_reduced)
cluster.view_dendrogram()
clusters_result = cluster.cut_tree(500)

path = f"./output/asignacion_{cluster.linkage}_{cluster.metric}_{cluster.p}_{SUB_SET}.csv"
clusters_result.to_csv(path, index=False)
    
metrics = get_metrics(path, cluster.metric, cluster.p, mode="mean")
metrics["linkage"] = cluster.linkage
metrics["metric"] = cluster.metric
metrics["p"] = cluster.p


output_file = "./output/metrics.csv"
file_exists = os.path.isfile(output_file)

pd.DataFrame([metrics]).to_csv(output_file, index=False, mode="a", header= not file_exists)
