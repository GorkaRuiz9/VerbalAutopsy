from clustering.embeddings import *
from clustering.hierarchical import *
import pandas as pd



#embedding = embeddings("./dataset/cleaned_PHMRC_VAI_redacted_free_text.train.csv")
#print(embedding)
df = pd.read_csv("./output/cleaned_PHMRC_VAI_redacted_free_text.train_embeddings.csv")
cluster = AgglomerativeClustering()
cluster.fit(df)
clusters_result = cluster.cut_tree(500)
print(clusters_result)
cluster.view_dendrogram()
