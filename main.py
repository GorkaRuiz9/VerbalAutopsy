from clustering.embeddings import *
from clustering.hierarchical import *
import pandas as pd



#embeddings = main()
#print(embeddings)
df = pd.read_csv("./clustering/instances_embeddings_Bio_ClinicalBERT.csv")

cluster = AgglomerativeClustering()
cluster.fit(df)
clusters_result = cluster.cut_tree(500)
print(clusters_result)
cluster.view_dendrogram()
