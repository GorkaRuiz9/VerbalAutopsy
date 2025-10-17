import pandas as pd
from utils import *
import copy
import distances
import json
from pathlib import Path

class AgglomerativeClustering:
    
    def __init__(self, linkage="average", metric="euclidean", minkowski_p=3, path=None):
        
        self.linkage = linkage
        self.metric = metric
        self.p = minkowski_p
        self.clusters = list()
        self.clusters_history = list()
        self.data_set = None
        self.labels_ = None
        self.centroides = None
        
        if path:
            self.load_model(path)


    def fit(self, data_set: pd.DataFrame):
        
        self.data_set = data_set
        self.clusters = [ClusterNode(datos=[dato], id=i) for i, dato in enumerate(data_set.values)]
        self.clusters_history = copy.deepcopy(self.clusters)
        self.set_centroids(self.clusters)
        
        while len(self.clusters) > 1:
            distancias = calcular_distancias(self.clusters, self.linkage, self.metric, self.p)
            min_dist, (i, j) = get_min_dist(distancias)
            clusterA = self.clusters[i]
            clusterB = self.clusters[j]
            self.update_clusters_list(clusterA, clusterB, min_dist)
            
    
    def predict(self, data_set: pd.DataFrame):
        
        centroides_items = list(self.centroides.items())
        
        asignaciones = []
        for i, instancia in data_set.iterrows():
            x = np.array(instancia.values, dtype=float)
            
            distancias = [distances.heuclidean_distance(x, centroide) for _, centroide in centroides_items]
            
            id_cluster = centroides_items[int(np.argmin(distancias))][0]
            asignaciones.append(id_cluster)

        resultado = data_set.copy()
        resultado["cluster"] = asignaciones
        
        return resultado
        
        
    def set_centroids(self, clusters):
        
        centroides = {}
        for cluster in clusters:
            datos = np.array(cluster.datos)
            centroide = np.mean(datos, axis=0)
            centroides[cluster.id] = centroide
        return centroides
            
    def update_clusters_list(self, clusterA, clusterB, min_dist):
        
        new_cluster = ClusterNode(clusterA, clusterB, min_dist, clusterA.datos + clusterB.datos, len(self.clusters_history))
        self.clusters.remove(clusterA)
        self.clusters.remove(clusterB)
        self.clusters.append(new_cluster)
        self.clusters_history.append(new_cluster)
        
        
    def cut_tree(self, dist_to_cut):
        
        if not self.clusters_history:
            return []
        top = self.clusters_history[-1]
        clusters = self.get_clusters(top, dist_to_cut)
        self.centroides = self.set_centroids(clusters)
        return get_results_df(clusters, self.data_set)
    
    
    def get_clusters(self, cluster, dist_to_cut):
        
        if cluster.left == None and cluster.right == None:
            return [cluster]
        if cluster.distance <= dist_to_cut:
            return [cluster]
        else:
            clusters = list()
            clusters += self.get_clusters(cluster.left, dist_to_cut)
            clusters += self.get_clusters(cluster.right, dist_to_cut)
            return clusters
        
    def view_dendrogram(self):
        plt_dendrogram(self.clusters_history)
        
    
    def export(self):
        
        if not self.centroides:
            raise Exception("Ejecuta primero fit(dataset) antes de exportar el modelo")
        
        print(self.centroides)
        
        c = {k: v.tolist() for k, v in self.centroides.items()}
        
        data_to_export = {
            "linkage": self.linkage,
            "metric": self.metric,
            "p": self.p,
            "centroides": c
        }

        with open("./output/model.json", "w") as f:
            json.dump(data_to_export, f, indent=4)
            
    
    def load_model(self, path):
        
        with open(path, "r") as f:
            conf = json.load(f)
            
        self.linkage = conf["linkage"]
        self.metric = conf["metric"]
        self.p = conf["p"]
        self.centroides = conf["centroides"]

class ClusterNode:
    def __init__(self, left=None, right=None, distance=0.0, datos=None, id=None):
        self.left = left          
        self.right = right        
        self.distance = distance  
        self.datos = datos    
        self.id = id
        
    def __str__(self):
        return f"Cluster: {self.id} [{self.datos}]"
        


#=====Pruebas=====#
if __name__ == "__main__":
    
    data = [
    [1.0, 2.0],
    [1.5, 1.8],
    [30.0, 50.0],
    [11.0, 11.0],
    [1.0, 0.6],
    [9.0, 11.0],
    [8.0, 19.0],
    [10.0, 20.0],
    [45.0, 35.0],
    [0.0, 1.0]
    ]
    
    data_test = [
    [2.0, 1.0],
    [2.0, 2.0],
    [35.0, 50.0],
    [10.0, 10.0]
    ]
    
    df = pd.DataFrame(data, columns=["atrib1", "atrib2"])

    print("###Dataset de train###")
    print(df)
    
    clustering = AgglomerativeClustering(linkage="average", metric="euclidean")
    clustering.fit(df)

    # Por ejemplo, cortar a distancia 5.0
    clusters_result = clustering.cut_tree(dist_to_cut=5.0)
    print("###Centroides###")
    print(clustering.centroides)
    
    print("###Dendograma###")
    print(clustering.view_dendrogram())    

    df_test = pd.DataFrame(data_test, columns=["atrib1", "atrib2"])
    print("###Dataset de test###")
    print(df_test)
    
    results = clustering.predict(df_test)
    print(results)
    clustering.export()
    
    AgglomerativeClustering(path=Path("./output/model.json").resolve())
    exit()
    print("=========Fin Pruebas=========")
    