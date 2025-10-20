import pandas as pd
from clustering.utils import *
import copy
import clustering.distances
import json
from pathlib import Path

class AgglomerativeClustering:
    """
    Implementación personalizada de un algoritmo de clustering jerárquico aglomerativo.
    """
    
    def __init__(self, linkage="average", metric="euclidean", minkowski_p=3, path=None):
        
        """
        Inicializa el modelo de clustering jerárquico.

        Parámetros:
        -----------
        linkage : str
            Método de enlace usado para calcular la distancia entre clusters ('single', 'complete', 'average', 'mean').
        metric : str
            Métrica de distancia utilizada ('euclidean', 'manhattan', 'minkowski', 'sentence').
        minkowski_p : int
            Parámetro p del cálculo de la distancia de Minkowski (solo relevante si metric="minkowski").
        path : str | None
            Ruta a un archivo JSON que contenga un modelo previamente exportado.
        """
        
        self.linkage = linkage
        self.metric = metric
        self.p = minkowski_p
        # contiene la lista de cluster candidatos a funsionarse
        self.clusters = list()
        # contiene todo los historial de clusters creados
        self.clusters_history = list()
        self.data_set = None
        self.labels_ = None
        self.centroides = None
        
        if path:
            self.load_model(path)


    def fit(self, data_set: pd.DataFrame):
        
        """
        Construye el dendrograma a partir del conjunto de datos y ejecuta el proceso aglomerativo completo.

        Parámetros:
        -----------
        data_set : pd.DataFrame
            Dataset de entrada donde cada fila representa una instancia y cada columna un atributo.
        """
        
        self.data_set = data_set
        self.clusters = [ClusterNode(datos=[dato], id=i) for i, dato in enumerate(data_set.values)]
        self.clusters_history = copy.deepcopy(self.clusters)
        
        while len(self.clusters) > 1:
            distancias = calcular_distancias(self.clusters, self.linkage, self.metric, self.p)
            min_dist, (i, j) = get_min_dist(distancias)
            clusterA = self.clusters[i]
            clusterB = self.clusters[j]
            self.update_clusters_list(clusterA, clusterB, min_dist)
            
    
    def predict(self, data_set: pd.DataFrame):
        
        """
        Asigna nuevas instancias a los clusters existentes según los centroides aprendidos.

        Parámetros:
        -----------
        data_set : pd.DataFrame
            Nuevas instancias a clasificar.

        Retorna:
        --------
        pd.DataFrame
            Copia del dataset con una nueva columna "cluster" que indica el identificador del cluster asignado.
        """
        
        centroides_items = list(self.centroides.items())
        
        asignaciones = []
        for i, instancia in data_set.iterrows():
            x = np.array(instancia.values, dtype=float)
            
            distancias = [clustering.distances.heuclidean_distance(x, centroide) for _, centroide in centroides_items]
            
            id_cluster = centroides_items[int(np.argmin(distancias))][0]
            asignaciones.append(id_cluster)

        resultado = data_set.copy()
        resultado["cluster"] = asignaciones
        
        return resultado
        
        
    def set_centroids(self, clusters):
        
        """
        Calcula los centroides de una lista de clusters.

        Parámetros:
        -----------
        clusters : list[ClusterNode]
            Lista de objetos ClusterNode a partir de los cuales se calcularán los centroides.

        Retorna:
        --------
        dict
            Diccionario {id_cluster: vector_centroide}.
        """
        
        centroides = {}
        for cluster in clusters:
            datos = np.array(cluster.datos)
            centroide = np.mean(datos, axis=0)
            centroides[cluster.id] = centroide
        return centroides
       
            
    def update_clusters_list(self, clusterA, clusterB, min_dist):
        
        """
        Fusiona dos clusters y actualiza las listas internas del modelo.

        Parámetros:
        -----------
        clusterA : ClusterNode
            Primer cluster a fusionar.
        clusterB : ClusterNode
            Segundo cluster a fusionar.
        min_dist : float
            Distancia entre ambos clusters.
        """
        
        new_cluster = ClusterNode(clusterA, clusterB, min_dist, clusterA.datos + clusterB.datos, len(self.clusters_history))
        self.clusters.remove(clusterA)
        self.clusters.remove(clusterB)
        self.clusters.append(new_cluster)
        self.clusters_history.append(new_cluster)
        
        
    def cut_tree(self, dist_to_cut):
        
        """
        Realiza una poda del dendrograma a una distancia de corte específica.

        Parámetros:
        -----------
        dist_to_cut : float
            Distancia umbral a partir de la cual se detiene la unión de clusters.

        Retorna:
        --------
        pd.DataFrame
            DataFrame con las instancias originales y su cluster asignado.
        """
        
        if not self.clusters_history:
            return []
        top = self.clusters_history[-1]
        clusters = self.get_clusters(top, dist_to_cut)
        self.centroides = self.set_centroids(clusters)
        return get_results_df(clusters, self.data_set)
    
    
    def get_clusters(self, cluster, dist_to_cut):
        
        """
        Recorre recursivamente el árbol de clusters y devuelve los clusters finales 
        tras aplicar la distancia de corte.

        Parámetros:
        -----------
        cluster : ClusterNode
            Nodo raíz desde el cual comenzar la búsqueda.
        dist_to_cut : float
            Distancia umbral para detener la expansión.

        Retorna:
        --------
        list[ClusterNode]
            Lista de clusters obtenidos tras la poda.
        """
        
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
        
        """
        Dibuja el dendrograma del clustering a partir del historial de fusiones.
        """
        
        plt_dendrogram(self.clusters_history)
        
    
    def export(self):
        
        """
        Exporta el modelo entrenado (configuración y centroides) a un archivo JSON.
        """
        
        if not self.centroides:
            raise Exception("Ejecuta primero fit(dataset) antes de exportar el modelo")
        
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
        
        """
        Carga un modelo previamente exportado desde un archivo JSON.

        Parámetros:
        -----------
        path : str
            Ruta al archivo JSON que contiene la configuración y los centroides del modelo.
        """
        
        with open(path, "r") as f:
            conf = json.load(f)
            
        self.linkage = conf["linkage"]
        self.metric = conf["metric"]
        self.p = conf["p"]
        self.centroides = conf["centroides"]

class ClusterNode:
    
    """
    Representa un nodo del árbol de clustering jerárquico.
    """
    
    def __init__(self, left=None, right=None, distance=0.0, datos=None, id=None):
        
        """
        Inicializa un nodo del dendrograma.

        Parámetros:
        -----------
        left : ClusterNode | None
            Hijo izquierdo (cluster fusionado).
        right : ClusterNode | None
            Hijo derecho (cluster fusionado).
        distance : float
            Distancia a la que se fusionaron los dos clusters hijos.
        datos : list
            Lista de vectores (instancias) contenidos en este cluster.
        id : int
            Identificador único del cluster.
        """
        
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
    print("###Resultados###")
    print(clusters_result)
    print("###Centroides###")
    print(clustering.centroides)
    
    print("###Dendograma###")
    clustering.view_dendrogram()    

    df_test = pd.DataFrame(data_test, columns=["atrib1", "atrib2"])
    print("###Dataset de test###")
    print(df_test)
    
    results = clustering.predict(df_test)
    print(results)
    clustering.export()
    
    AgglomerativeClustering(path=Path("./output/model.json").resolve())
    exit()
    print("=========Fin Pruebas=========")
    