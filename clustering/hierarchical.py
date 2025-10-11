# implementacion de la clase AgglomerativeClustering
import pandas

class AgglomerativeClustering:
    def __init__(self, linkage="average", metric="euclidean", minkowski_p=3):
        self.linkage = linkage
        self.metric = metric
        self.p = minkowski_p
        self.linkage_matrix_ = []
        self.labels_ = None

    def fit(self, data_set: pandas.DataFrame):
        # contruye el dendograma a partir del dataset
        while len(self.linkage_matrix_) < len(data_set):
            
            
    
    def get_clusters(self, dist_to_cut):
        
        return None
