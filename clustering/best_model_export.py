import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from clustering.embeddings import *
from clustering.hierarchical import *
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import joblib
import os
import traceback

def train_and_export_best_model(metrics_path="./output/metrics.csv",
                                embeddings_path="./dataset/cleaned_PHMRC_VAI_redacted_free_text.train.csv"):
    """
    Reconstruye el mejor modelo a partir de los resultados de métricas y lo exporta.
    Solo realiza una iteración con los mejores parámetros encontrados.
    """
    try:
        # Cargar resultados y calcular GlobalScore 
        df = pd.read_csv(metrics_path)

        # Invertir cohesion (cuanto más baja mejor)
        df["cohesion_inv"] = -df["cohesion"]
        internal_metrics = ["cohesion_inv", "silhouette_global"]

        # Escalar para que todas las métricas internas estén entre 0 y 1, y calcular media
        scaler = MinMaxScaler()
        df[internal_metrics] = scaler.fit_transform(df[internal_metrics])
        df["InternalScore"] = df[internal_metrics].mean(axis=1)

        # Calcular métricas externas (train + test), mayor peso a test
        external_train = ["ARI_Train", "NMI_Train", "FMI_Train", "Purity_Train", "Hungarian_accuracy_Train"]
        external_test  = ["ARI_test", "NMI_test", "FMI_test", "Purity_test", "Hungarian_accuracy_test"]
        df["ExternalScore"] = 0.4 * df[external_train].mean(axis=1) + 0.6 * df[external_test].mean(axis=1)

        # Score final combinado, más peso a la externa que a la interna
        w_int, w_ext = 0.3, 0.7
        df["GlobalScore"] = w_int * df["InternalScore"] + w_ext * df["ExternalScore"]

        # Mejor combinación
        best_row = df.loc[df["GlobalScore"].idxmax()]
        print("\nMejor combinación encontrada:")
        print(best_row[["linkage", "metric", "p", "pca", "poda", "GlobalScore"]])


    except Exception as e:
        print("[ERROR] Falló la reconstrucción del mejor modelo.")
        print("→", e)
        traceback.print_exc()
        return None
