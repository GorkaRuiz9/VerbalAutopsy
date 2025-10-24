import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from clustering.embeddings import *
from clustering.hierarchical import *
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import joblib
import os
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_export_best_model(metrics_path="./output/metrics.csv",
                                embeddings_path="./dataset/cleaned_PHMRC_VAI_redacted_free_text.train.csv"):
    """
    Reconstruye el mejor modelo a partir de los resultados de métricas y genera gráficos de GlobalScore vs cada hiperparámetro.
    """
    try:
        # Crear carpeta de salida de gráficos
        os.makedirs("./output/plots", exist_ok=True)

        
        #  CARGA Y CÁLCULO DE SCORES
        df = pd.read_csv(metrics_path)
        df["cohesion_inv"] = -df["cohesion"]
        internal_metrics = ["cohesion_inv", "silhouette_global"]

        scaler = MinMaxScaler()
        df[internal_metrics] = scaler.fit_transform(df[internal_metrics])
        df["InternalScore"] = df[internal_metrics].mean(axis=1)

        external_train = ["ARI_Train", "NMI_Train", "FMI_Train", "Purity_Train", "Hungarian_accuracy_Train"]
        external_test  = ["ARI_test", "NMI_test", "FMI_test", "Purity_test", "Hungarian_accuracy_test"]
        df["ExternalScore"] = 0.4 * df[external_train].mean(axis=1) + 0.6 * df[external_test].mean(axis=1)

        w_int, w_ext = 0.3, 0.7
        df["GlobalScore"] = w_int * df["InternalScore"] + w_ext * df["ExternalScore"]

        best_row = df.loc[df["GlobalScore"].idxmax()]
        print("\nMejor combinación encontrada:")
        print(best_row[["linkage", "metric", "p", "pca", "poda", "GlobalScore"]])

        
        #  GRÁFICOS GlobalScore vs cada hiperparámetro
       
        hiperparametros = ["linkage", "metric", "pca", "poda"]

        external_train_cols = [c for c in df.columns if c.endswith("_Train")]
        external_test_cols  = [c for c in df.columns if c.endswith("_test")]

        for param in hiperparametros:
            # --- GlobalScore ---
            grouped_score = df.groupby(param)["GlobalScore"].mean().reset_index()
            plt.figure(figsize=(8,5))
            sns.barplot(data=grouped_score, x=param, y="GlobalScore", palette="viridis", legend=False)
            plt.title(f"GlobalScore promedio vs {param}")
            plt.ylabel("GlobalScore promedio")
            min_val = grouped_score["GlobalScore"].min()
            max_val = grouped_score["GlobalScore"].max()
            margin = (max_val - min_val) * 0.05
            plt.ylim(min_val - margin, max_val + margin)
            plt.tight_layout()
            plt.savefig(f"./output/plots/globalscore_vs_{param}.png", dpi=300)
            plt.close()

            # --- Cohesion ---
            grouped_cohesion = df.groupby(param)["cohesion"].mean().reset_index()
            plt.figure(figsize=(8,5))
            sns.barplot(data=grouped_cohesion, x=param, y="cohesion", palette="magma", legend=False)
            plt.title(f"Cohesion promedio vs {param}")
            plt.ylabel("Cohesion promedio")
            min_val = grouped_cohesion["cohesion"].min()
            max_val = grouped_cohesion["cohesion"].max()
            margin = (max_val - min_val) * 0.05
            plt.ylim(min_val - margin, max_val + margin)
            plt.tight_layout()
            plt.savefig(f"./output/plots/cohesion_vs_{param}.png", dpi=300)
            plt.close()

            # --- Separabilidad (mean_pairwise_distance) ---
            grouped_sep = df.groupby(param)["mean_pairwise_distance"].mean().reset_index()
            plt.figure(figsize=(8,5))
            sns.barplot(data=grouped_sep, x=param, y="mean_pairwise_distance", palette="plasma", legend=False)
            plt.title(f"Separabilidad promedio vs {param}")
            plt.ylabel("Separabilidad promedio")
            min_val = grouped_sep["mean_pairwise_distance"].min()
            max_val = grouped_sep["mean_pairwise_distance"].max()
            margin = (max_val - min_val) * 0.05
            plt.ylim(min_val - margin, max_val + margin)
            plt.tight_layout()
            plt.savefig(f"./output/plots/separabilidad_vs_{param}.png", dpi=300)
            plt.close()

            # --- Métricas externas Train ---
            grouped_ext_train = df.groupby(param)[external_train_cols].mean().mean(axis=1).reset_index()
            plt.figure(figsize=(8,5))
            sns.barplot(data=grouped_ext_train, x=param, y=0, palette="cool", legend=False)
            plt.title(f"Métricas externas promedio (Train) vs {param}")
            plt.ylabel("Promedio métricas externas (Train)")
            min_val = grouped_ext_train[0].min()
            max_val = grouped_ext_train[0].max()
            margin = (max_val - min_val) * 0.05
            plt.ylim(min_val - margin, max_val + margin)
            plt.tight_layout()
            plt.savefig(f"./output/plots/external_train_vs_{param}.png", dpi=300)
            plt.close()

            # --- Métricas externas Test ---
            grouped_ext_test = df.groupby(param)[external_test_cols].mean().mean(axis=1).reset_index()
            plt.figure(figsize=(8,5))
            sns.barplot(data=grouped_ext_test, x=param, y=0, palette="coolwarm", legend=False)
            plt.title(f"Métricas externas promedio (Test) vs {param}")
            plt.ylabel("Promedio métricas externas (Test)")
            min_val = grouped_ext_test[0].min()
            max_val = grouped_ext_test[0].max()
            margin = (max_val - min_val) * 0.05
            plt.ylim(min_val - margin, max_val + margin)
            plt.tight_layout()
            plt.savefig(f"./output/plots/external_test_vs_{param}.png", dpi=300)
            plt.close()



    except Exception as e:
        print("[ERROR] Falló la reconstrucción o visualización del mejor modelo.")
        print("→", e)
        traceback.print_exc()
        return None


if __name__ == "__main__":
    train_and_export_best_model()
