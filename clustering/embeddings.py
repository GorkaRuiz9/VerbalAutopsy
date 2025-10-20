# Genera embeddings médicos a partir de notas clínicas en inglés
# usando el modelo transformer "emilyalsentzer/Bio_ClinicalBERT"
from sentence_transformers import SentenceTransformer
import pandas as pd
import os



MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"  # Modelo médico
EMBEDDING_DIM = 768  # Dimensión típica del modelo BERT-base


def generar_embeddings_biomedico(model_name, textos):
    #Genera embeddings de una lista de textos médicos con Bio_ClinicalBERT.
    #Devuelve una matriz de vectores.

    print(f"\nCargando modelo: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Generando embeddings para {len(textos)} instancias...")
    embeddings = model.encode(textos, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def cargar_dataset(csv_path, text_col, id_col):
    """ Carga el dataset y devuelve listas con los textos y sus IDs."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Archivo no encontrado: {csv_path}")

    df = pd.read_csv(csv_path)

    if text_col not in df.columns or id_col not in df.columns:
        raise ValueError(f"El CSV debe contener las columnas '{id_col}' y '{text_col}'.")

    textos = df[text_col].astype(str).fillna("").tolist()
    ids = df[id_col].tolist()
    return ids, textos


#Pruebas
def main():
    # Parámetros
    CSV_INPUT = "cleaned_PHMRC_VAI_redacted_free_text.train.csv"  # Dataset original
    TEXT_COL = "gs_text34"
    ID_COL = "newid"
    OUTPUT_CSV = "instances_embeddings_Bio_ClinicalBERT.csv"

    # 1. Cargar datos
    ids, textos = cargar_dataset(CSV_INPUT, TEXT_COL, ID_COL)

    # 2. Generar embeddings
    embeddings = generar_embeddings_biomedico(MODEL_NAME, textos)

    # 3. Crear DataFrame con resultados
    df_out = pd.DataFrame(embeddings)
    df_out.insert(0, "id", ids)

    # 4. Guardar CSV
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nEmbeddings guardados correctamente en '{OUTPUT_CSV}'")
    
    return df_out

if __name__ == "__main__":
    main()
