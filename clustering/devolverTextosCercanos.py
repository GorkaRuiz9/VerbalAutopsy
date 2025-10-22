import pandas as pd
import numpy as np
import joblib
import os
from embeddings import embeddings
from sklearn.metrics.pairwise import euclidean_distances

# --- Configuración ---
csv_embeddings_file = "../output/asignacion_single_euclidean_p1_pca50_poda4.csv"
csv_original_file = "CSV_Original.csv"  # ya estás dentro de clustering/
texto_col = "open_response"
label_col = "gs_text34"
pca_model_path = "../output/pca_model.pkl"
num_vecinos = 2
output_txt_file = "../output/vecinos_encontrados.txt"

# --- Cargar embeddings del CSV de clusters ---
df = pd.read_csv(csv_embeddings_file)
ids = df["newid"].values
clusters = df["cluster"].values
embeddings_array = df.drop(columns=["newid", "cluster"]).values  # embeddings reducidos

# --- Cargar PCA si existe ---
pca_model = None
if os.path.exists(pca_model_path):
    pca_model = joblib.load(pca_model_path)

# --- Función para encontrar vecinos ---
def buscar_vecinos(texto_input, embeddings_array, ids, clusters, csv_original_file, pca_model=None):
    # 1. Crear CSV temporal con el texto de entrada
    df_temp = pd.DataFrame({
        "newid": [0],
        "open_response": [texto_input],
        "gs_text34": [""]  # label vacía
    })
    temp_csv = "./output/temp_input.csv"
    os.makedirs("./output", exist_ok=True)
    df_temp.to_csv(temp_csv, index=False)

    # 2. Generar embedding usando la función existente
    df_embedding_input = embeddings(temp_csv)

    # 3. Aplicar PCA si corresponde
    emb_input = df_embedding_input.drop(columns=["id", "gs_text34"]).values
    if pca_model is not None:
        emb_input = pca_model.transform(emb_input)

    # 4. Calcular distancias
    dists = euclidean_distances(emb_input, embeddings_array)[0]

    # 5. Obtener índices de los n vecinos más cercanos
    nearest_idx = np.argsort(dists)[:num_vecinos]

    # 6. Cargar CSV original para obtener textos y etiquetas reales
    df_original = pd.read_csv(csv_original_file)

    vecinos = []
    for idx in nearest_idx:
        vecino_id = ids[idx]
        cluster_num = clusters[idx]

        fila = df_original.loc[df_original["newid"] == vecino_id]
        if not fila.empty:
            texto_vecino = fila[texto_col].values[0]
            etiqueta_real = fila[label_col].values[0]
        else:
            texto_vecino = "(texto no encontrado)"
            etiqueta_real = "(sin etiqueta)"

        vecinos.append({
            "id": vecino_id,
            "cluster": cluster_num,
            "distancia": dists[idx],
            "texto": texto_vecino,
            "etiqueta_real": etiqueta_real
        })

    # 7. Borrar CSV temporal
    os.remove(temp_csv)

    # 8. Guardar vecinos en archivo txt
    with open(output_txt_file, "w", encoding="utf-8") as f:
        f.write(f"Texto de entrada:\n{texto_input}\n\nVecinos más cercanos:\n")
        for v in vecinos:
            f.write(f"ID: {v['id']}\n")
            f.write(f"Cluster: {v['cluster']}\n")
            f.write(f"Etiqueta real: {v['etiqueta_real']}\n")
            f.write(f"Distancia: {v['distancia']:.4f}\n")
            f.write(f"Texto: {v['texto']}\n\n")

    return vecinos

# --- Ejemplo de uso ---
texto = "He felt a strong pain in the chest and had difficulty breathing. Blood pressure was high and pulse was rapid."
vecinos = buscar_vecinos(texto, embeddings_array, ids, clusters, csv_original_file, pca_model)

print("\nVecinos más cercanos encontrados:\n")
for v in vecinos:
    print(f"ID: {v['id']} | Cluster: {v['cluster']} | Distancia: {v['distancia']:.4f}")
    print(f"Etiqueta real: {v['etiqueta_real']}")
    print(f"Texto: {v['texto']}\n")
