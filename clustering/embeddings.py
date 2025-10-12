import numpy as np
import pandas as pd
import re
import csv
import os

# Parámetros
embedding_dim = 50
window_size = 2
learning_rate = 0.01



def limpiar_texto(t):
    t = t.lower()
    t = re.sub(r'[^a-z\s]', ' ', t)
    return t.split()


def generar_embeddings(csv_path, text_col="gs_text34"):

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Archivo no encontrado: {csv_path}")

    # Leer dataset
    df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        raise ValueError(f"Columna '{text_col}' no encontrada en CSV.")

  
    texts = df[text_col].dropna().astype(str).tolist()
    tokenized_docs = [limpiar_texto(t) for t in texts]

    # Crear vocabulario
    vocabulario = sorted(set([word for doc in tokenized_docs for word in doc]))
    # Inicializar vetcores aleatorios
    embeddings = {word: np.random.uniform(-0.5, 0.5, embedding_dim) for word in vocabulario}

    # Entrenamiento simple basado en coocurrencias
    for doc in tokenized_docs:
        for i, word in enumerate(doc):
            context = doc[max(0, i - window_size): i] + doc[i + 1: i + 1 + window_size]
            for c in context:
                embeddings[word] += learning_rate * (embeddings[c] - embeddings[word])

    return embeddings


def cosine_sim(v1, v2):
    """Similitud coseno entre dos vectores."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def cargar_embeddings_csv(csv_path):
    """Carga embeddings guardados en CSV y devuelve diccionario {palabra: vector}."""
    embeddings = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            word = row[0]
            vector = [float(x) for x in row[1:]]
            embeddings[word] = vector
    return embeddings


# Código de prueba
if __name__ == "__main__":
    csv_path = "cleaned_PHMRC_VAI_redacted_free_text.train.csv"
    embeddings = generar_embeddings(csv_path)

    # Guardar en CSV
    output_csv = "embeddings.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word"] + [f"dim_{i}" for i in range(50)])
        for word, vec in embeddings.items():
            writer.writerow([word] + vec.tolist())
          

    w1, w2 = "pneumonia", "fever"
    if w1 in embeddings and w2 in embeddings:
        sim = cosine_sim(embeddings[w1], embeddings[w2])
        print(f"Similitud entre '{w1}' y '{w2}': {sim:.4f}")
    print(f"Embedding de '{w1}': {embeddings[w1]}")
