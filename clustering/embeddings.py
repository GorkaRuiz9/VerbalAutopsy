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
    """ Devuelve un diccionario {palabra: vector}."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Archivo no encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        raise ValueError(f"Columna '{text_col}' no encontrada en CSV.")

    texts = df[text_col].dropna().astype(str).tolist()
    tokenized_docs = [limpiar_texto(t) for t in texts]

    vocabulario = sorted(set([word for doc in tokenized_docs for word in doc]))
    embeddings = {word: np.random.uniform(-0.5, 0.5, embedding_dim) for word in vocabulario}

    for doc in tokenized_docs:
        for i, word in enumerate(doc):
            context = doc[max(0, i - window_size): i] + doc[i + 1: i + 1 + window_size]
            for c in context:
                embeddings[word] += learning_rate * (embeddings[c] - embeddings[word])

    return embeddings


def frase_a_vector(frase, embeddings, dim=50):
    """Convierte una frase en un vector promediando los embeddings de sus palabras. Palabras desconocidas se representan con vector cero."""
    tokens = limpiar_texto(frase)
    vectores = []
    for tok in tokens:
        if tok in embeddings:
            vectores.append(embeddings[tok])
        else:
            vectores.append(np.zeros(dim))

    if len(vectores) == 0:
        return np.zeros(dim)
    else:
        return np.mean(vectores, axis=0)


# --- Código principal ---
if __name__ == "__main__":
    csv_path = "cleaned_PHMRC_VAI_redacted_free_text.train.csv"

    # Leer dataset
    df = pd.read_csv(csv_path)

    # Verificar que exista la columna newid
    if "newid" not in df.columns:
        raise ValueError("El CSV no contiene la columna 'newid'.")

    # Generar embeddings de palabras
    embeddings = generar_embeddings(csv_path)

    # CSV de embeddings por instancia
    output_csv_instances = "instances_embeddings.csv"
    with open(output_csv_instances, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["newid"] + [f"dim_{i}" for i in range(embedding_dim)])

        for idx, row in df.iterrows():
            instance_id = row["newid"]
            text = str(row["gs_text34"])  # texto de la fila
            vector_inst = frase_a_vector(text, embeddings)
            writer.writerow([instance_id] + vector_inst.tolist())

    print(f"Embeddings de instancias guardados en '{output_csv_instances}'")
