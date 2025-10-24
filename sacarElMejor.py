from clustering.best_model_export import train_and_export_best_model


train_and_export_best_model(metrics_path="./output/metrics.csv",embeddings_path="./dataset/cleaned_PHMRC_VAI_redacted_free_text.train.csv")