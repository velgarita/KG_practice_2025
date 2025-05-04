import logging
import json
import torch
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

from utils import plot_confusion_matrix, plot_graph

def vectorize(sentences, batch_size=32):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.eval()

    device = torch.device("cpu")
    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding with BERT"):
            batch = sentences[i:i+batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=128
            ).to(device)

            output = bert_model(**encoded)
            cls_embeddings = output.last_hidden_state[:, 0, :]
            embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(embeddings)

def train_llm(df):
    logging.info("Vectorizing...")
    X_bert = vectorize(df['sentence'].tolist())
    y = df['relation']
    X_train, X_test, y_train, y_test = train_test_split(X_bert, y, test_size=0.2, random_state=42)

    logging.info("Model learning and predicting...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    logging.info("Metrics and predictions saving...")
    report = classification_report(y_test, y_pred, output_dict=True)
    logging.info("LLM model classification report:\n%s", report)
    with open("results/metrics_llm.json", "w") as f:
        json.dump(report, f, indent=2)
    with open("results/predictions_llm.json", "w") as f:
        json.dump({"y_true": y_test.tolist(), "y_pred": y_pred.tolist()}, f, indent=2)

    plot_confusion_matrix(y_test, y_pred, labels=df['relation'].unique(), filename="results/confusion_matrix_llm.png")
    plot_graph(df, y_pred, filename="results/graph_llm.png")

    logging.info("LLM model pipeline complete")