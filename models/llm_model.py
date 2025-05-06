import logging
import json
import torch
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

from utils import plot_confusion_matrix, plot_graph

def vectorize_bert(sentences, device='cpu'):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.eval()

    bert_model.to(device)
    embeddings = []

    for i, text in enumerate(tqdm(sentences)):
        text = str(text)

        if '[E1]' not in text or '[/E1]' not in text or '[E2]' not in text or '[/E2]' not in text:
            continue

        try:
            e1 = text.split('[E1]')[1].split('[/E1]')[0]
            e2 = text.split('[E2]')[1].split('[/E2]')[0]
        except IndexError:
            continue

        inputs_e1 = tokenizer(e1, return_tensors="pt")
        inputs_e2 = tokenizer(e2, return_tensors="pt")

        with torch.no_grad():
            out_e1 = bert_model(**inputs_e1).last_hidden_state.mean(dim=1)
            out_e2 = bert_model(**inputs_e2).last_hidden_state.mean(dim=1)

        combined = torch.cat((out_e1, out_e2), dim=1)
        embeddings.append(combined.numpy().flatten())

    return np.array(embeddings)

def mark_entities(sentence):
    sentence = sentence.replace('<e1>', '[E1]').replace('</e1>', '[/E1]')
    sentence = sentence.replace('<e2>', '[E2]').replace('</e2>', '[/E2]')
    return sentence

def train_llm(df):
    logging.info("Vectorizing via BERT...")
    df['bert_input'] = df['sentence'].apply(mark_entities)
    X_bert = vectorize_bert(df['bert_input'])
    y = df['relation']
    X_train, X_test, y_train, y_test = train_test_split(X_bert, y, test_size=0.2, random_state=42)

    logging.info("LogisticRegression learning and predicting...")
    clf = LogisticRegression(random_state=42, solver='lbfgs', multi_class='multinomial', max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    logging.info("Metrics and predictions saving...")
    report = classification_report(y_test, y_pred, output_dict=True)
    logging.info("LLM model classification report:\n%s", report)
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            logging.info(
                f"{label} - precision: {metrics['precision']:.3f}, "
                f"recall: {metrics['recall']:.3f}, "
                f"f1-score: {metrics['f1-score']:.3f}, "
                f"support: {metrics['support']}"
            )
        else:
            logging.info(f"{label}: {metrics:.3f}")
            
    with open("results/metrics_llm.json", "w") as f:
        json.dump(report, f, indent=2)
    with open("results/predictions_llm.json", "w") as f:
        json.dump({"y_true": y_test.tolist(), "y_pred": y_pred.tolist()}, f, indent=2)

    plot_confusion_matrix(y_test, y_pred, labels=df['relation'].unique(), filename="results/confusion_matrix_llm.png")
    plot_graph(df, y_pred, filename="results/graph_llm.png")

    logging.info("LLM model pipeline complete")