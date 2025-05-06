import re
import json
import spacy
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from utils import plot_confusion_matrix, plot_graph


nlp = spacy.load("en_core_web_sm")


def get_words_between_entities(text: str) -> str:
    e1_start = text.find('<e1>')
    e1_end = text.find('</e1>')
    e2_start = text.find('<e2>')
    e2_end = text.find('</e2>')
    if e1_end < e2_start:
        between = text[e1_end+5 : e2_start]
    else:
        between = text[e2_end+5 : e1_start]
    return between.strip()

def get_pos_tags_between_entities(text: str) -> list:
    pattern1 = r"<e1>(.*?)</e1>.*?<e2>(.*?)</e2>"
    pattern2 = r"<e2>(.*?)</e2>.*?<e1>(.*?)</e1>"
    if re.search(pattern1, text):
        between = text.split('</e1>')[1].split('<e2>')[0]
    elif re.search(pattern2, text):
        between = text.split('</e2>')[1].split('<e1>')[0]
    else:
        return []
    between = between.strip()
    doc = nlp(between)
    return [token.pos_ for token in doc]

def get_dependency_path_between_entities(text: str) -> list:
    e1 = re.search(r"<e1>(.*?)</e1>", text).group(1)
    e2 = re.search(r"<e2>(.*?)</e2>", text).group(1)
    clean_text = re.sub(r"</?e[12]>", "", text)
    doc = nlp(clean_text)
    token_e1 = next((t for t in doc if t.text == e1), None)
    token_e2 = next((t for t in doc if t.text == e2), None)
    if not token_e1 or not token_e2:
        return []

    def get_ancestors(token):
        return list(token.ancestors)

    ancestors_e1 = [token_e1] + get_ancestors(token_e1)
    ancestors_e2 = [token_e2] + get_ancestors(token_e2)

    common = set(ancestors_e1) & set(ancestors_e2)
    if not common:
        return []

    lca = next(t for t in ancestors_e1 if t in common)

    path_e1 = []
    curr = token_e1
    while curr != lca:
        path_e1.append(curr.dep_)
        curr = curr.head

    path_e2 = []
    curr = token_e2
    while curr != lca:
        path_e2.append(curr.dep_)
        curr = curr.head

    return path_e1 + [lca.dep_] + list(reversed(path_e2))

def get_entity_ner_types(text: str) -> tuple:
    e1_match = re.search(r"<e1>(.*?)</e1>", text)
    e2_match = re.search(r"<e2>(.*?)</e2>", text)
    if not e1_match or not e2_match:
        return ("O", "O")

    e1_text = e1_match.group(1)
    e2_text = e2_match.group(1)
    clean_text = re.sub(r"</?e[12]>", "", text)
    doc = nlp(clean_text)

    e1_type = "O"
    e2_type = "O"
    for ent in doc.ents:
        if ent.text == e1_text:
            e1_type = ent.label_
        if ent.text == e2_text:
            e2_type = ent.label_

    return (e1_type, e2_type)


def vectorize_features(df, feature_columns):
    matrices = []
    vectorizers = {}

    for col in feature_columns:
        processed = df[col].apply(lambda x: " ".join(map(str, x)) if isinstance(x, list) else str(x))
        
        vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        X_col = vec.fit_transform(processed)

        matrices.append(X_col)
        vectorizers[col] = vec

    X = hstack(matrices)
    return X, vectorizers


def train_classic(df):
    logging.info("Features extracting...")
    df['words between entities'] = [get_words_between_entities(text) for text in df['sentence']]
    df['pos tags'] = [get_pos_tags_between_entities(text) for text in df['sentence']]
    df['syntactic dependencies'] = [get_dependency_path_between_entities(text) for text in df['sentence']]
    df['ner types'] = [get_entity_ner_types(text) for text in df['sentence']]

    logging.info("Vectorizing via TF-IDF...")
    cols = ['words between entities', 'pos tags', 'syntactic dependencies', 'ner types']
    X, vectorizers = vectorize_features(df, cols)
    y = df['relation']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logging.info("RandomForest learning and predicting...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    logging.info("Metrics and predictions saving...")
    report = classification_report(y_test, y_pred, output_dict=True)
    logging.info("Classic model classification report:")
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

    with open("results/metrics_classic.json", "w") as f:
        json.dump(report, f, indent=2)
    with open("results/predictions_classic.json", "w") as f:
        json.dump({"y_true": y_test.tolist(), "y_pred": y_pred.tolist()}, f, indent=2)

    plot_confusion_matrix(y_test, y_pred, labels=df['relation'].unique(), filename="results/confusion_matrix_classic.png")
    plot_graph(df, y_pred, filename="results/graph_classic.png")

    logging.info("Classic model pipeline complete")