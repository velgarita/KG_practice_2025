import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def load_dataset():
    df = pd.read_parquet("hf://datasets/SemEvalWorkshop/sem_eval_2010_task_8/data/train-00000-of-00001.parquet")
    return df

def plot_confusion_matrix(y_test, y_pred, labels, filename):
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=90, cmap='Purples', values_format='d')
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()

def plot_graph(df, y_pred, filename):
    relation_dict = relation_dict = {
    0: 'Cause-Effect(e1,e2)',
    1: 'Cause-Effect(e2,e1)',
    2: 'Component-Whole(e1,e2)',
    3: 'Component-Whole(e2,e1)',
    4: 'Content-Container(e1,e2)',
    5: 'Content-Container(e2,e1)',
    6: 'Entity-Destination(e1,e2)',
    7: 'Entity-Destination(e2,e1)',
    8: 'Entity-Origin(e1,e2)',
    9: 'Entity-Origin(e2,e1)',
    10: 'Instrument-Agency(e1,e2)',
    11: 'Instrument-Agency(e2,e1)',
    12: 'Member-Collection(e1,e2)',
    13: 'Member-Collection(e2,e1)',
    14: 'Message-Topic(e1,e2)',
    15: 'Message-Topic(e2,e1)',
    16: 'Product-Producer(e1,e2)',
    17: 'Product-Producer(e2,e1)',
    18: 'Other'
    }

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    G = nx.DiGraph()
    for i in range(min(len(df_test), 60)):
        row = df_test.iloc[i]
        pred_label = y_pred[i]
        relation = relation_dict.get(pred_label, "Unknown")
        sent = row['sentence']
        e1 = sent.split('<e1>')[1].split('</e1>')[0]
        e2 = sent.split('<e2>')[1].split('</e2>')[0]
        if '(e1,e2)' in relation:
            G.add_edge(e1, e2, label=relation)
        elif '(e2,e1)' in relation:
            G.add_edge(e2, e1, label=relation)
        else:
            G.add_edge(e1, e2, label=relation)

    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=1, iterations=100)
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color='lightblue', font_size=10, arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Relations graph")
    plt.savefig(filename)
    plt.close()
