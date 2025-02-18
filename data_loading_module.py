# src/data_loader.py
import pandas as pd

def load_data(filename):
    df = pd.read_csv(filename)
    articles = df['text'].tolist()
    labels = df['category'].tolist()
    return articles, labels
