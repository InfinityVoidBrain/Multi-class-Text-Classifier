# src/preprocessor.py
import numpy as np
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

stop_words = set(stopwords.words('english'))

def clean_articles(articles):
    cleaned = []
    for article in articles:
        for word in stop_words:
            article = article.replace(f' {word} ', ' ')
        cleaned.append(article)
    return cleaned

def tokenize_and_pad(articles, labels, vocab_size, max_length, oov_token, training_split):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(articles)
    sequences = tokenizer.texts_to_sequences(articles)
    padded = pad_sequences(sequences, maxlen=max_length, padding='post')

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    label_sequences = np.array(label_tokenizer.texts_to_sequences(labels))

    split_idx = int(len(padded) * training_split)
    return (padded[:split_idx], label_sequences[:split_idx],
            padded[split_idx:], label_sequences[split_idx:], tokenizer, label_tokenizer)
