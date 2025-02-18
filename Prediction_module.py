# utils/predictor.py
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_category(model, tokenizer, label_tokenizer, text, max_length):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length, padding='post')
    prediction = model.predict(padded)
    index = np.argmax(prediction) + 1
    labels = list(label_tokenizer.word_index.keys())
    return labels[index - 1] if index <= len(labels) else "Unknown"
