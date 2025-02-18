# main.py
from config import VOCAB_SIZE, EMBEDDING_DIM, MAX_LENGTH, OOV_TOKEN, TRAINING_SPLIT, EPOCHS
from src.data_loader import load_data
from src.preprocessor import clean_articles, tokenize_and_pad
from src.model import build_rnn_model
from src.trainer import train_model
from utils.plotter import plot_metrics
from utils.predictor import predict_category

# Load and preprocess data
articles, labels = load_data('news-data.csv')
cleaned_articles = clean_articles(articles)
train_padded, train_labels, val_padded, val_labels, tokenizer, label_tokenizer = tokenize_and_pad(
    cleaned_articles, labels, VOCAB_SIZE, MAX_LENGTH, OOV_TOKEN, TRAINING_SPLIT)

# Build and train model
model = build_rnn_model(VOCAB_SIZE, EMBEDDING_DIM, MAX_LENGTH)
history = train_model(model, train_padded, train_labels, val_padded, val_labels, EPOCHS)

# Display results
plot_metrics(history)

# Make predictions
sample_text = "Stock markets surge as tech companies report gains."
predicted_category = predict_category(model, tokenizer, label_tokenizer, sample_text, MAX_LENGTH)
print(f"Predicted Category: {predicted_category}")
