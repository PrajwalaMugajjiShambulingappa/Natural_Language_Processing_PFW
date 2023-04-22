import torch
import torchvision
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
from sklearn.metrics import f1_score, precision_score, recall_score

# Load IMDb dataset
rotten_tomatoes_dataset = load_dataset('rotten_tomatoes').shuffle(seed=5)

tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-rotten_tomatoes-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-rotten_tomatoes-sentiment-analysis")
model_name = 'mrm8488/distilroberta-finetuned-rotten_tomatoes-sentiment-analysis'

tokenizer_1 = AutoTokenizer.from_pretrained("RJZauner/distilbert_rotten_tomatoes_sentiment_classifier")
model_1 = AutoModelForSequenceClassification.from_pretrained("RJZauner/distilbert_rotten_tomatoes_sentiment_classifier")
model_name_1 = 'RJZauner/distilbert_rotten_tomato'

tokenizer_2 = AutoTokenizer.from_pretrained("Ghost1/bert-base-uncased-finetuned_for_sentiment_analysis1-sst2")
model_2 = AutoModelForSequenceClassification.from_pretrained("Ghost1/bert-base-uncased-finetuned_for_sentiment_analysis1-sst2")
model_name_2 = 'Ghost1/bert-base-uncased-finetuned'

# Load 1000 examples from test set of IMDb dataset
test_dataset = rotten_tomatoes_dataset['test'][:1000]['text']

# Function to predict sentiment using a given model
def predict_sentiment(model, tokenizer, text):
    max_length = model.config.max_position_embeddings
    inputs = tokenizer.encode_plus(text, add_special_tokens=True, truncation=True, max_length=max_length, return_tensors='pt')
    outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    if len(probabilities) == 2:
        if probabilities[0] > probabilities[1]:
            prediction = '0'
        else:
            prediction = '1'
    else:
        prediction = str(probabilities.index(max(probabilities)))
    return prediction

# Loop through models and test accuracy on Rotten Tomatoes test set
for i, model in enumerate([model_2]):
    misclassified_indices = []
    y_true, y_pred = [], []
    for j, text in enumerate(test_dataset):
        label = rotten_tomatoes_dataset['test'][j]['label']
        prediction = predict_sentiment(model, tokenizer=tokenizer_2, text=text)
        y_true.append(int(label))
        y_pred.append(int(prediction))
        if int(prediction) != int(label):
            misclassified_indices.append(j)
        #   print(f'Misclassified example {len(misclassified_indices)}:')
        #     print(f'True Label: {label}, Predicted Label: {prediction}')
        else:
            print(f'Correctly classified example {j}:')
        #     print(f'True Label: {label}, Predicted Label: {prediction}')

    accuracy = (len(test_dataset) - len(misclassified_indices)) / len(test_dataset)
    accuracy_percentage = accuracy * 100
    print(f'{model_name_2}: {accuracy_percentage:.2f}%')

    print(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    print(f'{model_name_2}: F1 score = {f1}, Precision = {precision}, Recall = {recall}')
