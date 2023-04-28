import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score

dataset = pd.read_csv("C:/Users/Prajju/OneDrive/Desktop/Sem-1/NLP/final_project/final_dataset.csv").sample(n=10000)
dataset = dataset.reset_index(drop=True)

tokenizer = AutoTokenizer.from_pretrained("juliensimon/reviews-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("juliensimon/reviews-sentiment-analysis")
model_name = 'juliensimon/reviews-sentiment-analysis'

# Load 10000 examples from dataset
test_dataset = dataset['Text'][:10000]

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

# Loop through models and test accuracy on your dataset
for i, model in enumerate([model]):
    misclassified_indices = []
    y_true, y_pred = [], []
    for j, text in enumerate(test_dataset):
        label = dataset['oh_label'][j]
        prediction = predict_sentiment(model, tokenizer=tokenizer, text=text)
        y_true.append(int(label))
        y_pred.append(int(prediction))
        if int(prediction) != int(label):
            misclassified_indices.append(j)
        #   print(f'Misclassified example {len(misclassified_indices)}:')
        #   print(f'True Label: {label}, Predicted Label: {prediction}')
        #else:
        #    print(f'Correctly classified example {j}:')
        #    print(f'True Label: {label}, Predicted Label: {prediction}')

    accuracy = (len(test_dataset) - len(misclassified_indices)) / len(test_dataset)
    accuracy_percentage = accuracy * 100
    print(f'{model_name}: {accuracy_percentage:.2f}%')

    print(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    print(f'{model_name}: F1 score = {f1}, Precision = {precision}, Recall = {recall}')
