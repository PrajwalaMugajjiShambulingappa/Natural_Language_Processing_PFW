import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC

# normalizing the training data
def normalize_text(dataset):
    normalized_training_data = []
    for index, row in dataset.iterrows():
        data = row['Text']
        target = row['oh_label']
        if isinstance(data, str):
            for f in re.findall("([A-Z]+)", data):
                data = data.replace(f, f.lower())
            processed_text = re.sub(r"[^\w\s]", "", data)
            processed_text = re.split("\W", processed_text)
            processed_text = [i for i in processed_text if i != '']

            normalized_training_data.append((' '.join(processed_text), target))
    return normalized_training_data


def train_LM(path_to_train_file):
    training_data = pd.read_csv(path_to_train_file)
    training_data.dropna(subset=['oh_label'], inplace=True)

    normalized_training_data = normalize_text(training_data)
    X_train, X_val, y_train, y_val = train_test_split(
        [data[0] for data in normalized_training_data],
        [data[1] for data in normalized_training_data],
        test_size=0.2,
        random_state=42
    )
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    svm_classifier = LinearSVC()
    svm_classifier.fit(X_train, y_train)

    accuracy = svm_classifier.score(X_val, y_val)
    print("Validation accuracy:", accuracy)

    y_pred = svm_classifier.predict(X_val)
    precision = precision_score(y_val, y_pred, average='weighted')
    print("Validation precision:", precision)

    recall = recall_score(y_val, y_pred, average='weighted')
    print("Validation recall:", recall)

    f1 = f1_score(y_val, y_pred, average='weighted')
    print("Validation f1:", f1)

    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", square=True, xticklabels=svm_classifier.classes_,
                yticklabels=svm_classifier.classes_)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

train_LM("C:/Users/Prajju/OneDrive/Desktop/Sem-1/NLP/final project/final_dataset.csv")