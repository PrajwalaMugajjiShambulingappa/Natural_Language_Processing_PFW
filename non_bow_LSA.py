import pandas as pd
import numpy as np
import tensorflow as tf
import gensim.downloader as api
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

def tokenize_text(dataset):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True, oov_token='<OOV>')
    tokenizer.fit_on_texts(dataset['Text'])
    X = tokenizer.texts_to_sequences(dataset['Text'])
    X = tf.keras.preprocessing.sequence.pad_sequences(X, padding='post')
    y = dataset['oh_label']
    return X, y, tokenizer

def train_LM(path_to_train_file):
    training_data = pd.read_csv(path_to_train_file)
    training_data.dropna(subset=['oh_label'], inplace=True)

    word_vectors = api.load('glove-wiki-gigaword-300')  # Load pre-trained embeddings
    X_train = np.zeros((len(training_data), 300))  # Initialize empty array for sentence embeddings
    y_train = training_data['oh_label']

    # Convert each sentence to a sentence embedding
    for i, sentence in enumerate(training_data['Text']):
        words = sentence.split()
        word_vectors_in_sentence = [word_vectors[word] for word in words if word in word_vectors]
        if len(word_vectors_in_sentence) > 0:
            X_train[i] = np.mean(word_vectors_in_sentence, axis=0)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Train a LinearSVC classifier on the sentence embeddings
    svm_classifier = LinearSVC()
    svm_classifier.fit(X_train, y_train)

    # Evaluate the classifier on the validation set
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
    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        fmt="d",
        square=True,
        xticklabels=np.unique(y_val),
        yticklabels=np.unique(y_val),
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

train_LM("C:/Users/Prajju/OneDrive/Desktop/Sem-1/NLP/final project/final_dataset.csv")
