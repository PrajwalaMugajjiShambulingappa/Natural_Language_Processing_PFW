# Installing packages used.
import re
import nltk
import gensim.downloader as downloader
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neural_network import MLPClassifier

nltk.download('stopwords')
nltk.download('punkt')
import numpy as np
import pandas as pd
import pickle

# Load the pre-trained model (e.g. Google News)
w2v_model = downloader.load("word2vec-google-news-300")

# normalizing the dataset
def normalize_text(processed_text):
    # convert all text into lowercase
    processed_text = processed_text.lower()
    # remove all non-word and non-space characters
    processed_text = re.sub(r"[^\w\s]", "", processed_text)
    # Tokenize the text into individual words
    processed_text = nltk.word_tokenize(processed_text)
    # Removing stop words
    stopwords = nltk.corpus.stopwords.words('english')
    processed_text = [t for t in processed_text if t not in stopwords]

    embeddings = [w2v_model[token] for token in processed_text if token in w2v_model.key_to_index]

    max_length = 300
    if not embeddings:
        embeddings.append(np.zeros(max_length))

    # Pad embeddings with zeros to ensure they all have the same length
    padding_length = max_length - len(embeddings)
    if padding_length > 0:
        embeddings += [np.zeros_like(embeddings[0])] * padding_length
    elif padding_length < 0:
        embeddings = embeddings[:max_length]

    return np.array(embeddings).mean(axis=0)


def train_MLP_model(path_to_train_file, num_layers):

    # loading the dataset and normalizing the data
    train_data = pd.read_csv(path_to_train_file, usecols=['id', 'comment_text', 'toxic'])

    train_embeddings = [normalize_text(text) for text in train_data['comment_text']]

    # Convert labels to integers
    train_labels = train_data['toxic'].astype(int)
    train_data = None

    # Initialize MLP classifier
    hidden_layer_sizes = [128] + [64] * (num_layers - 1)
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=10, verbose=True, solver="adam",
                          learning_rate_init=0.5)

    # Train MLP classifier
    model.fit(train_embeddings, train_labels)

    # Calculate accuracy on training data
    train_predictions = model.predict(train_embeddings)
    train_accuracy = accuracy_score(train_labels, train_predictions)
    print(f'Training Accuracy: {train_accuracy:.4f}')

    # storing the trained model for each fold
    with open(f"MLP_model.pkl", "wb") as f:
        pickle.dump(model, f)

#train_MLP_model("train.csv", num_layers=1)
#train_MLP_model("train.csv", num_layers=2)
train_MLP_model("train.csv", num_layers=3)


def create_test_data():
    test_data = pd.read_csv('test.csv', usecols=['id', 'comment_text'])
    test_label_data = pd.read_csv('test_labels.csv', usecols=['id', 'toxic'])
    test_data = pd.merge(test_data, test_label_data, on='id', how='inner')
    test_label_data = None
    test_data = test_data[(test_data['toxic'] == 0) | (test_data['toxic'] == 1)].reset_index(drop=True)
    test_data.to_csv("test_data.csv")

def test_MLP_model(path_to_test_file, MLP_model):

    test_data = pd.read_csv(path_to_test_file, header=0)
    test_embeddings = [normalize_text(text) for text in test_data['comment_text']]
    test_labels = test_data['toxic'].astype(int)

    with open(MLP_model, 'rb') as f:
        trained_NLP_model = pickle.load(f)

    # Evaluate MLP classifier on test data
    predictions = trained_NLP_model.predict(test_embeddings)
    probabilities = trained_NLP_model.predict_proba(test_embeddings)[:, 1]

    # Add predicted probabilities and class labels to test data
    test_data['toxic_probability'] = probabilities
    test_data['class_prediction'] = ['toxic' if p > 0.5 else 'not toxic' for p in probabilities]

    accuracy = np.mean(predictions == test_labels)
    print(f'Test Accuracy: {accuracy:.4f}')
    f1 = f1_score(test_labels, predictions, average='micro')
    print(f'Test F1 Score (micro): {f1:.4f}')
    f1 = f1_score(test_labels, predictions, average='macro')
    print(f'Test F1 Score (macro): {f1:.4f}')
    f1 = f1_score(test_labels, predictions, average='weighted')
    print(f'Test F1 Score (weighted): {f1:.4f}')

    # Save test data with predicted probabilities and class labels to file
    test_data.to_csv("test_results.csv", index=False)

test_MLP_model("test_data.csv", "MLP_model.pkl")


