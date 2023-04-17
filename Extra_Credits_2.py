# Installing packages used.
import re
import nltk
import gensim
import gensim.downloader as downloader
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from gensim.models import Word2Vec

nltk.download('stopwords')
nltk.download('punkt')
import numpy as np
import pandas as pd
import pickle

# Load the pre-trained model
w2v_model = downloader.load("word2vec-google-news-300")
w2v_model.save_word2vec_format("word2vec-google-news-300.bin", binary=True)

# Normalizing the dataset
def normalize_text(processed_text, w2v_model=None):
    # Convert all text into lowercase
    processed_text = processed_text.lower()
    # Remove all non-word and non-space characters
    processed_text = re.sub(r"[^\w\s]", "", processed_text)
    # Tokenize the text into individual words
    processed_text = nltk.word_tokenize(processed_text)
    # Removing stop words
    stopwords = nltk.corpus.stopwords.words('english')
    processed_text = [t for t in processed_text if t not in stopwords]

    if w2v_model is not None:
        embeddings = [w2v_model[token] for token in processed_text if token in w2v_model.key_to_index]
    else:
        embeddings = []

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

def update_embeddings(path_to_train_file):
    # Load training data
    train_data = pd.read_csv(path_to_train_file, usecols=['comment_text'])
    # Tokenize the text into individual words
    tokenized_train_data = [nltk.word_tokenize(text.lower()) for text in train_data['comment_text']]

    # Load pre-trained word2vec model
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format('word2vec-google-news-300.bin', binary=True)

    # Train the Word2Vec model on the corpus
    model = gensim.models.Word2Vec(
        tokenized_train_data,
        vector_size=300,
        min_count=1,
        epochs=5,
        workers=4,
        window=5,
        sg=0,
        sample=0.001,
        hs=1
    )

    # Fine-tune the model's weights with the pre-trained model's weights
    model.build_vocab([list(w2v_model.key_to_index.keys())], update=True)
    model.intersect_word2vec_format('word2vec-google-news-300.bin', binary=True, lockf=1.0)

    # Save the fine-tuned embeddings
    model.wv.save_word2vec_format('fine_tuned_embeddings.bin', binary=True)

    return 'fine_tuned_embeddings_pretrained.bin'

update_embeddings("train.csv")

def train_MLP_model(path_to_train_file, num_layers, embeddings_file):

    if embeddings_file == "word2vec-google-news-300.bin":
        x = True
    else:
        x = False

    # Load the embeddings
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_file, binary=True)

    # Load the training data
    train_data = pd.read_csv(path_to_train_file, usecols=['comment_text', 'toxic'])

    # Normalize the text and extract embeddings
    train_embeddings = [normalize_text(text, w2v_model) for text in train_data['comment_text']]

    # Convert labels to integers
    train_labels = train_data['toxic'].astype(int)

    # Initialize MLP classifier
    hidden_layer_sizes = [128] + [64] * (num_layers - 1)
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=10, verbose=True, solver="adam",
                          learning_rate_init=0.005)

    # Train MLP classifier
    model.fit(train_embeddings, train_labels)

    # Calculate accuracy on training data
    train_predictions = model.predict(train_embeddings)
    train_accuracy = accuracy_score(train_labels, train_predictions)
    print(f'Training Accuracy: {train_accuracy:.4f}')

    # Save the trained model
    if x == True:
        with open(f"MLP_model_{num_layers}layers.pkl", "wb") as f:
            pickle.dump(model, f)
    else:
        with open(f"MLP_model_{num_layers}layers_finetuned.pkl", "wb") as f:
            pickle.dump(model, f)

#train_MLP_model("train.csv", num_layers=1, embeddings_file="word2vec-google-news-300.bin")
#train_MLP_model("train.csv", num_layers=2, embeddings_file="word2vec-google-news-300.bin")
#train_MLP_model("train.csv", num_layers=3, embeddings_file="word2vec-google-news-300.bin")

#train_MLP_model("train.csv", num_layers=2, embeddings_file="fine_tuned_embeddings.bin")
#train_MLP_model("train.csv", num_layers=1, embeddings_file="fine_tuned_embeddings.bin")
#train_MLP_model("train.csv", num_layers=3, embeddings_file="fine_tuned_embeddings.bin")

def create_test_data():
    test_data = pd.read_csv('test.csv', usecols=['id', 'comment_text'])
    test_label_data = pd.read_csv('test_labels.csv', usecols=['id', 'toxic'])
    test_data = pd.merge(test_data, test_label_data, on='id', how='inner')
    test_label_data = None
    test_data = test_data[(test_data['toxic'] == 0) | (test_data['toxic'] == 1)].reset_index(drop=True)
    test_data.to_csv("EC_test_data.csv")


def evaluate_model(model_path, embeddings_file, test_file):

    create_test_data()
    # Load the embeddings
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_file, binary=True)

    # Load the test data
    test_data = pd.read_csv(test_file, usecols=['comment_text', 'toxic'])

    # Normalize the text and extract embeddings
    test_embeddings = [normalize_text(text, w2v_model) for text in test_data['comment_text']]

    # Convert labels to integers
    test_labels = test_data['toxic'].astype(int)

    # Load the trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Predict labels for test data using the trained model
    test_predictions = model.predict(test_embeddings)

    # Calculate F1 score and accuracy on test data
    f1 = f1_score(test_labels, test_predictions)
    acc = accuracy_score(test_labels, test_predictions)

    # Print results
    print(f"Test F1 score: {f1:.4f}")
    print(f"Test accuracy: {acc:.4f}")

    return f1, acc

# Evaluate the MLP model trained on pre-trained embeddings
#pretrained_model_path = "MLP_model_2layers.pkl"
#pretrained_embeddings_file = "word2vec-google-news-300.bin"
#test_file = "EC_test_data.csv"
#pretrained_f1, pretrained_acc = evaluate_model(pretrained_model_path, pretrained_embeddings_file, test_file)

# Evaluate the MLP model trained on fine-tuned embeddings
#fintuned_model_path = "MLP_model_2layers_finetuned.pkl"
#fintuned_embeddings_file = "fine_tuned_embeddings.bin"
#test_file = "EC_test_data.csv"
#fintuned_f1, fintuned_acc = evaluate_model(fintuned_model_path, fintuned_embeddings_file, test_file)

# Evaluate the MLP model trained on pre-trained embeddings
#pretrained_model_path = "MLP_model_1layers.pkl"
#pretrained_embeddings_file = "word2vec-google-news-300.bin"
#test_file = "EC_test_data.csv"
#pretrained_f1, pretrained_acc = evaluate_model(pretrained_model_path, pretrained_embeddings_file, test_file)

# Evaluate the MLP model trained on fine-tuned embeddings
#fintuned_model_path = "MLP_model_1layers_finetuned.pkl"
#fintuned_embeddings_file = "fine_tuned_embeddings.bin"
#test_file = "EC_test_data.csv"
#fintuned_f1, fintuned_acc = evaluate_model(fintuned_model_path, fintuned_embeddings_file, test_file)

# Evaluate the MLP model trained on pre-trained embeddings
#pretrained_model_path = "MLP_model_3layers.pkl"
#pretrained_embeddings_file = "word2vec-google-news-300.bin"
#test_file = "EC_test_data.csv"
#pretrained_f1, pretrained_acc = evaluate_model(pretrained_model_path, pretrained_embeddings_file, test_file)

# Evaluate the MLP model trained on fine-tuned embeddings
#fintuned_model_path = "MLP_model_3layers_finetuned.pkl"
#fintuned_embeddings_file = "fine_tuned_embeddings.bin"
#test_file = "EC_test_data.csv"
#fintuned_f1, fintuned_acc = evaluate_model(fintuned_model_path, fintuned_embeddings_file, test_file)

# Print the results
#print("Pre-trained MLP model F1 score:", pretrained_f1)
#print("Pre-trained MLP model accuracy:", pretrained_acc)
#print("Fine-tuned MLP model F1 score:", fintuned_f1)
#print("Fine-tuned MLP model accuracy:", fintuned_acc)
