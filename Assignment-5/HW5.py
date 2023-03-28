# Installing packages used.
import nltk
import re
import pandas as pd
import numpy as np
nltk.download('stopwords')
nltk.download('punkt')
from collections import Counter
import gensim.downloader as downloader

# Creating toxic subset from the train dataset
def create_data_toxic(path_to_train_file):
    train_df = pd.read_csv(path_to_train_file)

    toxic_data = train_df[train_df['toxic'] == 1]
    toxic_data.to_csv('toxic_data.csv', index=False)
    print("toxic test created!")

create_data_toxic("train.csv")

# Creating non-toxic subset from the train dataset
def create_data_non_toxic(path_to_train_file):
    train_df = pd.read_csv(path_to_train_file)

    non_toxic_data = train_df[train_df['toxic'] == 0]
    non_toxic_data.to_csv('non-toxic_data.csv', index=False)
    print("non-toxic test created!")

create_data_non_toxic("train.csv")

# Creating toxic subset from the test dataset for validation
def create_test_data_toxic (path_to_test_file, path_to_test_label):
  test_df = pd.read_csv(path_to_test_file)
  test_labels_df = pd.read_csv(path_to_test_label)

  merged_df = pd.merge(test_df, test_labels_df, on='id')

  toxic_data = merged_df[merged_df['toxic'] == 1]
  toxic_data.to_csv('test_toxic_data.csv', index=False)

create_test_data_toxic("test.csv", "test_labels.csv")

# Creating non-toxic subset from the test dataset for validation
def create_test_data_non_toxic (path_to_test_file, path_to_test_label):
  test_df = pd.read_csv(path_to_test_file)
  test_labels_df = pd.read_csv(path_to_test_label)

  merged_df = pd.merge(test_df, test_labels_df, on='id')

  toxic_data = merged_df[merged_df['toxic'] == 0]
  toxic_data.to_csv('test_non-toxic_data.csv', index=False)

create_test_data_non_toxic("test.csv", "test_labels.csv")

# normalizing the dataset
def normalize_text(processed_text):

    # join all the elements of the 'comment_text'
    processed_text = " ".join(processed_text['comment_text'])
    # convert all text into lowercase
    processed_text = processed_text.lower()
    # remove all non-word and non-space characters
    processed_text = re.sub(r"[^\w\s]", "", processed_text)
    # Tokenize the text into individual words
    processed_text = nltk.word_tokenize(processed_text)
    # Removing stop words
    stopwords = nltk.corpus.stopwords.words('english')
    processed_text = [t for t in processed_text if t not in stopwords]

    return processed_text

def compare_texts_w2v(file_one, file_two, k):

    # loading the pretrained word2vec model
    w2v_model = downloader.load("word2vec-google-news-300")

    # loading the toxic and non-toxic dataset
    toxic_dataset = pd.read_csv(file_one)
    non_toxic_dataset = pd.read_csv(file_two)

    # preprocessing the dataset
    print("preprocessing for toxic dataset started.")
    toxic_comments = normalize_text(toxic_dataset)
    #print(toxic_comments)

    print("preprocessing for non-toxic dataset started.")
    non_toxic_comments = normalize_text(non_toxic_dataset)
    #print(non_toxic_comments)

    # Calculating count of every word in the dataset
    words_1 = Counter(toxic_comments)
    words_2 = Counter(non_toxic_comments)

    # fetching the most common words
    top_words_1 = words_1.most_common(k)
    top_words_2 = words_2.most_common(k)
    print(top_words_1, top_words_2)

    # calculating the similaarity score
    similarity = 0
    count = 0
    for i, _ in top_words_1:
        for j, _ in top_words_2:
            try:
                sum = w2v_model.similarity(i, j)
                if not np.isnan(sum):
                    similarity += sum
                    count += 1
            except:
                continue

    # similarity_score = sum / k**2
    similarity_score = similarity / count
    print(f"Similarity score between {file_one} and {file_two} based on top {k} words: {similarity_score:.3f}")

compare_texts_w2v("toxic_data.csv", "non-toxic_data.csv", k=10)
#compare_texts_w2v("toxic_data.csv", "non-toxic_data.csv", k=5)
#compare_texts_w2v("toxic_data.csv", "non-toxic_data.csv", k=20)
#compare_texts_w2v("test_toxic_data.csv", "test_non-toxic_data.csv", k=5)
#compare_texts_w2v("test_toxic_data.csv", "test_non-toxic_data.csv", k=10)
#compare_texts_w2v("test_toxic_data.csv", "test_non-toxic_data.csv", k=20)

def normalize_txt_text(processed_text):

    print("preprocessing for word dataset started.")

    # lowercasing of text
    for f in re.findall("([A-Z]+)", processed_text):
        processed_text = processed_text.replace(f, f.lower())
    # remove all non-word and non-space characters
    processed_text = re.sub(r"[^\w\s]", "", processed_text)
    # tokenizing
    processed_text = nltk.word_tokenize(processed_text)
    # removing stop words
    stopwords = nltk.corpus.stopwords.words('english')
    processed_text = [t for t in processed_text if t not in stopwords]

    return processed_text

def doc_overview_w2v(text_file, k, n):

    # loading the pretrained word2vec model
    w2v_model = downloader.load("word2vec-google-news-300")

    # loading and pretraining the model
    txt_dataset = pd.read_csv(text_file, delimiter="\t")
    txt_dataset = " ".join(txt_dataset.iloc[:, 0].values)
    txt_dataset = normalize_txt_text(txt_dataset)

    # finding the frequency of all the words in the dataset
    word = Counter(txt_dataset)

    # fetching the most common words
    top_words = word.most_common(k)
    print(top_words)

    # find words that are similar to the most common words.
    for i, _ in top_words:
        print(w2v_model.most_similar(i, topn=n))

doc_overview_w2v("warofworlds.txt", k=5, n=5)
#doc_overview_w2v('on_liberty.txt', k=5, n=5)
#doc_overview_w2v('kingarthur.txt', k=5, n=10)
