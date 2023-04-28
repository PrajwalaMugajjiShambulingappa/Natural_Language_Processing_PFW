import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV

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

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(training_data['Text'])
    y_train = training_data['oh_label']

    svd = TruncatedSVD(n_components=150)
    X_train = svd.fit_transform(X_train)

    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'class_weight': [None, 'balanced']
    }

    svc_classifier = LinearSVC()
    svc_grid_search = GridSearchCV(svc_classifier, param_grid, scoring='f1_weighted', cv=5, n_jobs=-1)
    svc_grid_search.fit(X_train, y_train)

    best_svc = svc_grid_search.best_estimator_
    print("Best hyperparameters:", svc_grid_search.best_params_)
    print("Best validation score:", svc_grid_search.best_score_)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    best_svc.fit(X_train, y_train)

    accuracy = best_svc.score(X_val, y_val)
    print("Validation accuracy:", accuracy)

    y_pred = best_svc.predict(X_val)
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
        xticklabels=best_svc.classes_,
        yticklabels=best_svc.classes_,
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

train_LM("C:/Users/Prajju/OneDrive/Desktop/Sem-1/NLP/final project/final_dataset.csv")