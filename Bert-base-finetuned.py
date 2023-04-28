import pandas as pd
import re
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

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

def fine_tune(path_to_train_file):
    training_data = pd.read_csv(path_to_train_file).sample(n=10000)
    training_data.dropna(subset=['oh_label'], inplace=True)

    normalized_training_data = normalize_text(training_data)

    sentences = [s[0] for s in normalized_training_data]
    labels = [s[1] for s in normalized_training_data]

    tokenizer = AutoTokenizer.from_pretrained("ptaszynski/bert-base-polish-cyberbullying")

    encoding = tokenizer(sentences, padding=True, truncation=True, max_length=64, return_tensors='pt')

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    labels = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_mask, labels)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = 10
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    model = AutoModelForSequenceClassification.from_pretrained("ptaszynski/bert-base-polish-cyberbullying")

    optimizer = AdamW(model.parameters(), lr=1e-5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_loss_values, val_loss_values, train_preds, train_labels, val_preds, val_labels = [], [], [], [], [], []
    epochs = 3
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            batch_input_ids = torch.tensor(batch[0], dtype=torch.long).to(device)
            batch_attention_mask = torch.tensor(batch[1], dtype=torch.long).to(device)
            batch_labels = torch.tensor(batch[2], dtype=torch.long).to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
            loss = outputs.loss
            train_loss_values.append(loss.item())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            train_pred = torch.argmax(outputs.logits, dim=1).tolist()
            train_preds.extend(train_pred)
            train_label = batch_labels.tolist()
            train_labels.extend(train_label)

        print(f'Training loss: {train_loss / len(train_dataloader)}')
        print(classification_report(train_labels, train_preds))

        train_f1 = f1_score(train_labels, train_preds, average='macro')
        train_precision = precision_score(train_labels, train_preds, average='macro')
        train_recall = recall_score(train_labels, train_preds, average='macro')
        print(f'Training f1 score: {train_f1}, Training precision: {train_precision}, Training recall: {train_recall}')

        model.eval()
        val_loss = 0
        for batch in val_dataloader:
            batch_input_ids = torch.tensor(batch[0], dtype=torch.long).to(device)
            batch_attention_mask = torch.tensor(batch[1], dtype=torch.long).to(device)
            batch_labels = torch.tensor(batch[2], dtype=torch.long).to(device)
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
            loss = outputs.loss
            val_loss_values.append(loss.item())
            val_pred = torch.argmax(outputs.logits, dim=1).tolist()
            val_preds.extend(val_pred)
            val_label = batch_labels.tolist()
            val_labels.extend(val_label)
        print(f'Validation loss: {val_loss / len(val_dataloader)}')
        print(classification_report(val_labels, val_preds))

fine_tune("C:/Users/Prajju/OneDrive/Desktop/Sem-1/NLP/final_project/final_dataset.csv")
