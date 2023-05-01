import mlflow
import torch
import pandas as pd
from transformers import AutoTokenizer, RobertaTokenizerFast
from torch.utils.data import DataLoader
from datasets import Dataset

def tokenize_and_dataload_df(model_name, batch_size, df_cb_data):
    if model_name == 'bert':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    elif model_name == 'distilbert':
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')
    elif model_name == 'roberta':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base-cased')
    elif model_name == 'albert':
        tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
    else:
        tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')

    dataset = Dataset.from_pandas(df_cb_data)
    dataset_torch = dataset.map(lambda e: tokenizer(e['Text'], truncation=True, max_length=512, padding='max_length'), batched=True)
    dataset_torch.set_format(type='torch', columns=['input_ids', 'labels'])

    train_size = int(0.8 * len(dataset_torch))
    test_size = len(dataset_torch) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset_torch, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

def save_model(model, model_type, params, epoch, val_accuracy):
    params['current_epoch'] = epoch
    params['val_accuracy'] = val_accuracy
    accuracy = round(val_accuracy*100)
    mlflow.pytorch.log_model(model, f'cd_cb_{model_type}_epoch_{epoch}_acc_{accuracy}')

def suggest_hyperparameters(trial,
                            lr_range,
                            optimizer_list,
                            epoch_range,
                            model_list,
                            batch_size_list,
                            gamma_range,
                            scheduler_list):

    lr = trial.suggest_float("lr", lr_range[0], lr_range[1], log=True)
    optimizer_name = trial.suggest_categorical("optimizer_name", optimizer_list)
    epochs = trial.suggest_int("epochs", epoch_range[0], epoch_range[1], step=1)
    model = trial.suggest_categorical("model", model_list)
    batch_size = trial.suggest_categorical("batch_size", batch_size_list)
    gamma = trial.suggest_float("gamma", gamma_range[0], gamma_range[1], step=0.1)
    scheduler = trial.suggest_categorical("scheduler", scheduler_list)

    return lr, optimizer_name, epochs, model, batch_size, gamma, scheduler