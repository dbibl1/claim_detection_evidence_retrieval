import gc

import mlflow
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification, \
    RobertaForSequenceClassification, AlbertForSequenceClassification, XLNetForSequenceClassification

from train import train
from utils import tokenize_and_dataload_df, save_model, suggest_hyperparameters
from validate import validate
import config


def objective(trial,
              lr_range,
              optimizer_list,
              epoch_range,
              model_list,
              batch_size_list,
              gamma_range,
              scheduler_list,
              df_train):

    best_val_loss = float('Inf')

    with mlflow.start_run():
        lr, optimizer_name, epochs, model_type, batch_size, gamma, sched = suggest_hyperparameters(trial,
                                                                                                   lr_range,
                                                                                                   optimizer_list,
                                                                                                   epoch_range,
                                                                                                   model_list,
                                                                                                   batch_size_list,
                                                                                                   gamma_range,
                                                                                                   scheduler_list)
        mlflow.log_params(trial.params)
        print(f'Trial params: {trial.params}')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlflow.log_param("device", device)

        if model_type == 'bert':
            train_loader, test_loader = tokenize_and_dataload_df('bert', batch_size, df_train)
            #             bert = BertModel.from_pretrained('bert-base-cased').to(device)
            model = BertForSequenceClassification.from_pretrained('bert-base-cased',
                                                                  num_labels=2,
                                                                  output_attentions=False,
                                                                  output_hidden_states=False).to(device)
        elif model_type == 'distilbert':
            train_loader, test_loader = tokenize_and_dataload_df('distilbert', batch_size, df_train)
            model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased',
                                                                        num_labels=2,
                                                                        output_attentions=False,
                                                                        output_hidden_states=False).to(device)
        elif model_type == 'roberta':
            model = RobertaForSequenceClassification.from_pretrained('roberta-base-cased',
                                                                     num_labels=2,
                                                                     output_attentions=False,
                                                                     output_hidden_states=False).to(device)
            train_loader, test_loader = tokenize_and_dataload_df('roberta', batch_size, df_train)

        elif model_type == 'xlnet':
            model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased',
                                                                     num_labels=2,
                                                                     output_attentions=False,
                                                                     output_hidden_states=False).to(device)
            train_loader, test_loader = tokenize_and_dataload_df('xlnet', batch_size, df_train)


        else:
            model = AlbertForSequenceClassification.from_pretrained('albert-base-v2',
                                                                    num_labels=2,
                                                                    output_attentions=False,
                                                                    output_hidden_states=False).to(device)
            train_loader, test_loader = tokenize_and_dataload_df('albert', batch_size, df_train)

        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        if optimizer_name == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=lr)

        if sched == 'step':
            scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
        else:
            scheduler = ExponentialLR(optimizer, gamma=gamma)

        for epoch in range(epochs):
            avg_train_loss, train_accuracy = train(model, device, train_loader, optimizer, epoch)
            avg_val_loss, val_accuracy = validate(model, device, test_loader)

            if avg_val_loss <= best_val_loss:
                best_val_loss = avg_val_loss

            mlflow.log_metric("avg_train_losses", avg_train_loss, step=epoch)
            mlflow.log_metric("avg_val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            scheduler.step()

            if val_accuracy > config.global_val_accuracy:
                print(f'Saving model. Best validation accuracy: {val_accuracy}')
                save_model(model, model_type, trial.params, epoch, val_accuracy)
                config.global_val_accuracy = val_accuracy

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return val_accuracy
