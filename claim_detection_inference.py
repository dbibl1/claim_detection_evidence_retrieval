import pandas as pd
import torch
from transformers import BertForSequenceClassification


def load_model(device, path):
    model = BertForSequenceClassification.from_pretrained('bert-base-cased',
                                                                 num_labels = 2,
                                                                 output_attentions = False,
                                                                 output_hidden_states = False).to(device)
    model = torch.load(path)
    model.eval()
    return model

def predict(input_ids, model):
    out = model(input_ids)
    pred = out.logits.argmax(dim=1, keepdim=True)
    return pred.tolist()

def get_checkworthy(input_ids, model, queries):
    predictions = predict(input_ids, model)
    checkworthy = []
    not_checkworthy = []
    for i, e in enumerate(predictions):
        if e[0] == 1:
            checkworthy.append(queries[i])
        else:
            not_checkworthy.append(queries[i])
    return checkworthy, not_checkworthy

def output_not_checkworthy(not_checkworthy):
    out = ''
    if not_checkworthy:
        for e in not_checkworthy:
            out += f'Claim: \"{e}\" is not check-worthy \n'
    return out

