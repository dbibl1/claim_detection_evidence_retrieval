from transformers import DistilBertForSequenceClassification
from claim_detection_inference import load_model
import torch
from transformers import AutoTokenizer

def load_model(device, path):
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                                 num_labels = 2,
                                                                 output_attentions = False,
                                                                 output_hidden_states = False).to(device)
    model = torch.load(path)
    model.eval()
    return model

def setup():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = load_model(device, path='models/distilbert.pth')
    return model, device, tokenizer