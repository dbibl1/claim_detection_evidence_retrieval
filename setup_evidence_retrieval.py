import os
import argparse
import csv
import json
import logging
import pickle
import time
import glob
from pathlib import Path

import numpy as np
import torch
import transformers

import contriever.src.index
import contriever.src.contriever
import contriever.src.utils
import contriever.src.slurm
import contriever.src.data
from contriever.src.evaluation import calculate_matches
import contriever.src.normalize_text

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    return data

def setup():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = contriever.src.contriever.load_retriever('facebook/contriever')
    model.eval()
    model = model.cuda()
    index = contriever.src.index.Indexer(768, 0, 8)

    input_paths = glob.glob("contriever/contriever_embeddings/wikipedia_embeddings/*")
    input_paths = sorted(input_paths)
    embeddings_dir = os.path.dirname(input_paths[0])
    index_path = os.path.join(embeddings_dir, "index.faiss")
    index.deserialize_from(embeddings_dir)

    passages = contriever.src.data.load_passages('contriever/psgs_w100.tsv')
    passage_id_map = {x["id"]: x for x in passages}

    return model.to(device), tokenizer, index, passages, passage_id_map

