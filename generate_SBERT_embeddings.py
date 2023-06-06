
import numpy as np
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer

from tqdm import tqdm

from dataloader import load_zeroshot_data_allmentions

MODEL_NAME = 'sentence-transformers/all-MiniLM-L12-v2'

OUTPUT_FILE = "SBERT_embeddings_allmentions.pkl"

BATCH_SIZE = 16

model = SentenceTransformer(MODEL_NAME)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

tmp = model.to(device)

dataset = load_zeroshot_data_allmentions()

dataloader_params = {'batch_size': BATCH_SIZE,
          'shuffle': False,
          'num_workers': 0}

generator = DataLoader(dataset, **dataloader_params)


SBERT_embeddings = {}

model.eval()
with torch.set_grad_enabled(False):
    for d in tqdm(generator):
        text, labels, raw_text = d

        subtype_repres, subtype_oh, roles_oh, seen, EID, subtypes, all_roles_oh = labels

        embeddings = model.encode(raw_text)

        for e, emb in zip(EID, embeddings):
            SBERT_embeddings[e] = emb


with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(SBERT_embeddings, f)

