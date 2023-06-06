



import os
import numpy as np
import time
import sys
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from dataloader import load_zeroshot_data_allmentions
from networks import SentenceModel
from losses import type_contrastive_sim_ZS_mask_loss, type_representation_cos_ZS_loss

# Parameters
BATCH_SIZE = 10

SAVE_PATH = 'model_output/' #where the outputs are saved
MODEL = '7.pt' #which checkpoint to use


MODEL_NAME = 'sentence-transformers/all-MiniLM-L12-v2'

dataloader_params = {'batch_size': BATCH_SIZE,
          'shuffle': True,
          'num_workers': 0}


dataset = load_zeroshot_data_allmentions(pretrained_model=MODEL_NAME)

generator = DataLoader(dataset, **dataloader_params)

model = SentenceModel(BATCH_SIZE, pretrained_model=MODEL_NAME)
model.load_state_dict(torch.load(SAVE_PATH+MODEL))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

tmp = model.to(device)


queries = {}
keys = {}
cls_rep = {}
eid_subtypes = {}

start_time = time.time()
running_losses = np.array([0.0, 0.0])
running_loss = 0.0
running_acc = 0.0
model.eval()
with torch.set_grad_enabled(False):
    for i, d in enumerate(generator):
        text, labels, raw_text = d
        # Transfer to GPU
        
        dim = text['attention_mask'].shape[-1]
        mask = text['attention_mask'].reshape(-1,dim).to(device)
        input_ids = text['input_ids'].reshape(-1,dim).to(device)

        subtype_repres, subtype_oh, roles_oh, seen, EID, subtypes, all_roles_oh = labels
        subtype_repres = subtype_repres.to(device)
        subtype_oh = subtype_oh.float().to(device) #"addmm_cuda" not implemented for 'Long'
        roles_oh = roles_oh.float().to(device)
        seen = seen.to(device)

        clust_x, key, query, S, text_x, trigger = model(input_ids, mask)
        

        for q, k, id, st, c in zip(query, key, EID, subtypes, text_x):
            queries[id] = q.cpu().numpy()
            keys[id] = k.cpu().numpy()
            cls_rep[id] = c.cpu().numpy()

            eid_subtypes[id] = st

        tcon_loss = type_contrastive_sim_ZS_mask_loss(torch.sigmoid(S), subtype_oh, seen).to(device)
        trep_loss = type_representation_cos_ZS_loss(trigger, subtype_repres, seen).to(device)

        loss = tcon_loss + trep_loss
        #print(tcon_loss.item(), trep_loss.item())

        running_loss += loss.item()
        running_losses[0] += tcon_loss.item()
        running_losses[1] += trep_loss.item()
        
        if (i+1) % 50 == 0: print(i+1, "batches eval. Avg loss:\t", running_losses / (i+1), ". Avg ms/step =", 1000*(time.time()-start_time)/(i+1))

        

with open(SAVE_PATH + MODEL + ".query.npy", 'wb') as f:
    pickle.dump(queries, f)
with open(SAVE_PATH + MODEL + ".key.npy", 'wb') as f:
    pickle.dump(keys, f)

with open(SAVE_PATH + MODEL + ".CLS.npy", 'wb') as f:
    pickle.dump(cls_rep, f)


with open(SAVE_PATH + MODEL + "eid_subtypes.npy", 'wb') as f:
    pickle.dump(eid_subtypes, f)