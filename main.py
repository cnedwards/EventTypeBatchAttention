
import json
import os
import numpy as np
import time
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

import torch.optim as optim
from transformers.optimization import get_constant_schedule_with_warmup


import matplotlib.pyplot as plt

#import wandb

from dataloader import load_zeroshot_data_BT_allmentions
from networks import SentenceModel
from losses import type_contrastive_sim_ZS_mask_margin_loss, MSE_Margin, COS_Margin

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Parameters
BATCH_SIZE = 10
SAVE_PATH = 'model_output/'

MODEL_NAME = 'sentence-transformers/all-MiniLM-L12-v2'

DROPOUT = 0.1

N_CLUSTERS = [23, 50, 100, 300, 500]

MARGIN = 0.5 #this is used in the contrastive loss
#constrain_margin = 2 #an artifact which is used for logging

epochs = 10

init_lr = 1e-4
bert_lr = 2e-5

if not os.path.exists(SAVE_PATH):
  os.mkdir(SAVE_PATH)


dataloader_params = {'batch_size': BATCH_SIZE,
          'shuffle': True,
          'num_workers': 0}


dataset = load_zeroshot_data_BT_allmentions(source="../../../ACE05EN/source", bt_path = "backtranslations/", pretrained_model=MODEL_NAME, SBERT_embedding_file="SBERT_embeddings_allmentions.pkl")

train_generator = DataLoader(dataset, **dataloader_params)

model = SentenceModel(BATCH_SIZE, pretrained_model=MODEL_NAME, dropout=DROPOUT)

bert_params = list(model.text_transformer_model.parameters())

optimizer = optim.Adam([
                {'params': model.other_params},
                {'params': bert_params, 'lr': bert_lr}
            ], lr=init_lr)

num_warmup_steps = 1000
num_training_steps = epochs * len(train_generator) - num_warmup_steps
scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps) 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

tmp = model.to(device)
train_losses = []
val_losses = []

train_acc = []
val_acc = []

'''
wandb.init(
      # Set entity to specify your username or team name
      entity="",
      # Set the project where this run will be logged
      project="", 
      # Track hyperparameters and run metadata
      config={
      "learning_rate": {'model':init_lr, 'bert':bert_lr},
      "architecture": "clusterer",
      "dataset": "ACE2005",
      "warmup_steps" : num_warmup_steps,
      "batch_size" : BATCH_SIZE,
      "epochs" : epochs,
      "pretrained_model" : MODEL_NAME,
      "save_path" : SAVE_PATH,
      "dropout" : DROPOUT,
      "N_CLUSTERS" : N_CLUSTERS,
      "margin" : MARGIN,
      })
'''

COS = nn.CosineSimilarity()

# Loop over epochs
for epoch in range(epochs):

    epoch_queries = []
    epoch_keys = []
    epoch_CLS = []
    epoch_labels = []

    silhouette_scores = []
    ch_scores = []
    db_scores = []

    nmi_scores = []
    fm_scores = []
    ari_scores = []

    # Training
    
    start_time = time.time()
    running_losses = np.zeros((5))
    running_loss = 0.0
    running_MSE_all_constraint = 0.0
    running_MSE_std_all_constraint = 0.0
    running_acc = 0.0
    model.train()
    for i, d in enumerate(train_generator):
        text, bt_text, labels, raw_text, raw_bt_text = d
        # Transfer to GPU
        
        dim = text['attention_mask'].shape[-1]
        mask = text['attention_mask'].reshape(-1,dim).to(device)
        input_ids = text['input_ids'].reshape(-1,dim).to(device)
        
        dim = bt_text['attention_mask'].shape[-1]
        bt_mask = bt_text['attention_mask'].reshape(-1,dim).to(device)
        bt_input_ids = bt_text['input_ids'].reshape(-1,dim).to(device)

        subtype_repres, subtype_oh, roles_oh, seen, EID, subtypes, all_roles_oh, SBERT_orig = labels
        subtype_repres = subtype_repres.to(device)
        subtype_oh = subtype_oh.float().to(device) #"addmm_cuda" not implemented for 'Long'
        roles_oh = roles_oh.float().to(device)
        seen = seen.to(device)
        SBERT_orig = SBERT_orig.to(device)

        clust_x, keys, query, S, text_x, trigger = model(input_ids, mask)
        
        bt_clust_x, bt_keys, bt_query, bt_S, bt_text_x, bt_trigger = model(bt_input_ids, bt_mask)
        
        tcon_loss = type_contrastive_sim_ZS_mask_margin_loss(torch.sigmoid(S), subtype_oh, seen, margin=MARGIN).to(device)
        bt_tcon_loss = type_contrastive_sim_ZS_mask_margin_loss(torch.sigmoid(bt_S), subtype_oh, seen, margin=MARGIN).to(device)

        left_cross_tcon_loss = type_contrastive_sim_ZS_mask_margin_loss(torch.sigmoid(torch.matmul(query, bt_keys.T)/torch.sqrt(model.clusterer.f)), subtype_oh, seen, margin=MARGIN).to(device)
        right_cross_tcon_loss = type_contrastive_sim_ZS_mask_margin_loss(torch.sigmoid(torch.matmul(bt_query, keys.T)/torch.sqrt(model.clusterer.f)), subtype_oh, seen, margin=MARGIN).to(device)

        con_losses = tcon_loss + bt_tcon_loss + left_cross_tcon_loss + right_cross_tcon_loss
        loss = con_losses

        running_loss += loss.item()
        running_losses[0] += tcon_loss.item()
        running_losses[1] += bt_tcon_loss.item()
        running_losses[3] += left_cross_tcon_loss.item()
        running_losses[4] += right_cross_tcon_loss.item()

        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        
        scheduler.step()

        #wandb.log({'batch': i, 'loss': loss.item(), "contrastive losses": con_losses.item(), })
            
        if (i+1) % 100 == 0: print(i+1, "batches trained. Avg loss:\t", running_losses / (i+1), ". Avg ms/step =", 1000*(time.time()-start_time)/(i+1))
    train_losses.append(running_losses / (i+1))
    train_acc.append(running_acc / (i+1))

    print("Epoch", epoch, "training loss:\t\t", running_losses / (i+1), ". Time =", (time.time()-start_time), "seconds.")
    

    start_time = time.time()
    running_losses = np.zeros((5))
    running_loss = 0.0
    running_acc = 0.0
    model.eval()
    with torch.set_grad_enabled(False):
        for i, d in enumerate(train_generator):
            text, bt_text, labels, raw_text, raw_bt_text = d
            # Transfer to GPU
            
            dim = text['attention_mask'].shape[-1]
            mask = text['attention_mask'].reshape(-1,dim).to(device)
            input_ids = text['input_ids'].reshape(-1,dim).to(device)
            
            dim = bt_text['attention_mask'].shape[-1]
            bt_mask = bt_text['attention_mask'].reshape(-1,dim).to(device)
            bt_input_ids = bt_text['input_ids'].reshape(-1,dim).to(device)

            subtype_repres, subtype_oh, roles_oh, seen, EID, subtypes, all_roles_oh, SBERT_orig = labels
            subtype_repres = subtype_repres.to(device)
            subtype_oh = subtype_oh.float().to(device) #"addmm_cuda" not implemented for 'Long'
            roles_oh = roles_oh.float().to(device)
            seen = seen.to(device)
            SBERT_orig = SBERT_orig.to(device)

            clust_x, keys, query, S, text_x, trigger = model(input_ids, mask)
            
            bt_clust_x, bt_keys, bt_query, bt_S, bt_text_x, bt_trigger = model(bt_input_ids, bt_mask)

            epoch_queries.extend([q.cpu().detach().numpy() for q, s in zip(query, seen) if not s])
            epoch_keys.extend([k.cpu().detach().numpy() for k, s in zip(keys, seen) if not s])
            epoch_CLS.extend([c.cpu().detach().numpy() for c, s in zip(text_x, seen) if not s])
            epoch_labels.extend([st for st, s in zip(subtypes, seen) if not s])

            #cos_dist = 1 - COS(text_x, SBERT_orig)
            #mean_cos_distance, cos_std = cos_dist.mean(), cos_dist.std()
            #constrain_SBERT_loss, mean_cos_distance, cos_std = COS_Margin(text_x, SBERT_orig, constrain_margin)

            #wandb.log({'eval batch': i, 
            #    'COS Constraint All':mean_cos_distance.item(), 'COS Constraint Std':cos_std.item()})
        
            
    val_losses.append(running_losses / (i+1))
    val_acc.append(running_acc / (i+1))

    #clustering to test
    epoch_keys = np.array(epoch_keys)
    epoch_queries = np.array(epoch_queries)
    epoch_CLS = np.array(epoch_CLS)
    epoch_labels = np.array(epoch_labels)
    
    S = np.matmul(epoch_queries, epoch_keys.T)
    
    tmp_silos = []
    tmp_silos_mse = []
    for n in N_CLUSTERS:
        clustering = AgglomerativeClustering(n_clusters=n, affinity='precomputed', linkage='average').fit(1-S)
        
        pred = clustering.labels_
        silo = silhouette_score(epoch_CLS, pred, metric='cosine')
        tmp_silos.append(silo)
        silo_mse = silhouette_score(epoch_CLS, pred)
        tmp_silos_mse.append(silo_mse)
    silhouette_scores.append(tmp_silos)
    
    log = {'epoch' : epoch,
        "tcon_loss":train_losses[-1][0], "bt_tcon_loss":train_losses[-1][1], 
        "left_cross_tcon_loss":train_losses[-1][3], "right_cross_tcon_loss":train_losses[-1][4], 
        }

    for indc, n in enumerate(N_CLUSTERS):
        log["silhouette_CLS_cosine " + str(n)] = tmp_silos[indc]
        log["silhouette_CLS_MSE " + str(n)] = tmp_silos_mse[indc]

    #wandb.log(log)
    
    tmp = np.array(val_losses)

    torch.save(model.state_dict(), SAVE_PATH +str(epoch)+".pt")

    print("Epoch", epoch, ". Time =", (time.time()-start_time), "seconds.")
    print(log)

    with open(SAVE_PATH + '{}.log'.format(epoch), 'w') as f:
        json.dump(str(log), f)

    sys.stdout.flush()


torch.save(model.state_dict(), SAVE_PATH + "final_weights." +str(val_losses[-1]) + '_' + str(epochs)+".pt")

#wandb.finish()


plt.plot(train_losses)
plt.legend(['Train Type Contrastive', 'Train Type Regression', 'Test Type Contrastive', 'Test Type Regression'])
plt.savefig(SAVE_PATH + 'loss.png')


