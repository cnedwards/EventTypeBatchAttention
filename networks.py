
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, AutoModel



class Clusterer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, batch_size, num_features, dropout = 0.1):
        super().__init__()
        self.n = batch_size
        self.f = num_features
        self.drop = nn.Dropout(p=dropout)

        self.key_hidden1 = nn.Linear(self.f, self.f)
        self.key_hidden2 = nn.Linear(self.f, self.f)
        self.key_hidden3 = nn.Linear(self.f, self.f)
        self.ln = nn.LayerNorm((self.f))
        
        self.query_hidden1 = nn.Linear(self.f, self.f)
        self.query_hidden2 = nn.Linear(self.f, self.f)
        self.query_hidden3 = nn.Linear(self.f, self.f)
        self.lnq = nn.LayerNorm((self.f))

        self.temp = nn.Parameter(torch.Tensor([0.3]))

        self.f = torch.tensor(self.f)
        
        self.relu = nn.ReLU()

    def forward(self, F):
        keys = self.relu(self.key_hidden1(F))
        keys = self.drop(keys)
        keys = self.relu(self.key_hidden2(keys))
        keys = self.ln(keys)
        keys = self.drop(keys)
        keys = self.key_hidden3(keys)
        
        query = self.relu(self.query_hidden1(F))
        query = self.drop(query)
        query = self.relu(self.query_hidden2(query))
        query = self.ln(query)
        query = self.drop(query)
        query = self.query_hidden3(query)

        S = torch.matmul(query, keys.T) / torch.sqrt(self.f)#scaling term

        sims = torch.softmax(S, 1)

        F_hat = torch.matmul(sims, F) 

        return F_hat, S, keys, query


class SentenceModel(nn.Module):
    def __init__(self, batch_size, hidden =384, dropout=0.1, pretrained_model = "sentence-transformers/all-MiniLM-L12-v2"):
        super(SentenceModel, self).__init__()
        
        self.pretrained_model = pretrained_model
        self.hidden = hidden
        self.batch_size = batch_size
        self.drop = nn.Dropout(p=dropout)


        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        
        #layers:
        
        self.trigger_hidden1 = nn.Linear(hidden, hidden)
        self.trigger_hidden2 = nn.Linear(hidden, hidden)

        self.cls1 = nn.Linear(hidden, hidden)

        self.clusterer = Clusterer(batch_size, hidden, dropout=dropout)

        self.other_params = list(self.parameters()) #get all but bert params
        
        self.text_transformer_model = AutoModel.from_pretrained(pretrained_model)
        self.text_transformer_model.train()

    #See https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, text, text_mask = None):
      
        text_encoder_output = self.text_transformer_model(text, attention_mask = text_mask)

        text_x = self.mean_pooling(text_encoder_output, text_mask)
        text_x = F.normalize(text_x, p=2, dim=1)

        clust_x, S, keys, query = self.clusterer(text_x)

        trigger = self.relu(self.trigger_hidden1(clust_x)) #this probably shouldn't be called trigger. Would be better called "name" or "subtype_pred"
        trigger = self.trigger_hidden2(trigger)

        return clust_x, keys, query, S, text_x, trigger





class RoleModel(nn.Module):
    def __init__(self, batch_size, role_num, hidden =768, dropout=0.5, pretrained_model = "bert-base-uncased"):
        super(RoleModel, self).__init__()
        
        self.pretrained_model = pretrained_model
        self.hidden = hidden
        self.batch_size = batch_size
        self.drop = nn.Dropout(p=dropout)
        self.role_num = role_num


        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        
        #layers:
        
        self.trigger_hidden1 = nn.Linear(hidden, hidden)
        self.trigger_hidden2 = nn.Linear(hidden, hidden)

        self.cls1 = nn.Linear(hidden, hidden)
        self.cls2 = nn.Linear(hidden, hidden)

        self.clusterer = Clusterer(batch_size, hidden)
        #self.clusterer2 = Clusterer(batch_size, hidden)

        self.individual_role_hidden1 = nn.Linear(hidden, hidden) #predict roles before clusterer
        self.individual_role_hidden2 = nn.Linear(hidden, hidden)
        self.individual_role = nn.Linear(hidden, role_num)
        self.type_role_hidden1 = nn.Linear(hidden, hidden) #predict roles after clusterer
        self.type_role_hidden2 = nn.Linear(hidden, hidden)
        self.type_role = nn.Linear(hidden, role_num)

        self.other_params = list(self.parameters()) #get all but bert params
        
        self.text_transformer_model = AutoModel.from_pretrained(pretrained_model)
        self.text_transformer_model.train()

    def forward(self, text, text_mask = None):
      
        text_encoder_output = self.text_transformer_model(text, attention_mask = text_mask)

        text_x = text_encoder_output['pooler_output']
        text_x = self.relu(self.cls1(text_x))
        text_x = self.cls2(text_x)

        indiv = self.relu(self.individual_role_hidden1(text_x))
        indiv = self.relu(self.individual_role_hidden2(indiv))
        indiv = self.individual_role(indiv)

        clust_x, S, keys, query = self.clusterer(text_x)
        #clust_x, S, keys, query = self.clusterer2(clust_x)


        type_role = self.relu(self.type_role_hidden1(clust_x))
        type_role = self.relu(self.type_role_hidden2(type_role))
        type_role = self.type_role(type_role)

        trigger = self.relu(self.trigger_hidden1(clust_x))
        trigger = self.trigger_hidden2(trigger)

        return clust_x, keys, query, S, text_x, trigger, indiv, type_role



