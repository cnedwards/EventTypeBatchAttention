
import numpy as np

import os
from os import walk
from bs4 import BeautifulSoup
import xmltodict
import pickle as pkl
import random

import pandas as pd

from collections import OrderedDict, defaultdict


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from transformers import AutoModel, AutoTokenizer

from sklearn.preprocessing import MultiLabelBinarizer



def to_dict(txt):

    def helper_to_dict(tag):
        rv = {}
        tmp = tag.findChildren(recursive=False)
        
        num = len(tmp) 
        
        if num:
            for ct in tmp:
                if ct.name not in rv: #first instance of that key
                    rv[ct.name] = helper_to_dict(ct)
                else: #if there are multiple of the same key, create a list
                    if type(rv[ct.name]) is not list: rv[ct.name] = [rv[ct.name]]

                    rv[ct.name].append(helper_to_dict(ct))
        else:
            return tag.text.replace('\n', ' ').strip()
        
        return rv

    soup = BeautifulSoup(txt, features="lxml")
    
    
    rv = {}
    
    for tag in soup.find_all(recursive=False):
        
        rv[tag.name] = helper_to_dict(tag)
        
    
    return rv['html']['body']

def load_sgm(file):
        
    
    with open(file,'r', encoding='utf-8') as txt:
           txt = txt.read()
    
    sgm = to_dict(txt)
    
    return sgm
    
    
def load_apf(file):
        
    def string_clean(txt):
        txt = txt.replace('\n', ' ')
        return txt
        
    #not thouroughly tested
    def clean_apf(apf): #clean text fields (e.g. remove \n)
        apf_clean = apf.copy()
        for k,val in apf.items(): #enumerate dict
            made_list = False
            if not isinstance(val, list): 
                val = [val]
                made_list = True
            for i, v in enumerate(val):
                if isinstance(v, dict) or isinstance(v, OrderedDict):
                    if made_list: 
                        apf_clean[k] = clean_apf(v)
                    else: apf_clean[k][i] = clean_apf(v)
                else:
                    if made_list: apf_clean[k] = string_clean(v)
                    else:
                        apf_clean[k][i] = string_clean(v)
        return apf_clean
                
    with open(file,'r', encoding='utf-8') as txt:
           txt = txt.read()
           
    #soup = BeautifulSoup(txt,'xml')

    apf = xmltodict.parse(txt)
    #print(apf)
    apf = clean_apf(apf)
    
    return apf


def get_events(source = "../../../../ACE05EN/source"):
    file_names = os.listdir(source)


    text = {}
    relations = {}
    events = {}
    events_list = []

    for i, f in enumerate(file_names):
        p = os.path.join(source, f)
        if f.endswith(".sgm"):
            sgm = load_sgm(p)
            
            text[sgm['doc']['docid']] = sgm
        elif f.endswith(".apf.xml"):
            apf = load_apf(p)
            
            if 'event' not in apf['source_file']['document']: continue #some docs don't have events :(
            
            evs = apf['source_file']['document']['event']

            docid = apf['source_file']['document']['@DOCID']
            
            if type(evs) != list:
                evs = [evs]
                
            for e in evs:
                EID = e['@ID']
                events[EID] = e

            events_list.extend(evs)

        if (i+1) % 100 == 0: print(i+1)

    return events, events_list
    
def get_roles_from_mention(mention):
    role_list = set()
    
    if 'event_mention_argument' in mention:
        if isinstance(mention['event_mention_argument'], list):
            for b in mention['event_mention_argument']:
                role_list.add(b['@ROLE'])
        else:
            b = mention['event_mention_argument']
            role_list.add(b['@ROLE'])

    return role_list
    
def get_data(source = "../../../ACE05EN/source"):
    events, events_list = get_events(source)
        
    event_types = set([e['@TYPE'] for e in events_list])
    event_subtypes = set([e['@SUBTYPE'] for e in events_list])

    role_list = set()
    event_subtype_dict = dict()
    event_role_dict = dict()

    for e in events_list:
        EID = e['@ID']
        event_subtype_dict[EID] = e['@SUBTYPE']
        event_role_dict[EID] = []
        tmp = e['event_mention']
        if not isinstance(e['event_mention'], list): tmp = [tmp]
        for a in tmp:
            if 'event_mention_argument' in a:
                if isinstance(a['event_mention_argument'], list):
                    for b in a['event_mention_argument']:
                        role_list.add(b['@ROLE'])
                        event_role_dict[EID].append(b['@ROLE'])
                else:
                    b = a['event_mention_argument']
                    role_list.add(b['@ROLE'])
                    event_role_dict[EID].append(b['@ROLE'])
        
    return events, events_list, role_list, event_subtype_dict, event_role_dict, event_subtypes


class ZS_ACE2005EventDataset_allmentions(Dataset):
    def __init__(self, subtype_representations, data, pretrained_model = "bert-base-uncased", k = 10):
        #following huang
        order = ['Attack', 'Transport', 'Die', 'Meet', 'Arrest-Jail', 'Sentence', 'Transfer-Money', 'Elect', 'Transfer-Ownership', 'End-Position']
        others = ['Phone-Write', 'Start-Position', 'Divorce', 'Sue', 'Demonstrate', 'Appeal', 'Marry', 'Start-Org', 'Acquit', 'Nominate', 'Trial-Hearing', 'Convict', 'Be-Born', 'Extradite', 'Execute', 'Pardon', 'Charge-Indict', 'Merge-Org', 'Fine', 'Declare-Bankruptcy', 'Injure', 'Release-Parole', 'End-Org']
        
        self.seen_types = order[:k]
        self.unseen_types = others + order[k:]

        self.events, self.events_list, self.role_list, self.event_subtype_dict, self.event_role_dict, self.event_subtypes = data
        self.pretrained_model = pretrained_model

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)

        self.sohe = MultiLabelBinarizer() #subtype one hot encoder
        self.rohe = MultiLabelBinarizer() #roles one hot encoder
        
        self.sohe.fit([list(self.event_subtypes)])
        self.rohe.fit([list(self.role_list)])
        
        self.subtype_representations = subtype_representations
            
        self.subtypes_roles = defaultdict(set) #all possible roles for a subtype (this is a star graph essentially)
        self.mentions_list = []
        self.mentions = {}
        self.mSubTypes = {}
        self.mEIDs = []
        self.mEID_to_EID = {}
        self.mRoles = {} #mention roles
        idx_count = 0
        for e in self.events_list:
            EID = e['@ID']
            subtype = e['@SUBTYPE']
            if isinstance(self.event_role_dict[EID], list):
                self.subtypes_roles[subtype].update(self.event_role_dict[EID])
            else:
                self.subtypes_roles[subtype].add(self.event_role_dict[EID])

            if isinstance(e['event_mention'], list):
                for mention in e['event_mention']:
                    mEID = mention['@ID']
                    self.mentions_list.append(mention)
                    self.mentions[mEID] = mention
                    self.mEIDs.append(mEID)
                    self.mSubTypes[mEID] = subtype
                    self.mEID_to_EID[mEID] = EID
                    self.mRoles[mEID] = get_roles_from_mention(mention)
                    idx_count += 1
            else: 
                mention = e['event_mention']
                mEID = mention['@ID']
                self.mentions_list.append(mention)
                self.mentions[mEID] = mention
                self.mEIDs.append(mEID)
                self.mSubTypes[mEID] = subtype
                self.mEID_to_EID[mEID] = EID
                self.mRoles[mEID] = get_roles_from_mention(mention)
                idx_count += 1

    def __len__(self):
        return len(self.mentions_list)

    def __getitem__(self, idx):

        mEID = self.mEIDs[idx]
        
        mention = self.mentions[mEID]

        text = mention['ldc_scope']['charseq']['#text']

        text_input = self.tokenizer(text, truncation=True, 
                                        padding='max_length', return_tensors = 'pt')

        subtype = self.mSubTypes[mEID]

        seen = subtype in self.seen_types

        subtype_oh = self.sohe.transform([[subtype]]).squeeze() * seen
        subtype_repres = self.subtype_representations[subtype] * seen
        
        roles_oh = self.rohe.transform([self.mRoles[mEID]]).squeeze()
        all_roles_oh = self.rohe.transform([list(self.subtypes_roles[subtype])]).squeeze()
        labels = (subtype_repres, subtype_oh, roles_oh, seen, mEID, subtype, all_roles_oh)
        
        return text_input, labels, text


def load_zeroshot_data_allmentions(k=10, source = "../../../ACE05EN/source", pretrained_model = "bert-base-uncased"):
    data = get_data(source)

    #get representations for type prediction
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    text_model = AutoModel.from_pretrained(pretrained_model) 

    subtype_representations = {}
    for est in data[5]: #event_subtypes
        input = tokenizer(est, return_tensors='pt')
        num_tokens = len(input['input_ids'])
        output = text_model(**input)
        
        repres = output['last_hidden_state'][:, 1:-1, :].squeeze(0)
        repres = repres.mean(axis=0)

        subtype_representations[est] = repres.detach()

    return ZS_ACE2005EventDataset_allmentions(subtype_representations, data, k=k)


class ZS_ACE2005EventDataset_BT_allmentions(Dataset):
    def __init__(self, subtype_representations, data, pretrained_model = "bert-base-uncased", k = 10, 
            bt_path = "backtranslations/", SBERT_embedding_file="SBERT_embeddings_allmentions.pkl", bt_languages = ['german', 'french', 'spanish', 'chinese']):
        #following huang
        order = ['Attack', 'Transport', 'Die', 'Meet', 'Arrest-Jail', 'Sentence', 'Transfer-Money', 'Elect', 'Transfer-Ownership', 'End-Position']
        others = ['Phone-Write', 'Start-Position', 'Divorce', 'Sue', 'Demonstrate', 'Appeal', 'Marry', 'Start-Org', 'Acquit', 'Nominate', 'Trial-Hearing', 'Convict', 'Be-Born', 'Extradite', 'Execute', 'Pardon', 'Charge-Indict', 'Merge-Org', 'Fine', 'Declare-Bankruptcy', 'Injure', 'Release-Parole', 'End-Org']
        
        self.seen_types = order[:k]
        self.unseen_types = others + order[k:]

        self.events, self.events_list, self.role_list, self.event_subtype_dict, self.event_role_dict, self.event_subtypes = data
        self.pretrained_model = pretrained_model

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)

        self.sohe = MultiLabelBinarizer() #subtype one hot encoder
        self.rohe = MultiLabelBinarizer() #roles one hot encoder
        
        self.sohe.fit([list(self.event_subtypes)])
        self.rohe.fit([list(self.role_list)])
        
        self.subtype_representations = subtype_representations
            
        self.subtypes_roles = defaultdict(set) #all possible roles for a subtype (this is a star graph essentially)
        self.mentions_list = []
        self.mentions = {}
        self.mSubTypes = {}
        self.mEIDs = []
        self.mEID_to_EID = {}
        self.mRoles = {} #mention roles
        idx_count = 0
        for e in self.events_list:
            EID = e['@ID']
            subtype = e['@SUBTYPE']
            if isinstance(self.event_role_dict[EID], list):
                self.subtypes_roles[subtype].update(self.event_role_dict[EID])
            else:
                self.subtypes_roles[subtype].add(self.event_role_dict[EID])

            if isinstance(e['event_mention'], list):
                for mention in e['event_mention']:
                    mEID = mention['@ID']
                    self.mentions_list.append(mention)
                    self.mentions[mEID] = mention
                    self.mEIDs.append(mEID)
                    self.mSubTypes[mEID] = subtype
                    self.mEID_to_EID[mEID] = EID
                    self.mRoles[mEID] = get_roles_from_mention(mention)
                    idx_count += 1
            else: 
                mention = e['event_mention']
                mEID = mention['@ID']
                self.mentions_list.append(mention)
                self.mentions[mEID] = mention
                self.mEIDs.append(mEID)
                self.mSubTypes[mEID] = subtype
                self.mEID_to_EID[mEID] = EID
                self.mRoles[mEID] = get_roles_from_mention(mention)
                idx_count += 1

        self.bt_path = bt_path
        self.bt_languages = bt_languages
        self.backtranslations = {}
        
        for language in bt_languages:
            with open(self.bt_path + language +'_backtranslations.allmentions.pkl', 'rb') as f:
                self.backtranslations[language] = pkl.load(f)

        #load original SBERT embeddings:
        
        with open(SBERT_embedding_file, 'rb') as f:
            self.SBERT_orig_embeddings = pkl.load(f)


    def __len__(self):
        return len(self.mentions_list)

    def __getitem__(self, idx):

        mEID = self.mEIDs[idx]
        
        mention = self.mentions[mEID]

        text = mention['ldc_scope']['charseq']['#text']

        text_input = self.tokenizer(text, truncation=True, # max_length=self.text_trunc_length,
                                        padding='max_length', return_tensors = 'pt')

        subtype = self.mSubTypes[mEID]

        seen = subtype in self.seen_types

        subtype_oh = self.sohe.transform([[subtype]]).squeeze() * seen
        subtype_repres = self.subtype_representations[subtype] * seen

        roles_oh = self.rohe.transform([self.mRoles[mEID]]).squeeze()
        all_roles_oh = self.rohe.transform([list(self.subtypes_roles[subtype])]).squeeze()

        #randomly sample a backtranslation:
        l = random.choice(self.bt_languages)
        bt_text = self.backtranslations[l][mEID]
        bt_text_input = self.tokenizer(bt_text, truncation=True, padding='max_length', return_tensors = 'pt')

        orig_emb = self.SBERT_orig_embeddings[mEID]

        labels = (subtype_repres, subtype_oh, roles_oh, seen, mEID, subtype, all_roles_oh, orig_emb)
        
        return text_input, bt_text_input, labels, text, bt_text

def load_zeroshot_data_BT_allmentions(k=10, source = "../../../../ACE05EN/source", bt_path = "backtranslations/", pretrained_model = "bert-base-uncased", SBERT_embedding_file="SBERT_embeddings_allmentions.pkl"):
    data = get_data(source)

    #get representations for type prediction
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    text_model = AutoModel.from_pretrained(pretrained_model) 

    subtype_representations = {}
    for est in data[5]: #event_subtypes
        input = tokenizer(est, return_tensors='pt')
        num_tokens = len(input['input_ids'])
        
        output = text_model(**input)
        
        repres = output['last_hidden_state'][:, 1:-1, :].squeeze(0)
        repres = repres.mean(axis=0)

        subtype_representations[est] = repres.detach()

    return ZS_ACE2005EventDataset_BT_allmentions(subtype_representations, data, k=k, bt_path=bt_path, SBERT_embedding_file=SBERT_embedding_file)


#define Jaccard Similarity function - https://www.statology.org/jaccard-similarity-python/
def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union



#FrameNet version:

def load_zeroshot_data_BT_allmentions_FN(k=10, source = "../../../../ACE05EN/source", bt_path = "backtranslations/", pretrained_model = "bert-base-uncased", SBERT_embedding_file="SBERT_embeddings_allmentions.pkl"):
    data = get_data(source)


    #load fn examples and heirarchy:
    fn_examples = pd.read_csv('framenet/fn_examples.csv')
    fn_definitions = pd.read_csv('framenet/fn_definitions.csv')


    sentences = list(fn_definitions['Example'])
    frames = list(fn_definitions['Frame'])


    mapping_df = pd.read_csv('framenet/ace_manual_map.txt', delimiter=",")

    mapping = {}

    for row in mapping_df.iterrows():
        mapping[row[1]['ACE']] = row[1]['Frame'].split('|')


    #get representations for type prediction
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    text_model = AutoModel.from_pretrained(pretrained_model) 

    subtype_representations = {}
    for est in data[5]: #event_subtypes
        fn_def = sentences[frames.index(mapping[est][0])]
        input = tokenizer(fn_def, return_tensors='pt')
        num_tokens = len(input['input_ids'])
        output = text_model(**input)
        
        repres = output['last_hidden_state'][:, 1:-1, :].squeeze(0)
        repres = repres.mean(axis=0)

        subtype_representations[est] = repres.detach()


    return ZS_ACE2005EventDataset_BT_allmentions(subtype_representations, data, k=k, bt_path=bt_path, SBERT_embedding_file=SBERT_embedding_file)

