
import pickle
import numpy as np

from collections import defaultdict, Counter

import torch

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score

import matplotlib.pyplot as plt

import pandas as pd

import networkx as nx

import copy

from networks import SentenceModel

from transformers import AutoTokenizer

SBERT_FILE = "SBERT_embeddings_allmentions.pkl"


SAVE_PATH = 'model_output/' #where the outputs are saved
MODEL = '7.pt' #which checkpoint to use
cluster_file = SAVE_PATH + MODEL + '.23.agglo.UMAP_CLS.1300.clusters_downstream.txt' #which cluster file to use

#if this is true the ground truth clustering will be used instead
perfect_clusters = False


with open(SAVE_PATH + MODEL + ".CLS.npy", 'rb') as f:
    cls_rep = pickle.load(f)


with open(SAVE_PATH + MODEL + "eid_subtypes.npy", 'rb') as f:
    eid_subtypes = pickle.load(f)

MODEL_NAME = 'sentence-transformers/all-MiniLM-L12-v2'



seen = ['Attack', 'Transport', 'Die', 'Meet', 'Arrest-Jail', 'Sentence', 'Transfer-Money', 'Elect', 'Transfer-Ownership', 'End-Position']

unseen_mask = []

EIDs = []
subtypes = []
queries = []
embs = []

for id in eid_subtypes:
    EIDs.append(id)
    subtypes.append(eid_subtypes[id])
    unseen_mask.append(eid_subtypes[id] not in seen)
    embs.append(cls_rep[id])


EIDs = np.array(EIDs)
subtypes = np.array(subtypes)
unseen_mask = np.array(unseen_mask)

embs = np.array(embs)

EIDs = EIDs[unseen_mask]
embs = embs[unseen_mask]

subtypes = subtypes[unseen_mask]

subtype_names = list(np.unique(subtypes))

#load fn examples and heirarchy:
fn_examples = pd.read_csv('framenet/fn_examples.csv')
fn_definitions = pd.read_csv('framenet/fn_definitions.csv')


sentences = list(fn_definitions['Example'])
frames = list(fn_definitions['Frame'])


mapping_df = pd.read_csv('framenet/ace_manual_map.txt', delimiter=",")

mapping = {}

for row in mapping_df.iterrows():
    mapping[row[1]['ACE']] = row[1]['Frame'].split('|')


G = nx.readwrite.adjlist.read_adjlist("framenet/fn_heirarchy.adjlist", create_using=nx.DiGraph)



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def get_fn_sentence_embeddings(sentences):
    
    model = SentenceModel(MODEL_NAME)
    model.load_state_dict(torch.load(SAVE_PATH+MODEL))

    rv = []

    model.eval()
    with torch.set_grad_enabled(False):
        for i, name in enumerate(sentences):
            text = tokenizer(name, return_tensors='pt')
            
            dim = text['attention_mask'].shape[-1]
            mask = text['attention_mask'].reshape(-1,dim)
            input_ids = text['input_ids'].reshape(-1,dim)

            _, _, _, _, emb, _ = model(input_ids, mask)

            rv.append(emb.squeeze().numpy())

    return np.array(rv)


if 'fn_embeddings' not in locals():
    fn_embeddings = get_fn_sentence_embeddings(sentences)

num_subtypes = len(subtype_names)

def bijection_to_ints(strings):
    mapping = {}
    for st in strings:
        if st not in mapping:
            mapping[st] = len(mapping)
    return mapping

subtype_names_ints = bijection_to_ints(subtype_names)

#load clustering:

EID_to_cluster = {}
cluster_to_EID = defaultdict(list)

with open(cluster_file, 'r') as f:
    next(f)
    for line in f:
        if line.strip() == "": continue
        
        eid, subtype, clust = line.strip().split("\t")
        clust = int(clust)

        if perfect_clusters:
            clust = subtype_names_ints[subtype]

        EID_to_cluster[eid] = clust
        cluster_to_EID[clust].append(eid)


num_clusters = len(cluster_to_EID)


def build_relevancy_list(ACE_type):
    print(ACE_type)
    relevant_frames = set(mapping[ACE_type])

    #add_children_from_heirarchy
    tmp = copy.copy(relevant_frames)
    for t in tmp:
        print(t)
        relevant_frames.update(list(nx.descendants(G, t)))
        
    return relevant_frames, np.array([f in relevant_frames for f in frames], dtype=np.int64) #relevant here are 1 else 0



most_common_subtype_list = []

ranks = []
percents = []
cluster_info = []
ndcgs = []

real_pure_nom = 0
real_pure_denom = 0

for clust in range(num_clusters):

    mask = [EID_to_cluster[eid]==clust for eid in EIDs]

    num = np.sum(mask)

    clust_embs = embs[mask]
    clust_subtypes = subtypes[mask]

    clust_centroid = clust_embs.mean(axis=0).reshape((1,-1))

    unq, counts = np.unique(clust_subtypes, return_counts=True)
    #sort them...
    inds = np.argsort(counts)[::-1]
    unq, counts = unq[inds], counts[inds]

    most_common_subtype = unq[0]
    cluster_purity = counts[0] / num
    real_pure_nom += counts[0]
    real_pure_denom += num

    most_common_subtype_list.append(most_common_subtype)

    sims = cosine_similarity(clust_centroid, fn_embeddings).squeeze()
    ranking = np.argsort(sims)[::-1]
    ranking_frames = [frames[r] for r in ranking]

    relevant_frames, relevancy = build_relevancy_list(most_common_subtype)

    ranking_frames_unq = pd.unique(ranking_frames)
    num_frames = len(ranking_frames_unq)

    rank = 9999
    for f in relevant_frames:
        try:
            new_rank = list(ranking_frames_unq).index(f) + 1
        except: continue
        rank = min(rank, new_rank)

    ndcg = ndcg_score(relevancy.reshape((1,-1)), sims.reshape((1,-1)), k=None)

    ranks.append(rank)
    ndcgs.append(ndcg)

    percents.append(cluster_purity)
    cluster_info.append((clust, most_common_subtype, round(100*cluster_purity, 2), num, rank, ranking_frames_unq[:10]))
    
ranks = np.array(ranks)

perc = len(np.unique(most_common_subtype_list)) / len(subtype_names)

perc = len(np.unique(most_common_subtype_list)) / len(subtype_names)
print('Majority:', np.unique(most_common_subtype_list))
print('Not majority:', set(subtypes) - set(np.unique(most_common_subtype_list)))
print("Percent of subtypes represented: {}/{} = {:.3f}".format(len(np.unique(most_common_subtype_list)), len(subtype_names), perc))

print("Average cluster purity:", np.mean(percents))
print("Minimum purity:", np.min(percents))
print("Real Purity:", real_pure_nom/real_pure_denom)

plt.figure()
plt.hist(np.array(percents)*100)
plt.xlabel('Purity')
plt.ylabel('Count')
plt.xlim([0,100])
plt.savefig("figures/name_percent_histogram.png")

plt.figure()
plt.hist(ranks, bins=num_subtypes)
plt.xlabel('Rank')
plt.ylabel('Count')
plt.xlim([1,num_subtypes])
plt.xticks(np.arange(1,num_subtypes+1))
plt.savefig("figures/name_ranks_histogram.png")

print()
print("Mean rank:", np.mean(ranks), "/", num_frames)
print("Hits at 1:", np.mean(ranks <= 1))
print("Hits at 5:", np.mean(ranks <= 3))
print("Hits at 10:", np.mean(ranks <= 5))
print("Hits at 50:", np.mean(ranks <= 10))
print("Hits at 100:", np.mean(ranks <= 15))

print("MRR:", np.mean(1/ranks))
print("Mean nDCG", np.mean(ndcgs))
print()

for ci in cluster_info:
    print(ci)


