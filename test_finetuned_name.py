
import pickle
import numpy as np

from collections import defaultdict, Counter

import torch

from networks import SentenceModel

from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt


from transformers import AutoTokenizer

MODEL_NAME = 'sentence-transformers/all-MiniLM-L12-v2'

SAVE_PATH = 'model_output/' #where the outputs are saved
MODEL = '7.pt' #which checkpoint to use
cluster_file = SAVE_PATH + MODEL + '.23.agglo.UMAP_CLS.1300.clusters_downstream.txt' #which cluster file to use

#if this is true the ground truth clustering will be used instead
perfect_clusters = False

with open(SAVE_PATH + MODEL + ".query.npy", 'rb') as f:
    queries_dict = pickle.load(f)
with open(SAVE_PATH + MODEL + ".key.npy", 'rb') as f:
    keys_dict = pickle.load(f)


with open(SAVE_PATH + MODEL + ".CLS.npy", 'rb') as f:
    cls_rep = pickle.load(f)


with open(SAVE_PATH + MODEL + "eid_subtypes.npy", 'rb') as f:
    eid_subtypes = pickle.load(f)

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


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def get_subtype_names_embeddings(names):
    
    model = SentenceModel(MODEL_NAME)
    model.load_state_dict(torch.load(SAVE_PATH+MODEL))

    rv = []

    model.eval()
    with torch.set_grad_enabled(False):
        for name in names:
            text = tokenizer(name, return_tensors='pt')
            
            dim = text['attention_mask'].shape[-1]
            mask = text['attention_mask'].reshape(-1,dim)
            input_ids = text['input_ids'].reshape(-1,dim)

            _, _, _, _, emb, _ = model(input_ids, mask)

            rv.append(emb.squeeze().numpy())

    return np.array(rv)

subtype_name_embeddings = get_subtype_names_embeddings(subtype_names)

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


most_common_subtype_list = []

ranks = []
percents = []
cluster_info = []

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
    most_common_subtype_ind = subtype_names.index(most_common_subtype)

    most_common_subtype_list.append(most_common_subtype)

    sims = cosine_similarity(clust_centroid, subtype_name_embeddings).squeeze()
    ranking = np.argsort(sims)[::-1]
    ranking_names = [subtype_names[r] for r in ranking]

    rank = list(ranking).index(most_common_subtype_ind) + 1

    ranks.append(rank)

    percents.append(cluster_purity)
    cluster_info.append((clust, most_common_subtype, round(100*cluster_purity, 2), num, rank, ranking_names[:10]))

ranks = np.array(ranks)

perc = len(np.unique(most_common_subtype_list)) / len(subtype_names)
print("Percent of subtypes represented: {}/{} = {:.3f}".format(len(np.unique(most_common_subtype_list)), len(subtype_names), perc))
print("Average cluster purity:", np.mean(percents))
print("Minimum purity:", np.min(percents))

plt.figure()
plt.hist(np.array(percents)*100)
plt.xlabel('Purity')
plt.ylabel('Count')
plt.xlim([0,100])
plt.savefig("figures/percent_histogram.png")

plt.figure()
plt.hist(ranks, bins=num_subtypes)
plt.xlabel('Rank')
plt.ylabel('Count')
plt.xlim([1,num_subtypes])
plt.xticks(np.arange(1,num_subtypes+1))
plt.savefig("figures/ranks_histogram.png")

print()
print("Mean rank:", np.mean(ranks))
print("Hits at 1:", np.mean(ranks <= 1))
print("Hits at 3:", np.mean(ranks <= 3))
print("Hits at 5:", np.mean(ranks <= 5))
print("Hits at 10:", np.mean(ranks <= 10))
print("Hits at 15:", np.mean(ranks <= 15))

print("MRR:", np.mean(1/ranks))
print()

for ci in cluster_info:
    print(ci)


