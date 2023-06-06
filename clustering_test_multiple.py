
import pickle
import numpy as np
import os
from numpy.random.mtrand import normal
from clustering_test import PRINT_CLUSTERS

from sklearn.metrics import normalized_mutual_info_score, fowlkes_mallows_score, completeness_score, homogeneity_score, v_measure_score
from sklearn.metrics import rand_score, adjusted_rand_score
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering

SAVE_PATH = []
MODEL = []

#add as many checkpoints as desired here. In the paper different runs are used. 

SAVE_PATH.append('model_output/')
MODEL.append('5.pt')

SAVE_PATH.append('model_output/')
MODEL.append('6.pt')



def print_clusters(ppath):
    from dataloader import load_zeroshot_data_allmentions
    
    MODEL_NAME = 'sentence-transformers/all-MiniLM-L12-v2'
    dataset = load_zeroshot_data_allmentions(pretrained_model=MODEL_NAME)
    
    texts, sts, pds = {}, {}, {}

    with open(ppath, 'w') as f:
        f.write("EID\tpredicted\ttruth\tinput\n")
        for meid, s, p in zip(EIDs, subtypes, pred):
                    
            mention = dataset.mentions[meid]
            text = mention['ldc_scope']['charseq']['#text']

            texts[meid] = text
            sts[meid] = s
            pds[meid] = p

        old_p = 0
        for meid, p in sorted(pds.items(), key=lambda item: item[1]):
            if p != old_p:
                old_p = p
                f.write("\n")

            s = sts[meid]
            text = texts[meid]
            f.write(meid + "\t" + s + "\t" + str(p) + "\t" + text + "\n")


def print_clusters_downstream(ppath):
    sts, pds = {}, {}

    with open(ppath, 'w') as f:
        f.write("EID\ttruth\tpredicted\n")
        for meid, s, p in zip(EIDs, subtypes, pred):
                    

            sts[meid] = s
            pds[meid] = p

        old_p = 0
        for meid, p in sorted(pds.items(), key=lambda item: item[1]):
            if p != old_p:
                old_p = p
                f.write("\n")

            s = sts[meid]
            f.write(meid + "\t" + s + "\t" + str(p) + "\n")



queries_dict = []
keys_dict = []
cls_rep = []

for s, m in zip(SAVE_PATH, MODEL):
    with open(s + m + ".query.npy", 'rb') as f:
        queries_dict.append(pickle.load(f))
    with open(s + m + ".key.npy", 'rb') as f:
        keys_dict.append(pickle.load(f))


    with open(s + m + ".CLS.npy", 'rb') as f:
        cls_rep.append(pickle.load(f))

    with open(s + m + "eid_subtypes.npy", 'rb') as f:
        eid_subtypes = pickle.load(f)

EIDs = []
subtypes = []
queries = []
keys = []
CLS = []

seen = ['Attack', 'Transport', 'Die', 'Meet', 'Arrest-Jail', 'Sentence', 'Transfer-Money', 'Elect', 'Transfer-Ownership', 'End-Position']

unseen_mask = []

for i in range(len(SAVE_PATH)):
    EIDs.append([])
    queries.append([])
    keys.append([])
    CLS.append([])
    for id in queries_dict[i]:
        EIDs[-1].append(id)
        queries[-1].append(queries_dict[i][id])
        keys[-1].append(keys_dict[i][id])
        CLS[-1].append(cls_rep[i][id])
        if i == 0:
            unseen_mask.append(eid_subtypes[id] not in seen)
            subtypes.append(eid_subtypes[id])

    EIDs[-1] = np.array(EIDs[-1])
    unseen_mask[-1] = np.array(unseen_mask[-1])
    keys[-1] = np.array(keys[-1])
    queries[-1] = np.array(queries[-1])
    CLS[-1] = np.array(CLS[-1])

    if i != 0:
        tmp = EIDs[-1].tolist()
        indexes = [tmp.index(i) for i in EIDs[0]]

        keys[-1] = keys[-1][indexes]
        queries[-1] = queries[-1][indexes]
        CLS[-1] = CLS[-1][indexes]

subtypes = np.array(subtypes)
EIDs = EIDs[0]


EIDs = EIDs[unseen_mask]
for i in range(len(keys)):
    keys[i] = keys[i][unseen_mask]
    queries[i] = queries[i][unseen_mask]
    CLS[i] = CLS[i][unseen_mask]

subtypes = subtypes[unseen_mask]


S = []

for i in range(len(keys)):

    S.append(np.matmul(queries[i], keys[i].T))

S = np.array(S)
S = np.mean(S, axis=0)

N_CLUSTERS = 23


if True: #AffinityPropagation
    clustering = AffinityPropagation(random_state=0, affinity='precomputed').fit(S)

    pred = clustering.labels_

    print('AffinityPropagation: k =', len(np.unique(clustering.labels_)))
    print('Arithmetic NMI: {:.3f}'.format(100*normalized_mutual_info_score(subtypes, pred)))
    print('Geometric NMI: {:.3f}'.format(100*normalized_mutual_info_score(subtypes, pred, average_method='geometric')))
    print('Fowlkes Mallows: {:.3f}'.format(100*fowlkes_mallows_score(subtypes, pred)))
    print('Completeness: {:.3f}'.format(100*completeness_score(subtypes, pred)))
    print('Homogeneity: {:.3f}'.format(100*homogeneity_score(subtypes, pred)))
    print('V-measure: {:.3f}'.format(100*v_measure_score(subtypes, pred)))
    print('Rand: {:.3f}'.format(100*rand_score(subtypes, pred)))
    print('Adjusted Rand: {:.3f}'.format(100*adjusted_rand_score(subtypes, pred)))



if True: #AgglomerativeClustering
    clustering = AgglomerativeClustering(n_clusters=N_CLUSTERS, affinity='precomputed', linkage='average').fit(1-S)

    pred = clustering.labels_


    print('AgglomerativeClustering:')
    print('Arithmetic NMI: {:.3f}'.format(100*normalized_mutual_info_score(subtypes, pred)))
    print('Geometric NMI: {:.3f}'.format(100*normalized_mutual_info_score(subtypes, pred, average_method='geometric')))
    print('Fowlkes Mallows: {:.3f}'.format(100*fowlkes_mallows_score(subtypes, pred)))
    print('Completeness: {:.3f}'.format(100*completeness_score(subtypes, pred)))
    print('Homogeneity: {:.3f}'.format(100*homogeneity_score(subtypes, pred)))
    print('V-measure: {:.3f}'.format(100*v_measure_score(subtypes, pred)))
    print('Rand: {:.3f}'.format(100*rand_score(subtypes, pred)))
    print('Adjusted Rand: {:.3f}'.format(100*adjusted_rand_score(subtypes, pred)))


if True: #Manifold+AgglomerativeClustering
    from manifoldIR import reranker
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics.pairwise import cosine_similarity
    import math
    n = S.shape[0]
    k = n


    S = []

    for i in range(len(keys)):
        S.append(1 - cosine_similarity(queries[i], keys[i]))

    S = np.array(S)
    S = np.mean(S, axis=0)

    umap_weights = np.zeros((n,n))

    for i in range(n):
        # Ignore the distance to the passage itself
        #cosine sim:
        labels = np.argsort(S[i,:])[:k]
        distances = np.sort(S[i,:])[:k]
        
        dist = distances[1:]
        # Find the closest point
        rho = min(dist)
        # k-1 since we ignored the sentence itself
        sigma = reranker.calcSigma(dist, k-1, rho)
        if sigma:
            edge_weights = [math.exp( (-1 * max(0,dist_i - rho)) / sigma) for dist_i in dist]
            for edge_weight, label in zip(edge_weights, labels[1:]):                
                umap_weights[i, label] = edge_weight
            umap_weights[i,i] = 1
        else:
            #assume they are all the same:
            for label in labels[1:]:
                umap_weights[i, label] = 1
            umap_weights[i,i] = 1

    clustering = AgglomerativeClustering(n_clusters=N_CLUSTERS, affinity='precomputed', linkage='average').fit(1 - umap_weights)

    pred = clustering.labels_


    print('Manifold+CosineDistance+Agglo:')
    print('Arithmetic NMI: {:.3f}'.format(100*normalized_mutual_info_score(subtypes, pred)))
    print('Geometric NMI: {:.3f}'.format(100*normalized_mutual_info_score(subtypes, pred, average_method='geometric')))
    print('Fowlkes Mallows: {:.3f}'.format(100*fowlkes_mallows_score(subtypes, pred)))
    print('Completeness: {:.3f}'.format(100*completeness_score(subtypes, pred)))
    print('Homogeneity: {:.3f}'.format(100*homogeneity_score(subtypes, pred)))
    print('V-measure: {:.3f}'.format(100*v_measure_score(subtypes, pred)))
    print('Rand: {:.3f}'.format(100*rand_score(subtypes, pred)))
    print('Adjusted Rand: {:.3f}'.format(100*adjusted_rand_score(subtypes, pred)))


if True: #Manifold+CLS+AgglomerativeClustering
    from manifoldIR import reranker
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics.pairwise import cosine_similarity
    import math
    n = S.shape[0]
    k = n

    S = []

    for i in range(len(keys)):
        S.append(1 - cosine_similarity(CLS[i]))

    S = np.array(S)
    S = np.mean(S, axis=0)

    umap_weights = np.zeros((n,n))

    for i in range(n):
        # Ignore the distance to the passage itself
        #cosine sim:
        labels = np.argsort(S[i,:])[:k]
        distances = np.sort(S[i,:])[:k]
        
        dist = distances[1:]
        # Find the closest point
        rho = min(dist)
        # k-1 since we ignored the sentence itself
        sigma = reranker.calcSigma(dist, k-1, rho)
        if sigma:
            edge_weights = [math.exp( (-1 * max(0,dist_i - rho)) / sigma) for dist_i in dist]
            for edge_weight, label in zip(edge_weights, labels[1:]):                
                umap_weights[i, label] = edge_weight
            umap_weights[i,i] = 1
        else:
            #assume they are all the same:
            for label in labels[1:]:
                umap_weights[i, label] = 1
            umap_weights[i,i] = 1

    clustering = AgglomerativeClustering(n_clusters=N_CLUSTERS, affinity='precomputed', linkage='average').fit(1 - umap_weights)

    pred = clustering.labels_


    print('Manifold+CLS+Agglomerative:')
    print('Arithmetic NMI: {:.3f}'.format(100*normalized_mutual_info_score(subtypes, pred)))
    print('Geometric NMI: {:.3f}'.format(100*normalized_mutual_info_score(subtypes, pred, average_method='geometric')))
    print('Fowlkes Mallows: {:.3f}'.format(100*fowlkes_mallows_score(subtypes, pred)))
    print('Completeness: {:.3f}'.format(100*completeness_score(subtypes, pred)))
    print('Homogeneity: {:.3f}'.format(100*homogeneity_score(subtypes, pred)))
    print('V-measure: {:.3f}'.format(100*v_measure_score(subtypes, pred)))
    print('Rand: {:.3f}'.format(100*rand_score(subtypes, pred)))
    print('Adjusted Rand: {:.3f}'.format(100*adjusted_rand_score(subtypes, pred)))

    print_clusters_downstream("Ensemble "+ "." + str(N_CLUSTERS) +".agglo.UMAP_CLS."+str(k)+".clusters_downstream.txt")