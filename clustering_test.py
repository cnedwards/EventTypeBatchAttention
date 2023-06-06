
import pickle
import numpy as np
import os
from numpy.random.mtrand import normal


from sklearn.metrics import normalized_mutual_info_score, fowlkes_mallows_score, completeness_score, homogeneity_score, v_measure_score
from sklearn.metrics import rand_score, adjusted_rand_score
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
from sklearn.metrics import silhouette_score


PRINT_CLUSTERS = False


SAVE_PATH = 'model_output/' #where the outputs are saved
MODEL = '7.pt' #which checkpoint to use

N_CLUSTERS = 23 #what cluster size to use


with open(SAVE_PATH + MODEL + ".query.npy", 'rb') as f:
    queries_dict = pickle.load(f)
with open(SAVE_PATH + MODEL + ".key.npy", 'rb') as f:
    keys_dict = pickle.load(f)


with open(SAVE_PATH + MODEL + ".CLS.npy", 'rb') as f:
    cls_rep = pickle.load(f)

with open(SAVE_PATH + MODEL + "eid_subtypes.npy", 'rb') as f:
    eid_subtypes = pickle.load(f)

EIDs = []
subtypes = []
queries = []
keys = []
CLS = []

seen = ['Attack', 'Transport', 'Die', 'Meet', 'Arrest-Jail', 'Sentence', 'Transfer-Money', 'Elect', 'Transfer-Ownership', 'End-Position']

unseen_mask = []

for id in queries_dict:
    EIDs.append(id)
    subtypes.append(eid_subtypes[id])
    queries.append(queries_dict[id])
    keys.append(keys_dict[id])
    CLS.append(cls_rep[id])
    unseen_mask.append(eid_subtypes[id] not in seen)



EIDs = np.array(EIDs)
subtypes = np.array(subtypes)
unseen_mask = np.array(unseen_mask)

keys = np.array(keys)
queries = np.array(queries)
CLS = np.array(CLS)

EIDs = EIDs[unseen_mask]
keys = keys[unseen_mask]
queries = queries[unseen_mask]
CLS = CLS[unseen_mask]

subtypes = subtypes[unseen_mask]

X = np.concatenate((keys, queries, CLS), axis=1)


if not os.path.exists(SAVE_PATH+MODEL[:-3]):
  os.mkdir(SAVE_PATH+MODEL[:-3])

FINAL_PATH = SAVE_PATH+MODEL[:-3] + "/clustering_test/"

if not os.path.exists(FINAL_PATH):
  os.mkdir(FINAL_PATH)


def print_clusters(ppath):
    if print_clusters: #print clusters -- make this a function later
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

if True: #AffinityPropagation
    S = np.matmul(queries, keys.T)
    clustering = AffinityPropagation(random_state=0, affinity='precomputed').fit(S)
    
    pred = clustering.labels_

    try:
        print('AffinityPropagation: k =', len(np.unique(clustering.labels_)))
        print('Arithmetic NMI: {:.3f}'.format(100*normalized_mutual_info_score(subtypes, pred)))
        print('Geometric NMI: {:.3f}'.format(100*normalized_mutual_info_score(subtypes, pred, average_method='geometric')))
        print('Fowlkes Mallows: {:.3f}'.format(100*fowlkes_mallows_score(subtypes, pred)))
        print('Completeness: {:.3f}'.format(100*completeness_score(subtypes, pred)))
        print('Homogeneity: {:.3f}'.format(100*homogeneity_score(subtypes, pred)))
        print('V-measure: {:.3f}'.format(100*v_measure_score(subtypes, pred)))
        print('Rand: {:.3f}'.format(100*rand_score(subtypes, pred)))
        print('Adjusted Rand: {:.3f}'.format(100*adjusted_rand_score(subtypes, pred)))
        print('Silhouette: {:.3f}'.format(100*silhouette_score(CLS, pred, metric='cosine')))
    except:
        print('There was an issue with the clustering.')



if True: #AffinityPropagation-Cosine
    from sklearn.metrics.pairwise import cosine_similarity
    S = cosine_similarity(queries, keys)
    clustering = AffinityPropagation(random_state=0, affinity='precomputed').fit(S)
    
    pred = clustering.labels_

    try:
        print('AffinityPropagation-Cosine: k =', len(np.unique(clustering.labels_)))
        print('Arithmetic NMI: {:.3f}'.format(100*normalized_mutual_info_score(subtypes, pred)))
        print('Geometric NMI: {:.3f}'.format(100*normalized_mutual_info_score(subtypes, pred, average_method='geometric')))
        print('Fowlkes Mallows: {:.3f}'.format(100*fowlkes_mallows_score(subtypes, pred)))
        print('Completeness: {:.3f}'.format(100*completeness_score(subtypes, pred)))
        print('Homogeneity: {:.3f}'.format(100*homogeneity_score(subtypes, pred)))
        print('V-measure: {:.3f}'.format(100*v_measure_score(subtypes, pred)))
        print('Rand: {:.3f}'.format(100*rand_score(subtypes, pred)))
        print('Adjusted Rand: {:.3f}'.format(100*adjusted_rand_score(subtypes, pred)))
        print('Silhouette: {:.3f}'.format(100*silhouette_score(CLS, pred, metric='cosine')))
    except:
        print('There was an issue with the clustering.')


if True: #AgglomerativeClustering
    S = np.matmul(queries, keys.T)
    clustering = AgglomerativeClustering(n_clusters=N_CLUSTERS, affinity='precomputed', linkage='average').fit(-S)

    pred = clustering.labels_


    print('DotProduct+Agglomerative:')
    print('Arithmetic NMI: {:.3f}'.format(100*normalized_mutual_info_score(subtypes, pred)))
    print('Geometric NMI: {:.3f}'.format(100*normalized_mutual_info_score(subtypes, pred, average_method='geometric')))
    print('Fowlkes Mallows: {:.3f}'.format(100*fowlkes_mallows_score(subtypes, pred)))
    print('Completeness: {:.3f}'.format(100*completeness_score(subtypes, pred)))
    print('Homogeneity: {:.3f}'.format(100*homogeneity_score(subtypes, pred)))
    print('V-measure: {:.3f}'.format(100*v_measure_score(subtypes, pred)))
    print('Rand: {:.3f}'.format(100*rand_score(subtypes, pred)))
    print('Adjusted Rand: {:.3f}'.format(100*adjusted_rand_score(subtypes, pred)))
    print('Silhouette: {:.3f}'.format(100*silhouette_score(CLS, pred, metric='cosine')))

    if PRINT_CLUSTERS:
        print_clusters(SAVE_PATH + MODEL + "." + str(N_CLUSTERS) +".agglo.clusters.txt")
    

if True: #AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    
    S = 1 - cosine_similarity(queries, keys)
    clustering = AgglomerativeClustering(n_clusters=N_CLUSTERS, affinity='precomputed', linkage='average').fit(S)

    pred = clustering.labels_

    print('CosineDistance+Agglomerative:')
    print('Arithmetic NMI: {:.3f}'.format(100*normalized_mutual_info_score(subtypes, pred)))
    print('Geometric NMI: {:.3f}'.format(100*normalized_mutual_info_score(subtypes, pred, average_method='geometric')))
    print('Fowlkes Mallows: {:.3f}'.format(100*fowlkes_mallows_score(subtypes, pred)))
    print('Completeness: {:.3f}'.format(100*completeness_score(subtypes, pred)))
    print('Homogeneity: {:.3f}'.format(100*homogeneity_score(subtypes, pred)))
    print('V-measure: {:.3f}'.format(100*v_measure_score(subtypes, pred)))
    print('Rand: {:.3f}'.format(100*rand_score(subtypes, pred)))
    print('Adjusted Rand: {:.3f}'.format(100*adjusted_rand_score(subtypes, pred)))
    print('Silhouette: {:.3f}'.format(100*silhouette_score(CLS, pred, metric='cosine')))

    if PRINT_CLUSTERS:
        print_clusters(SAVE_PATH + MODEL + "." + str(N_CLUSTERS) +".agglo.clusters.txt")
    

if True: #AgglomerativeClustering+CLS
    from sklearn.metrics.pairwise import cosine_similarity
    
    S = 1 - cosine_similarity(CLS)
    clustering = AgglomerativeClustering(n_clusters=N_CLUSTERS, affinity='precomputed', linkage='average').fit(S)

    pred = clustering.labels_

    print('CosineDistance+Agglomerative+CLS:')
    print('Arithmetic NMI: {:.3f}'.format(100*normalized_mutual_info_score(subtypes, pred)))
    print('Geometric NMI: {:.3f}'.format(100*normalized_mutual_info_score(subtypes, pred, average_method='geometric')))
    print('Fowlkes Mallows: {:.3f}'.format(100*fowlkes_mallows_score(subtypes, pred)))
    print('Completeness: {:.3f}'.format(100*completeness_score(subtypes, pred)))
    print('Homogeneity: {:.3f}'.format(100*homogeneity_score(subtypes, pred)))
    print('V-measure: {:.3f}'.format(100*v_measure_score(subtypes, pred)))
    print('Rand: {:.3f}'.format(100*rand_score(subtypes, pred)))
    print('Adjusted Rand: {:.3f}'.format(100*adjusted_rand_score(subtypes, pred)))
    print('Silhouette: {:.3f}'.format(100*silhouette_score(CLS, pred, metric='cosine')))

    if PRINT_CLUSTERS:
        print_clusters(SAVE_PATH + MODEL + "." + str(N_CLUSTERS) +".agglo.clusters.txt")
    

if True: #Manifold+AgglomerativeClustering
    from manifoldIR import reranker
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics.pairwise import cosine_similarity
    import math
    n = S.shape[0]
    k = n

    S = 1 - cosine_similarity(queries, keys)
    #S = -S #for using raw dot products

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
    print('Silhouette: {:.3f}'.format(100*silhouette_score(CLS, pred, metric='cosine')))


if True: #Manifold+CLS+AgglomerativeClustering
    from manifoldIR import reranker
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics.pairwise import cosine_similarity
    import math
    n = S.shape[0]
    k = n

    S = 1 - cosine_similarity(CLS)

    umap_weights = np.zeros((n,n))

    for i in range(n):
        # Ignore the distance to the passage itself
        #labels = np.argsort(S[i,:])[::-1][:k]
        #distances = np.sort(S[i,:])[::-1][:k]
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
    print('Silhouette: {:.3f}'.format(100*silhouette_score(CLS, pred, metric='cosine')))

    print_clusters_downstream(SAVE_PATH + MODEL + "." + str(N_CLUSTERS) +".agglo.UMAP_CLS."+str(k)+".clusters_downstream.txt")

        