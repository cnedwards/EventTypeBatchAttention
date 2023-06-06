import math

def loadRun(path_to_run):
    """
    Loads run into dict, where key is topic and value is array of (docid, score) tuples

    :param path_to_run: the path to the run
    :type path_to_run: str

    :rtype: dict
    :returns: dict of run
    """
    run = {}
    with open(path_to_run, 'r') as f:
        for line in f:
            split_line = line.strip().split()
            topic = split_line[0]
            docid = split_line[2]
            score = float(split_line[4])
            if topic not in run:
                run[topic] = []
            run[topic].append((docid, score))
    return run

def crossEncode(path_to_run, cross_encoder, topics, docid_to_doc, topk=20):
    """
    Reranks topk documents using cross-encoder
    
    :param path_to_run: the path to the run
    :type path_to_run: str
   
    :param cross_encoder: the cross encoder model
    :type cross_encoder: sentence_transformer CrossEncoder

    :param topics: dict of topics
    :type: dict

    :param docid_to_doc: a map from the document id to the document passages
    :type docid_to_doc: array

    :param topk: number of documents to rerank
    :type topk: int

    :rtype: dict
    :returns: dict of reranked run
    """
    run = loadRun(path_to_run)
    for topic in run:
        query = topics[topic]['title']
        print(query)
        reranked_run = []
        for docid,score in run[topic][:topk]:
            # Each passage is cross-encoded, and the score for a document is the average of all passage scores
            doc_score = 0
            try:
                for passage in docid_to_doc[docid]:
                    doc_score = max(doc_score, cross_encoder.predict([(query, passage)])[0])

                reranked_run.append((docid, doc_score))
            except Exception as e:
                print(e)
        sorted_run = sorted(reranked_run, reverse=True, key=lambda x: x[1])
        run[topic] = sorted_run
    return run

def calcSigma(dist, k, rho):
    """
    Helper function for nn_pf_manifold
    Calculates (approximates) sigma, the distance normalizer for the manifold given a point
    
    :param dist: the distances to the k nearest neighbors
    :type dist: array of floats

    :param k: the number of nearest neighbors
    :type k: int

    :param rho: the distance to the closest nearest neighbor
    :type rho: float

    :rtype: float
    :returns: sigma, distance metric for local manifold
    """

    low = 0.0
    high = 1000
    mid = 1.0
    
    goal = math.log2(k)
    # The acceptable difference between our guess and the ideal psum
    tolorance = 0.0005

    psum = sum([math.exp((-1 * max(0, dist_i - rho)) / mid) for dist_i in dist])
    while True:
        # Sometimes, it may not converge (many points of same distance)
        if mid < 0.0000000000000001:
            return False

        if abs(psum-goal) < tolorance:
            return mid
        elif psum > goal:
            high = mid
            mid = (low+high) / 2.0
        else:
            low = mid
            mid = (low+high) / 2.0
        psum = sum([math.exp((-1 * max(0, dist_i - rho)) / mid) for dist_i in dist])


def nn_pf_manifold(path_to_run, model, topics, index, idx_to_docid, docid_to_doc, rel_docs=3, k=50, rerank_cutoff=None):
    """
    Nearest neighbor pseudo feedback but approximates the manifold like UMAP
    :param path_to_run: path to the run to rerank
    :type path_to_run: str

    :param model: the semantic encoder
    :type model: SentenceTransformer

    :param topics: dict of the topics
    :type topics: dict

    :param index: the hnswlib index for knn search
    :type index:

    :param idx_to_docid: the mapping between the hnswlib index output and the docid
    :type idx_to_docid: array

    :param docid_to_doc: the mapping between docid and the text in the doc
    :type docid_to_doc: dict

    :param rel_docs: number of relevant docs to gather passages from
    :type rel_docs: int

    :param k: the number of nearest neighbors to return
    :type k: int

    :param rerank_cutoff: if None, then the entire corpus is considered, otherwise only documents in 
                          path_to_run up to rerank_cutoff are considered
    :type rerank_cutoff: None or int

    :rtype: dict
    :returns: dict of reranked run
    """

    run = loadRun(path_to_run)
    manifold_runs = {}
    for topic in run:
        manifold_runs[topic] = []
        passages = []
        for docid,_ in run[topic][:rel_docs]:
            passages += docid_to_doc[docid]
        encoded_passages = model.encode(passages)
        labels, distances = index.knn_query(encoded_passages, k=k)
        document_sums = {}
        for i in range(len(encoded_passages)):
            # Ignore the distance to the passage itself
            passage_distances = distances[i][1:]
            # Find the closest point
            rho = min(passage_distances)
            # k-1 since we ignored the passage itself
            sigma = calcSigma(passage_distances, k-1, rho)
            if sigma:
                edge_weights = [math.exp( (-1 * max(0,dist_i - rho)) / sigma) for dist_i in passage_distances]
                for edge_weight, label in zip(edge_weights, labels[i][1:]):
                    docid = idx_to_docid[label]
                    if docid not in document_sums:
                        document_sums[docid] = 0
                    document_sums[docid] += edge_weight
                # add 1 to the original document as well
                orig_docid = idx_to_docid[labels[i][0]]
                if orig_docid not in document_sums:
                    document_sums[orig_docid] = 0
                document_sums[orig_docid] += 1
            else:
                print('Warning: the calculated sigma approached 0:', passage_distances)
                #print(passages[i])
        # create list sorted list, and note we don't normalize by length
        # assume longer documents have stronger arguments
        sorted_document_sums = sorted([(docid, document_sums[docid]) for docid in document_sums], reverse=True, key=lambda x: x[1])
        if rerank_cutoff:
            top_orig_docs = [docid[0] for docid in run[topic][:rerank_cutoff]]
            for doc in sorted_document_sums:
                if doc[0] in top_orig_docs:
                    # hack to make sure the manifold documents are ranked the highest
                    manifold_runs[topic].append((doc[0],doc[1] * 100))
            manifold_runs[topic] += run[topic][rerank_cutoff:]
            manifold_runs[topic] = manifold_runs[topic][:1000]
        else:            
            manifold_runs[topic] = sorted_document_sums[:1000]
    return manifold_runs

def nn_pf(path_to_run, model, topics, index, idx_to_docid, docid_to_doc, rel_docs=5, k=20):
    """
    Nearest neighbor pseudo feedback
    Assumes the top rel_docs are relevant, then does a k-nn search for all passages in those documents,
    and aggregates the scores of the similar passages

    :param path_to_run: path to the run to rerank
    :type path_to_run: str

    :param model: the semantic encoder
    :type model: SentenceTransformer

    :param topics: dict of the topics
    :type topics: dict

    :param index: the hnswlib index for knn search
    :type index: 

    :param idx_to_docid: the mapping between the hnswlib index output and the docid
    :type idx_to_docid: array

    :param docid_to_doc: the mapping between docid and the text in the doc
    :type docid_to_doc: dict

    :param rel_docs: number of relevant docs to gather passages from
    :type rel_docs: int
    
    :param k: the number of nearest neighbors to return
    :type k: int

    :rtype: dict
    :returns: dict of reranked run
    """


    run = loadRun(path_to_run)
    for topic in run:
        passages = []
        for docid,_ in run[topic][:rel_docs]:
            passages += docid_to_doc[docid]


        encoded_passages = model.encode(passages)
        scores = {}
        labels, distances = index.knn_query(encoded_passages, k=k)
        for i in range(len(encoded_passages)):
            for docidx, dist in zip(labels[i], distances[i]):
                docid = idx_to_docid[docidx]
                if docid not in scores:
                    scores[docid] = 0
                scores[docid] += 1-dist
        sorted_scores = sorted([(docidx, scores[docidx]) for docidx in scores], reverse=True, key=lambda x: x[1])
        run[topic] = sorted_scores
    return run


# interpolate runs
def interpolate(path_to_run1, path_to_run2, alpha):
    """
    Given to runs, combines the scores by run1 + (run2 * alpha)

    :param path_to_run1: path to the first run
    :type path_to_run1: str

    :param path_to_run2: path to the second run:
    :type path_to_run2: str

    :param alpha: how much of the second run should be added to the first run
    :type alpha: float

    :rtype: dict
    :returns: dict of reranked run
    """
    run1 = loadRun(path_to_run1)
    run2 = loadRun(path_to_run2)
    interpolated_runs = {}
    for topic in run1:
        # make run into dict
        interpolated_topic_run = {doc_score[0]: doc_score[1] for doc_score in run1[topic]}  
        for doc_score in run2[topic]:
            if doc_score[0] not in interpolated_topic_run:
                interpolated_topic_run[doc_score[0]] = alpha * doc_score[1]
            else:
                interpolated_topic_run[doc_score[0]] += alpha * doc_score[1]

        interpolated_topic_run = sorted([(doc, interpolated_topic_run[doc]) for doc in interpolated_topic_run], reverse=True, key=lambda x:x[1])
            
        interpolated_runs[topic] = interpolated_topic_run[:1000]
    return interpolated_runs

        