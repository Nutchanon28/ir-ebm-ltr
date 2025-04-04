import math
from collections import Counter
from ebm.query_parser import convert_query


# ---------- Step#1: Calculate the weights of keywords in each docs ----------


def calculate_weight(documents):
    keywords = ["bird", "cat", "dog", "tiger"]
    N = len(documents)

    # Count how many documents contain each keyword
    df = {word: sum(word in doc for doc in documents) for word in keywords}
    idf = {word: math.log(N / df[word]) if df[word] else 0 for word in keywords}
    max_idf = max(idf.values())

    # Compute tf-idf-norm weights for each document
    doc_weights = []
    for doc in documents:
        counter = Counter(doc)
        max_tf = max(counter.values()) if counter else 1
        weights = {}
        for word in keywords:
            tf = counter[word] / max_tf
            idf_norm = (idf[word] / max_idf) if max_idf else 0
            weights[word] = tf * idf_norm
        doc_weights.append(weights)

    return doc_weights


# ---------- Step#2: Create functions to parse the query to calculate relevance ----------


def sim_or(w1, w2):
    return math.sqrt((w1**2 + w2**2) / 2)


def sim_and(w1, w2):
    return 1 - math.sqrt(((1 - w1) ** 2 + (1 - w2) ** 2) / 2)


def sim_not(w):
    return 1 - w


# ---------- Step#3: Calculate relevance of each docs ----------


def evaluate_query(query, weights):
    query_tree = convert_query(query)
    return _evaluate_query_tree(query_tree, weights)


# query_tree needs to be in a special format
# query = (cat AND dog) AND NOT tiger
# query_tree = ["AND", ["AND", "cat", "dog"], ["NOT", "tiger"]]
def _evaluate_query_tree(query_tree, weights):
    if isinstance(query_tree, str):
        return weights.get(query_tree, 0)

    op = query_tree[0]
    if op == "NOT":
        return sim_not(_evaluate_query_tree(query_tree[1], weights))
    elif op == "AND":
        w1 = _evaluate_query_tree(query_tree[1], weights)
        w2 = _evaluate_query_tree(query_tree[2], weights)
        return sim_and(w1, w2)
    elif op == "OR":
        w1 = _evaluate_query_tree(query_tree[1], weights)
        w2 = _evaluate_query_tree(query_tree[2], weights)
        return sim_or(w1, w2)


# ---------- Step#4: Rank the documents ----------


def rank_docs(query, documents):
    doc_weights = calculate_weight(documents)

    ranked_docs = sorted(
        [
            (i + 1, evaluate_query(query, weights))
            for i, weights in enumerate(doc_weights)
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    return ranked_docs
