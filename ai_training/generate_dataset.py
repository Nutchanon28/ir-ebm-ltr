import csv
from collections import Counter
from ebm.ebm import calculate_weight, evaluate_query  # EBM similarity evaluator

# ---------- Configurable ----------
queries = [
    "(cat AND dog) AND NOT tiger",
    "(bird OR cat) AND dog",
    "bird AND NOT (dog OR tiger)",
    "dog OR (cat AND bird)",
    "tiger AND NOT cat",
    "cat OR dog",
    "(cat AND bird) OR (dog AND tiger)",
    "NOT bird",
    "(tiger AND cat) AND (dog OR bird)",
    "cat OR bird OR dog",
]

documents = [
    ["bird", "cat", "bird", "cat", "dog", "dog", "bird"],  # D1
    ["cat", "tiger", "cat", "dog"],  # D2
    ["dog", "bird", "bird"],  # D3
    ["cat", "tiger"],  # D4
    ["tiger", "tiger", "dog", "tiger", "cat"],  # D5
    ["bird", "cat", "bird", "cat", "tiger", "tiger", "bird"],  # D6
    ["bird", "tiger", "cat", "dog"],  # D7
    ["dog", "cat", "bird"],  # D8
    ["cat", "dog", "tiger"],  # D9
    ["tiger", "tiger", "tiger"],  # D10
    ["cat", "cat", "cat", "dog", "dog"],  # D11
    ["bird", "bird", "bird"],  # D12
    ["dog", "dog", "tiger", "cat"],  # D13
    ["bird", "dog", "tiger"],  # D14
    ["cat", "dog", "dog", "dog", "cat"],  # D15
]

keywords = ["bird", "cat", "dog", "tiger"]

# ---------- Token Encoding ----------
token_map = {
    "AND": 0,
    "OR": 1,
    "NOT": 2,
    "(": 3,
    ")": 4,
    "cat": 5,
    "dog": 6,
    "bird": 7,
    "tiger": 8,
}


# Encode the query into a fixed-length vector (max 10 tokens, pad with -1)
def encode_query_tokens(query, max_len=10):
    tokens = query.replace("(", " ( ").replace(")", " ) ").split()
    encoded = [token_map.get(tok, -1) for tok in tokens]
    encoded += [-1] * (max_len - len(encoded))
    return encoded[:max_len]


# ---------- Generate CSV ----------


def generate_dataset():
    # TF-IDF-Norm Calculation
    doc_weights = calculate_weight(documents)

    with open("./ai_training/ranking_dataset.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["query_id", "doc_id"]
        header += [f"position{i}" for i in range(10)]
        header += ["bird_count", "cat_count", "dog_count", "tiger_count", "relevance"]
        writer.writerow(header)

        for query_id, query in enumerate(queries):
            query_features = encode_query_tokens(query)
            for doc_id, (doc, weights) in enumerate(zip(documents, doc_weights)):
                counter = Counter(doc)
                row = [query_id, doc_id]
                row += query_features
                row += [counter.get(w, 0) for w in keywords]
                row += [evaluate_query(query, weights)]
                writer.writerow(row)

        print("✅ Dataset saved to 'ranking_dataset.csv'")
