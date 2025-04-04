from collections import Counter

import pandas as pd

import lightgbm as lgb
from ai_training.lightgbm_train import train
from ai_training.generate_dataset import generate_dataset, encode_query_tokens
from ebm.ebm import rank_docs


def parse_documents_from_txt(input_path):
    """
    example documents format
    D1: {bird,cat,bird,cat,dog,dog,bird}
    D2: {cat,tiger,cat,dog}
    D3: {dog,bird,bird}
    """
    documents = []

    with open(input_path, "r") as file:
        for line in file:
            if line.strip():  # Skip empty lines
                # Extract the list of keywords from the curly braces and split by comma
                doc_str = line.split(":")[1].strip(" {}\n")
                keywords = doc_str.split(",")
                # Append the list of keywords to the documents array
                documents.append(keywords)

    return documents


if __name__ == "__main__":
    # generate_dataset()

    # # ---------- NDCG Score ----------
    # NDCG@5: 0.8513661466555181
    # train()

    model = lgb.Booster(model_file="./ai_training/lambdamart_model.txt")

    # ex: (cat AND dog) AND NOT tiger
    query = input("Enter query: ").strip() or "(cat AND dog) AND NOT tiger"
    input_path = (
        input("Enter documents' list path (documents.txt): ").strip() or "documents.txt"
    )

    docs = parse_documents_from_txt(input_path)
    encoded_query = encode_query_tokens(query)

    test_rows = []
    for i, doc in enumerate(docs):
        counter = Counter(doc)
        row = {f"position{j}": encoded_query[j] for j in range(len(encoded_query))}
        row.update(
            {
                "bird_count": counter.get("bird", 0),
                "cat_count": counter.get("cat", 0),
                "dog_count": counter.get("dog", 0),
                "tiger_count": counter.get("tiger", 0),
            }
        )
        test_rows.append(row)

    test_df = pd.DataFrame(test_rows)

    preds = model.predict(test_df)
    test_df["score"] = preds
    ranked_docs = test_df.sort_values("score", ascending=False)

    # Now, re-arrange documents in the same order as sorted by the model score
    ranked_documents = []
    for index in ranked_docs.index:
        ranked_documents.append(f"D{index + 1}: {{{','.join(docs[index])}}}")

    # Output the ranked documents
    ranked_documents_str = "\n".join(ranked_documents)
    ranked_documents_ebm = rank_docs(query, docs)

    # Print documents ranked by EBM
    print("\n\n# ---------- EBM Ranked Documents ----------")
    for rank, (doc_id, score) in enumerate(ranked_documents_ebm, 1):
        print(f"Rank {rank}: Doc {doc_id} (Relevance Score: {score:.4f})")

    # Print documents ranked by AI
    print("\n\n# ---------- AI Ranked Documents ----------")
    print(ranked_documents_str)
