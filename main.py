from ebm.ebm import rank_docs

if __name__ == "__main__":
    # documents and query from Chap02_2 page 55
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
    ]

    query = "(cat AND dog) AND NOT tiger"

    ranked_docs = rank_docs(query, documents)
    print("\n\n# ---------- Ranked Documents ----------")
    for rank, (doc_id, score) in enumerate(ranked_docs, 1):
        print(f"Rank {rank}: Doc {doc_id} (Relevance Score: {score:.4f})")
