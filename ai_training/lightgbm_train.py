import lightgbm as lgb
import pandas as pd
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split


# Discretizing relevance scores
def discretize_relevance(score):
    if score >= 0.8:
        return 3
    elif score >= 0.5:
        return 2
    elif score >= 0.2:
        return 1
    else:
        return 0


def train():
    # Load your dataset
    df = pd.read_csv("./ai_training/ranking_dataset.csv")

    # Features and labels
    features = [
        col for col in df.columns if col not in ["query_id", "doc_id", "relevance"]
    ]
    X = df[features]

    df["relevance"] = df["relevance"].apply(discretize_relevance)
    y = df["relevance"]

    # Group info (how many docs per query)
    group = df.groupby("query_id").size().to_list()

    # Ensure query-level splitting, not document-level, to preserve group structure
    train_query_ids, test_query_ids = train_test_split(
        df["query_id"].unique(), test_size=0.3, random_state=42
    )

    # Filter the dataset based on the query IDs in the split
    train_df = df[df["query_id"].isin(train_query_ids)]
    test_df = df[df["query_id"].isin(test_query_ids)]

    # Prepare the train and test datasets for LightGBM
    X_train = train_df[features]
    y_train = train_df["relevance"]
    group_train = train_df.groupby("query_id").size().to_list()

    X_test = test_df[features]
    y_test = test_df["relevance"]
    group_test = test_df.groupby("query_id").size().to_list()

    train_data = lgb.Dataset(X_train, label=y_train, group=group_train)

    # Model params
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5],
        "learning_rate": 0.1,
        "num_leaves": 31,
    }

    # Train the model
    model = lgb.train(params, train_data, num_boost_round=100)

    # Save the trained model
    model.save_model("./ai_training/lambdamart_model.txt")

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Manually calculate NDCG score
    # Reshape the predictions and true values according to group structure
    y_true = [
        y_test.iloc[sum(group_test[:i]) : sum(group_test[: i + 1])].values
        for i in range(len(group_test))
    ]
    y_pred = [
        y_pred[sum(group_test[:i]) : sum(group_test[: i + 1])].tolist()
        for i in range(len(group_test))
    ]

    # Calculate NDCG for the test set
    ndcg = ndcg_score(y_true, y_pred, k=5)

    print(f"\n# ---------- NDCG Score ----------")
    print(f"NDCG@5: {ndcg}")
