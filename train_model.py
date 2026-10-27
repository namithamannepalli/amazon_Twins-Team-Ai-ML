"""
Training: teach a classifier to tell "true match" from "not a match" using
the similarity features from features.py and the official ground truth.

Two ways to use this module, from simplest to more powerful:

1. THRESHOLD BASELINE (no training needed): just call `label_pairs()` to see
   how well a single similarity column already separates matches from
   non-matches, then pick a cutoff. Good first submission.

2. CLASSIFIER (this module's main purpose): a RandomForest that learns to
   combine ALL the features at once, which usually beats a single threshold.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

FEATURE_COLUMNS = [
    "name_token_sort_ratio",
    "name_partial_ratio",
    "address_token_sort_ratio",
    "address_partial_ratio",
    "name_jaccard",
    "address_jaccard",
    "name_exact_match",
    "address_exact_match",
    "country_exact_match",
]


def label_pairs(feature_df: pd.DataFrame, ground_truth_matches: dict, a_id_col: str, b_id_col: str) -> pd.DataFrame:
    """
    Attach a 0/1 'label' column to feature_df by checking each (a_id, b_id)
    pair against the ground truth.

    ground_truth_matches: dict mapping a_id -> set of true matching b_ids.
    (See run_pipeline.py for how this dict gets built from train_ground_truth.tsv)
    """
    def is_match(row):
        true_matches = ground_truth_matches.get(row[a_id_col], set())
        return int(row[b_id_col] in true_matches)

    feature_df = feature_df.copy()
    feature_df["label"] = feature_df.apply(is_match, axis=1)
    return feature_df


def train_classifier(labeled_feature_df: pd.DataFrame):
    """
    Train a RandomForest on labeled candidate pairs and print a quick
    validation report so you know how it's doing before you trust it.
    """
    X = labeled_feature_df[FEATURE_COLUMNS]
    y = labeled_feature_df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # class_weight="balanced" matters here: true matches are usually a small
    # minority of all candidate pairs, and we don't want the model to just
    # learn to always predict "not a match".
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    print("--- Validation performance (held-out slice of training data) ---")
    val_preds = model.predict(X_val)
    print(classification_report(y_val, val_preds, digits=3))

    return model


def save_model(model, path: str):
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: str):
    return joblib.load(path)
