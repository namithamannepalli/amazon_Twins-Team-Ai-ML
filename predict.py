"""
Prediction + formatting: score test candidate pairs with the trained model,
keep only the ones predicted as true matches, and write matching_results.tsv
in the shape the challenge expects (one row per Source 1 entity, matches as
a comma-separated list, empty if none).

candidate_pairs.tsv uses the exact same one-row-per-Source-1-entity shape
(see blocking.group_ids_by_source1), just built from the pre-model
candidate set instead of the post-model predictions.
"""

import pandas as pd
from .train_model import FEATURE_COLUMNS

# Because a false merge costs ~2x a miss under the F0.5 metric, being
# slightly conservative (a higher threshold) usually helps your score more
# than it hurts. 0.5 is a neutral starting point -- try raising it (e.g.
# 0.6, 0.7) and see how your local validation-split F0.5 responds.
DEFAULT_MATCH_THRESHOLD = 0.5


def predict_matches(model, feature_df: pd.DataFrame, threshold: float = DEFAULT_MATCH_THRESHOLD) -> pd.DataFrame:
    """
    Score every candidate pair (feature_df must have columns "id_a", "id_b"
    plus the FEATURE_COLUMNS) and keep only those above the threshold.
    Returns a DataFrame with columns ["id_a", "id_b"] containing only
    predicted true matches -- ready to feed into
    blocking.group_ids_by_source1().
    """
    if len(feature_df) == 0:
        return feature_df[["id_a", "id_b"]].copy()

    X = feature_df[FEATURE_COLUMNS]
    proba = model.predict_proba(X)
    if 1 in model.classes_:
        match_col_index = list(model.classes_).index(1)
        probs = proba[:, match_col_index]
    else:
        # Extremely small/unbalanced training data can leave the model
        # having only ever seen one class. Fall back safely rather than
        # crashing -- this should not happen once you're training on the
        # real, full-size dataset.
        probs = [0.0] * len(feature_df)

    feature_df = feature_df.copy()
    feature_df["match_probability"] = probs
    matches = feature_df[feature_df["match_probability"] >= threshold]
    return matches[["id_a", "id_b"]].reset_index(drop=True)


def save_tsv(df: pd.DataFrame, path: str):
    df.to_csv(path, sep="\t", index=False)
    print(f"Wrote {len(df)} rows to {path}")
