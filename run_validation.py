"""
Hold out a slice of Source 1 TRAINING entities as a fake "test set", train
on the rest, predict on the held-out slice, and score yourself with the
official macro F0.5 formula.

Why this exists: the real test set has no ground truth, so this is the only
honest way to know whether a change (new feature, different threshold,
tighter blocking) actually helps before spending a real submission on it.

Run this BEFORE run_pipeline.py, and any time you change something in src/.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.blocking import generate_candidate_pairs
from src.features import build_feature_table
from src.train_model import label_pairs, train_classifier
from src.predict import predict_matches, DEFAULT_MATCH_THRESHOLD
from src.scoring import macro_f0_5

# Keep these in sync with run_pipeline.py's CONFIG section.
DATA_DIR = os.environ.get("ER_DATA_DIR", "dataset")
TRAIN_SOURCE1_PATH = os.path.join(DATA_DIR, "train", "train_source1.tsv")
TRAIN_SOURCE2_PATH = os.path.join(DATA_DIR, "train", "train_source2.tsv")
TRAIN_SOURCE3_PATH = os.path.join(DATA_DIR, "train", "train_source3.tsv")
TRAIN_GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")

ID_COL = "entity_id"
NAME_COL = "business_name"
ADDRESS_COL = "business_address"
COUNTRY_COL = "country"

GT_SOURCE1_ID_COL = "source1_entity_id"
GT_MATCHES_COL = "matched_entity_ids"
SOURCE2_ID_PREFIX = "S2"
SOURCE3_ID_PREFIX = "S3"

VALIDATION_FRACTION = 0.2
RANDOM_SEED = 42
MATCH_THRESHOLD = DEFAULT_MATCH_THRESHOLD


def load_ground_truth_as_dicts(path):
    gt = pd.read_csv(path, sep="\t").fillna("")
    s1_to_s2, s1_to_s3, s1_to_all = {}, {}, {}
    for _, row in gt.iterrows():
        s1_id = row[GT_SOURCE1_ID_COL]
        all_matches = {m.strip() for m in str(row[GT_MATCHES_COL]).split(",") if m.strip()}
        s1_to_s2[s1_id] = {m for m in all_matches if m.startswith(SOURCE2_ID_PREFIX)}
        s1_to_s3[s1_id] = {m for m in all_matches if m.startswith(SOURCE3_ID_PREFIX)}
        s1_to_all[s1_id] = all_matches
    return s1_to_s2, s1_to_s3, s1_to_all


def block_and_featurize(source1_df, other_df):
    pairs = generate_candidate_pairs(
        source1_df, other_df, ID_COL, NAME_COL, COUNTRY_COL, ID_COL, NAME_COL, COUNTRY_COL
    )
    return build_feature_table(
        pairs, source1_df, other_df,
        ID_COL, NAME_COL, ADDRESS_COL, COUNTRY_COL,
        ID_COL, NAME_COL, ADDRESS_COL, COUNTRY_COL,
    )


def main():
    print("Loading training data...")
    s1 = pd.read_csv(TRAIN_SOURCE1_PATH, sep="\t")
    s2 = pd.read_csv(TRAIN_SOURCE2_PATH, sep="\t")
    s3 = pd.read_csv(TRAIN_SOURCE3_PATH, sep="\t")
    s1_to_s2_gt, s1_to_s3_gt, s1_to_all_gt = load_ground_truth_as_dicts(TRAIN_GROUND_TRUTH_PATH)

    # Split Source 1 ENTITIES (not candidate pairs) so no information about
    # a held-out entity leaks into training.
    train_ids, val_ids = train_test_split(
        s1[ID_COL].tolist(), test_size=VALIDATION_FRACTION, random_state=RANDOM_SEED
    )
    train_s1 = s1[s1[ID_COL].isin(train_ids)].reset_index(drop=True)
    val_s1 = s1[s1[ID_COL].isin(val_ids)].reset_index(drop=True)
    print(f"Split: {len(train_s1)} training entities, {len(val_s1)} validation entities")

    # ---- Train on the training slice ----
    feats_s2 = label_pairs(block_and_featurize(train_s1, s2), s1_to_s2_gt, "id_a", "id_b")
    feats_s3 = label_pairs(block_and_featurize(train_s1, s3), s1_to_s3_gt, "id_a", "id_b")
    combined = pd.concat([feats_s2, feats_s3], ignore_index=True)
    print("Training classifier on the training slice...")
    model = train_classifier(combined)

    # ---- Predict on the held-out validation slice ----
    val_feats_s2 = block_and_featurize(val_s1, s2)
    val_feats_s3 = block_and_featurize(val_s1, s3)

    # Recall ceiling check: did blocking even include the true matches?
    val_ground_truth = {s1_id: s1_to_all_gt.get(s1_id, set()) for s1_id in val_ids}
    candidate_ids_per_entity = {}
    for _, row in pd.concat([val_feats_s2[["id_a", "id_b"]], val_feats_s3[["id_a", "id_b"]]]).iterrows():
        candidate_ids_per_entity.setdefault(row["id_a"], set()).add(row["id_b"])
    missed_recall = 0
    for s1_id, true_ids in val_ground_truth.items():
        if true_ids and not true_ids.issubset(candidate_ids_per_entity.get(s1_id, set())):
            missed_recall += 1
    print(f"Blocking recall check: {missed_recall}/{len(val_ids)} validation entities "
          f"have at least one true match that blocking failed to surface "
          f"(this caps your best possible score regardless of the model).")

    predicted_s2 = predict_matches(model, val_feats_s2, threshold=MATCH_THRESHOLD)
    predicted_s3 = predict_matches(model, val_feats_s3, threshold=MATCH_THRESHOLD)

    predictions = {}
    for _, row in pd.concat([predicted_s2, predicted_s3]).iterrows():
        predictions.setdefault(row["id_a"], set()).add(row["id_b"])

    score = macro_f0_5(predictions, val_ground_truth, val_ids)
    print(f"\n=== Estimated macro F0.5 on held-out validation slice: {score:.4f} ===")
    print("(This should track your real leaderboard score reasonably well --")
    print(" use it to compare thresholds/features before submitting.)")


if __name__ == "__main__":
    main()
