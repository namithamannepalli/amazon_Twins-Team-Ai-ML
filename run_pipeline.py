"""
End-to-end pipeline: blocking -> features -> train -> predict -> submission files.

Matches the OFFICIAL Amazon ML Challenge 2026 "Business Entity Resolution"
spec: real column names (entity_id, business_name, business_address,
country), real ground-truth format, and both required output files
(matching_results.tsv AND candidate_pairs.tsv, in the same one-row-per-
Source-1-entity shape).

HOW TO USE
----------
1. Put the challenge's dataset/ folder (as downloaded) next to this script,
   or edit DATA_DIR below to point at it.
2. Run:  python run_pipeline.py
3. Check output/matching_results.tsv and output/candidate_pairs.tsv, then
   run the official scripts/validate_submission.py on them before submitting.
"""

import os
import pandas as pd
from src.blocking import generate_candidate_pairs, group_ids_by_source1
from src.features import build_feature_table
from src.train_model import label_pairs, train_classifier, save_model
from src.predict import predict_matches, save_tsv

# ============================== CONFIG ======================================
DATA_DIR = os.environ.get("ER_DATA_DIR", "dataset")

TRAIN_SOURCE1_PATH = os.path.join(DATA_DIR, "train", "train_source1.tsv")
TRAIN_SOURCE2_PATH = os.path.join(DATA_DIR, "train", "train_source2.tsv")
TRAIN_SOURCE3_PATH = os.path.join(DATA_DIR, "train", "train_source3.tsv")
TRAIN_GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")

TEST_SOURCE1_PATH = os.path.join(DATA_DIR, "test", "test_source1.tsv")
TEST_SOURCE2_PATH = os.path.join(DATA_DIR, "test", "test_source2.tsv")
TEST_SOURCE3_PATH = os.path.join(DATA_DIR, "test", "test_source3.tsv")

OUTPUT_DIR = "output"
CANDIDATE_PAIRS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
MATCHING_RESULTS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "matching_results.tsv")

# Official column names -- these match the published spec exactly.
ID_COL = "entity_id"
NAME_COL = "business_name"
ADDRESS_COL = "business_address"
COUNTRY_COL = "country"

GT_SOURCE1_ID_COL = "source1_entity_id"
GT_MATCHES_COL = "matched_entity_ids"
SOURCE2_ID_PREFIX = "S2"
SOURCE3_ID_PREFIX = "S3"

MATCH_PROBABILITY_THRESHOLD = 0.5
# =============================================================================


def load_ground_truth_as_dicts(path: str):
    """
    Parse train_ground_truth.tsv into two lookup dicts:
      s1_to_s2_matches: {source1_entity_id: {source2_entity_id, ...}}
      s1_to_s3_matches: {source1_entity_id: {source3_entity_id, ...}}
    """
    gt = pd.read_csv(path, sep="\t").fillna("")
    s1_to_s2, s1_to_s3 = {}, {}

    for _, row in gt.iterrows():
        s1_id = row[GT_SOURCE1_ID_COL]
        all_matches = [m.strip() for m in str(row[GT_MATCHES_COL]).split(",") if m.strip()]
        s1_to_s2[s1_id] = {m for m in all_matches if m.startswith(SOURCE2_ID_PREFIX)}
        s1_to_s3[s1_id] = {m for m in all_matches if m.startswith(SOURCE3_ID_PREFIX)}

    return s1_to_s2, s1_to_s3


def block_and_featurize(source1_df, other_df):
    """Shared blocking + feature step for Source1<->Source2 or Source1<->Source3."""
    pairs = generate_candidate_pairs(
        source1_df, other_df,
        ID_COL, NAME_COL, COUNTRY_COL,
        ID_COL, NAME_COL, COUNTRY_COL,
    )
    features = build_feature_table(
        pairs, source1_df, other_df,
        ID_COL, NAME_COL, ADDRESS_COL, COUNTRY_COL,
        ID_COL, NAME_COL, ADDRESS_COL, COUNTRY_COL,
    )
    return features


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- 1. Load training data ----
    print("Loading training data...")
    train_s1 = pd.read_csv(TRAIN_SOURCE1_PATH, sep="\t")
    train_s2 = pd.read_csv(TRAIN_SOURCE2_PATH, sep="\t")
    train_s3 = pd.read_csv(TRAIN_SOURCE3_PATH, sep="\t")
    s1_to_s2_gt, s1_to_s3_gt = load_ground_truth_as_dicts(TRAIN_GROUND_TRUTH_PATH)

    # ---- 2. Block + featurize + label training pairs ----
    print("Building Source1<->Source2 training pairs...")
    feats_s2 = block_and_featurize(train_s1, train_s2)
    print(f"  {len(feats_s2)} candidate pairs")
    feats_s2 = label_pairs(feats_s2, s1_to_s2_gt, "id_a", "id_b")

    print("Building Source1<->Source3 training pairs...")
    feats_s3 = block_and_featurize(train_s1, train_s3)
    print(f"  {len(feats_s3)} candidate pairs")
    feats_s3 = label_pairs(feats_s3, s1_to_s3_gt, "id_a", "id_b")

    # ---- 3. Train one shared classifier on both sets of pairs combined ----
    # Features are symmetric regardless of whether the "other" record came
    # from Source 2 or Source 3, so one model handles both.
    print("Training classifier...")
    combined_training_rows = pd.concat([feats_s2, feats_s3], ignore_index=True)
    n_positive = combined_training_rows["label"].sum()
    n_total = len(combined_training_rows)
    print(f"  Training pairs: {n_total} total, {n_positive} true matches "
          f"({100 * n_positive / max(n_total, 1):.2f}%)")
    model = train_classifier(combined_training_rows)
    save_model(model, os.path.join(OUTPUT_DIR, "model.joblib"))

    # ---- 4. Load test data and repeat blocking + features (no labels) ----
    print("Loading test data and generating test candidate pairs...")
    test_s1 = pd.read_csv(TEST_SOURCE1_PATH, sep="\t")
    test_s2 = pd.read_csv(TEST_SOURCE2_PATH, sep="\t")
    test_s3 = pd.read_csv(TEST_SOURCE3_PATH, sep="\t")

    test_feats_s2 = block_and_featurize(test_s1, test_s2)
    test_feats_s3 = block_and_featurize(test_s1, test_s3)
    print(f"  Test candidate pairs: {len(test_feats_s2)} vs Source2, {len(test_feats_s3)} vs Source3")

    # ---- 5. Write candidate_pairs.tsv -- the EXACT set fed to the model ----
    all_test_s1_ids = test_s1[ID_COL].tolist()
    all_candidate_pairs = pd.concat([
        test_feats_s2[["id_a", "id_b"]],
        test_feats_s3[["id_a", "id_b"]],
    ], ignore_index=True)
    candidate_pairs_output = group_ids_by_source1(
        all_candidate_pairs, all_test_s1_ids, "candidate_entity_ids"
    )
    save_tsv(candidate_pairs_output, CANDIDATE_PAIRS_OUTPUT_PATH)

    # ---- 6. Predict matches on the test set ----
    print("Scoring test candidate pairs...")
    predicted_s2 = predict_matches(model, test_feats_s2, threshold=MATCH_PROBABILITY_THRESHOLD)
    predicted_s3 = predict_matches(model, test_feats_s3, threshold=MATCH_PROBABILITY_THRESHOLD)

    # ---- 7. Write matching_results.tsv ----
    all_predicted_matches = pd.concat([predicted_s2, predicted_s3], ignore_index=True)
    matching_results = group_ids_by_source1(
        all_predicted_matches, all_test_s1_ids, "matched_entity_ids"
    )
    save_tsv(matching_results, MATCHING_RESULTS_OUTPUT_PATH)

    print("\nDone. Now run the official validator, e.g.:")
    print("  python3 scripts/validate_submission.py \\")
    print(f"      --matching {MATCHING_RESULTS_OUTPUT_PATH} \\")
    print(f"      --candidate {CANDIDATE_PAIRS_OUTPUT_PATH} \\")
    print(f"      --test-dir {os.path.join(DATA_DIR, 'test')}")


if __name__ == "__main__":
    main()
