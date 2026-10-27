"""
Build a small sample of the training data so the pipeline can be developed
and tested quickly (the full files are millions of rows).

Picks N random Source 1 entities, keeps ALL of their true Source 2/3 matches
from the ground truth, and adds random "distractor" records from Source 2/3
so blocking and the classifier see realistic negatives.

Usage:
    python scripts/make_sample.py --raw /path/to/dataset_files --out sample/dataset/train \
        --n-entities 5000 --distractor-rate 0.004
"""

import argparse
import os
import zlib

import pandas as pd


def keep_by_hash(entity_id: str, rate: float) -> bool:
    # Deterministic pseudo-random selection (no RNG state needed while streaming).
    return (zlib.crc32(entity_id.encode()) % 1_000_000) < rate * 1_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder containing train_source*.tsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-entities", type=int, default=5000)
    ap.add_argument("--distractor-rate", type=float, default=0.004)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gt = pd.read_csv(os.path.join(args.raw, "train_ground_truth.tsv"),
                     sep="\t", dtype=str, keep_default_na=False)
    gt = gt.sample(n=min(args.n_entities, len(gt)), random_state=args.seed)
    gt.to_csv(os.path.join(args.out, "train_ground_truth.tsv"), sep="\t", index=False)

    s1_ids = set(gt["source1_entity_id"])
    matched = {m for lst in gt["matched_entity_ids"] for m in lst.split(",") if m}

    for name, wanted in (("train_source1.tsv", s1_ids),
                         ("train_source2.tsv", matched),
                         ("train_source3.tsv", matched)):
        parts = []
        for chunk in pd.read_csv(os.path.join(args.raw, name), sep="\t", dtype=str,
                                 keep_default_na=False, chunksize=500_000):
            if name == "train_source1.tsv":
                mask = chunk["entity_id"].isin(wanted)
            else:
                mask = chunk["entity_id"].isin(wanted) | chunk["entity_id"].map(
                    lambda x: keep_by_hash(x, args.distractor_rate))
            parts.append(chunk[mask])
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(os.path.join(args.out, name), sep="\t", index=False)
        print(f"{name}: {len(df):,} rows")


if __name__ == "__main__":
    main()
