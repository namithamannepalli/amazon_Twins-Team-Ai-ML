# Business Entity Resolution (Amazon ML Challenge 2026)

Matching business records that describe the same real-world company across
three independent data sources that share **no common identifier** — using only
the noisy `business_name`, `business_address` and `country` fields.

> **Status:** working baseline (blocking → similarity features → classifier →
> thresholded matches). Validated on a sample of the real training data.
> The row-by-row implementation is **not yet optimised for the full ~2–5 M
> rows per source file** (see *Limitations*).

## The problem

- Source 1 is a deduplicated reference list; Sources 2 and 3 are noisy vendor
  feeds. For every Source 1 entity, find all matching Source 2/3 records
  (zero, one or many).
- Noise: abbreviations (Pvt/Private, Rd/Road), legal-suffix differences,
  typos, word-order changes, missing address parts, and **non-Latin scripts**
  (~28 % of Indian names in Source 2 were written in Devanagari/Tamil etc.
  while Source 1 is English).
- Metric: macro **F0.5** per Source 1 entity (precision weighted 2×; a correct
  empty prediction for a singleton scores 1.0).
- The test set contains a country (France) never seen in training, so no
  logic is hard-coded to specific country values.

## Approach

| Stage | What it does | Code |
|---|---|---|
| Normalisation | Transliterates non-Latin scripts with `unidecode`, lowercases, strips punctuation and legal suffixes, expands street abbreviations | `src/normalize.py` |
| Blocking | Candidates = same country **and** a shared significant name token (inverted index). Keeps the candidate set small instead of comparing all pairs | `src/blocking.py` |
| Features | Token-sort / partial ratio and Jaccard on name and address, exact-match flags, country match (RapidFuzz) | `src/features.py` |
| Model | scikit-learn `RandomForestClassifier` (class-balanced) trained on labelled candidate pairs | `src/train_model.py` |
| Decision | Probability threshold (default 0.5, tunable — precision-weighted metric favours higher) | `src/predict.py` |
| Evaluation | Implementation of the official macro F0.5 on a held-out validation split | `src/scoring.py`, `run_validation.py` |

No external data, APIs or geocoding services are used, and no pretrained
foundation model — only scikit-learn and string-similarity libraries
(MIT/BSD licensed).

## Results

Measured by `run_validation.py` on a **sample** (3,000 Source 1 entities, their
true matches plus ~0.3 % random distractors from Sources 2/3), 80/20 split by
Source 1 entity:

| Metric | Value |
|---|---|
| Pair-classifier precision / recall (match class) | 0.955 / 0.982 |
| Macro F0.5 (validation slice) | 0.908 |
| Entities where blocking missed ≥ 1 true match | 215 / 600 |

**Read these numbers carefully.** The sample contains far fewer distractor
records than the real data, so negatives are much easier than at full scale and
the F0.5 above is optimistic. The blocking-recall figure shows the main weakness:
a shared-name-token rule loses true matches whose names differ a lot
(e.g. transliterated or DBA names). Full-scale and leaderboard results:
`TODO — fill in after the full run.`

## Quick start

```bash
pip install -r requirements.txt

# 1. (optional) build a small sample from the full training files
python scripts/make_sample.py --raw /path/to/raw_tsvs --out sample/dataset/train

# 2. estimate your score on a held-out split
ER_DATA_DIR=sample/dataset python run_validation.py

# 3. generate submission files (expects dataset/train and dataset/test)
python run_pipeline.py

# 4. check the output format
python scripts/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Data files are **not** included (they belong to the challenge organiser).
Expected layout: `dataset/train/train_source{1,2,3}.tsv`,
`dataset/train/train_ground_truth.tsv`, `dataset/test/test_source{1,2,3}.tsv`.

## Limitations / next steps

- **Scale:** blocking and feature extraction use Python loops (`iterrows`),
  fine for sample-size data but too slow for millions of rows. Next step:
  vectorised/chunked processing with an integer-coded inverted index and
  frequency-capped tokens.
- **Blocking recall:** add address-based blocking keys and phonetic/character
  n-gram name keys to recover matches with dissimilar names.
- **Features:** PIN/postal-code agreement, city match, acronym matching,
  TF-IDF cosine; try gradient boosting.
- **Threshold tuning** for F0.5 on a larger validation set.

## Layout

```
├── run_pipeline.py        # generate matching_results.tsv + candidate_pairs.tsv
├── run_validation.py      # local macro-F0.5 estimate on a held-out split
├── requirements.txt
├── src/                   # normalize, blocking, features, train_model, predict, scoring
├── scripts/               # make_sample.py, validate_submission.py (organiser's validator)
└── docs/METHODOLOGY.md    # methodology write-up
```

## License

MIT — see `LICENSE`.
