# ML Challenge 2026: Business Entity Resolution — Methodology

**Team Name:** <YOUR TEAM NAME>
**Team Members:** <YOUR NAME(S)>
**Submission Date:** <DATE>

## 1. Executive Summary
A blocking + classifier pipeline. Records are normalised (including
transliteration of non-Latin scripts), a country-and-shared-name-token blocking
step produces a small candidate set per Source 1 entity, string-similarity
features are computed per candidate pair, and a Random Forest decides which
candidates are true matches under a precision-favouring threshold.

## 2. Methodology

### 2.1 Problem Analysis
- Training data: ~2.2 M Source 1 entities; ~5 M records each in Sources 2 and 3;
  countries US and India. Most Source 1 entities have 2–6 matches; ~5.6 % are
  singletons (no match).
- Source 1 names are always Latin script; ~28 % of Indian Source 2 names and a
  similar share in Source 3 are in native scripts (Devanagari, Tamil, ...).
- Test data adds a third country (France) unseen in training.
- Addresses vary in format and completeness; some are missing.

### 2.2 Solution Strategy
**Approach type:** Blocking + classifier.
**Core ideas:** transliteration before matching; country as a generic
(non-hard-coded) blocking key; precision-weighted decision threshold.

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** (country, significant name token), tokens ≥ 3
  characters after normalisation and legal-suffix removal.
- **Candidate pairs generated:** <FILL IN from full run>
- **Protecting recall:** token-level (rather than whole-name) keys tolerate
  word-order changes and extra words. Measured weakness: matches with very
  different names are missed (see §5); planned fix is additional address and
  phonetic/n-gram keys.

## 4. Matching Model
**Features**
- Name: token-sort ratio, partial ratio, Jaccard of tokens, exact match.
- Address: token-sort ratio, partial ratio, Jaccard of tokens, exact match.
- Other: country equality.

**Model type:** scikit-learn RandomForestClassifier (300 trees, depth 8,
balanced class weights). MIT/BSD-licensed, far below the 8 B-parameter limit,
no pretrained model.
**Threshold selection:** default 0.5; to be tuned for macro F0.5 on validation.

## 5. Results & Error Analysis
- **Macro F0.5 (sample validation):** 0.908 (3,000-entity sample with sparse
  distractors — optimistic relative to the full data).
- **Full-scale / leaderboard F0.5:** <FILL IN>
- **Blocking recall:** 215 of 600 validation entities had ≥ 1 true match not
  surfaced by blocking.
- **Common false positives:** <FILL IN after error analysis>
- **Common false negatives:** <FILL IN after error analysis>

## 6. Conclusion
<WRITE 2–3 SENTENCES after the final run.>

## Appendix
### A. Code Artefacts
Entry points: `run_validation.py` (local scoring), `run_pipeline.py` (generates
`output/matching_results.tsv` and `output/candidate_pairs.tsv`). Modules live in
`src/`.

### B. Fair play
No external databases, APIs or geocoding services were used; transliteration
uses the offline `unidecode` character-mapping library.
