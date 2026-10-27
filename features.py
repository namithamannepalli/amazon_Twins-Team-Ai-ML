"""
Feature engineering: turn a pair of records (one from Source 1, one from
Source 2 or 3) into a row of numbers describing how similar they are.

A classifier can't read "Acme Robotics Inc" vs "Acme Robotics Incorporated"
directly — it needs numbers. This module produces those numbers.
"""

import pandas as pd
from rapidfuzz import fuzz
from .normalize import normalize_name, normalize_address, name_tokens, address_tokens, normalize_country


def jaccard(set_a: set, set_b: set) -> float:
    """Overlap between two word sets: 1.0 = identical words, 0.0 = no overlap."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def compute_pair_features(name_a: str, address_a: str, name_b: str, address_b: str,
                           country_a: str = "", country_b: str = "") -> dict:
    """
    Compute similarity features for one pair of records.
    Returns a flat dict of numeric features -- add more of your own here as
    you find patterns in the data (e.g. matching postal codes, matching
    city, numeric street-number similarity).
    """
    norm_name_a, norm_name_b = normalize_name(name_a), normalize_name(name_b)
    norm_addr_a, norm_addr_b = normalize_address(address_a), normalize_address(address_b)

    return {
        # Overall string similarity, robust to word order and small typos
        "name_token_sort_ratio": fuzz.token_sort_ratio(norm_name_a, norm_name_b) / 100.0,
        # Good for when one name is a substring/subset of the other
        "name_partial_ratio": fuzz.partial_ratio(norm_name_a, norm_name_b) / 100.0,
        "address_token_sort_ratio": fuzz.token_sort_ratio(norm_addr_a, norm_addr_b) / 100.0,
        "address_partial_ratio": fuzz.partial_ratio(norm_addr_a, norm_addr_b) / 100.0,
        # Word-overlap based similarity (catches cases character-similarity misses)
        "name_jaccard": jaccard(name_tokens(name_a), name_tokens(name_b)),
        "address_jaccard": jaccard(address_tokens(address_a), address_tokens(address_b)),
        # Simple exact-match flags -- cheap but often surprisingly informative
        "name_exact_match": float(norm_name_a == norm_name_b),
        "address_exact_match": float(norm_addr_a == norm_addr_b),
        # Country should almost always agree for a true match; disagreement
        # is a strong negative signal (works for any country string, not
        # just US/India -- nothing here is hardcoded to specific values).
        "country_exact_match": float(normalize_country(country_a) == normalize_country(country_b)),
    }


def build_feature_table(
    pairs: pd.DataFrame,
    source_a: pd.DataFrame,
    source_b: pd.DataFrame,
    a_id_col: str,
    a_name_col: str,
    a_addr_col: str,
    a_country_col: str,
    b_id_col: str,
    b_name_col: str,
    b_addr_col: str,
    b_country_col: str,
) -> pd.DataFrame:
    """
    Given candidate pairs (as produced by blocking.generate_candidate_pairs,
    with fixed columns "id_a"/"id_b") plus the two source tables, compute
    features for every pair.

    Returns a DataFrame with columns ["id_a", "id_b", <feature columns...>].
    """
    id_cols = ["id_a", "id_b"]
    feature_names = list(compute_pair_features("", "", "", "", "", "").keys())

    # Blocking sometimes legitimately produces zero candidate pairs (e.g. a
    # small/synthetic test slice, or a source with no name overlap at all).
    # Handle that up front so downstream code always gets the expected
    # columns instead of an empty, columnless DataFrame.
    if len(pairs) == 0:
        return pd.DataFrame(columns=id_cols + feature_names)

    a_lookup = source_a.set_index(a_id_col)
    b_lookup = source_b.set_index(b_id_col)

    rows = []
    for _, pair_row in pairs.iterrows():
        a_id, b_id = pair_row["id_a"], pair_row["id_b"]
        a_rec = a_lookup.loc[a_id]
        b_rec = b_lookup.loc[b_id]
        feats = compute_pair_features(
            a_rec[a_name_col], a_rec[a_addr_col],
            b_rec[b_name_col], b_rec[b_addr_col],
            a_rec[a_country_col], b_rec[b_country_col],
        )
        feats["id_a"] = a_id
        feats["id_b"] = b_id
        rows.append(feats)

    feature_df = pd.DataFrame(rows)
    return feature_df[id_cols + feature_names]
