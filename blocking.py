"""
Blocking: cutting down a huge number of possible comparisons to a small,
precise shortlist per Source 1 entity, without throwing away true matches.

IMPORTANT (per the official rules): candidate_pairs.tsv is now part of your
ranking, and a SMALLER candidate set per Source 1 entity scores better, as
long as recall doesn't suffer. So this isn't just "cast a wide net" blocking
-- it's a balance between staying small and not losing true matches.

Two blocking signals are combined here:

1. Country match -- a business's country shouldn't change across sources
   (US, India, France, whatever it is), and the problem statement doesn't
   flag "country" as a noisy field the way name/address are. Filtering to
   same-country pairs alone throws away a huge fraction of irrelevant
   candidates for free. This works generically for ANY country string,
   including "France" appearing only in the test set -- nothing is
   hardcoded to specific country values.
2. Shared significant word in the (normalized) business name -- classic
   token blocking, restricted to same-country pairs.

If you find recall is suffering (true matches are being blocked out),
the most likely cause is noisy/inconsistent country labels -- check that
before loosening the token-matching side.
"""

from collections import defaultdict
import pandas as pd
from .normalize import name_tokens, normalize_country

MIN_TOKEN_LENGTH = 3  # ignore very short/common words as blocking keys


def build_inverted_index(df: pd.DataFrame, id_col: str, name_col: str, country_col: str) -> dict:
    """
    Build a {(country, word): [ids]} index. Keying by (country, word)
    instead of just word is what keeps candidate sets small -- two
    businesses in different countries never end up compared at all.
    """
    index = defaultdict(list)
    for _, row in df.iterrows():
        country = normalize_country(row[country_col])
        tokens = name_tokens(row[name_col])
        for token in tokens:
            if len(token) >= MIN_TOKEN_LENGTH:
                index[(country, token)].append(row[id_col])
    return index


def generate_candidate_pairs(
    source_a: pd.DataFrame,
    source_b: pd.DataFrame,
    a_id_col: str,
    a_name_col: str,
    a_country_col: str,
    b_id_col: str,
    b_name_col: str,
    b_country_col: str,
) -> pd.DataFrame:
    """
    Given two sources, return every (a_id, b_id) pair that is in the same
    country AND shares at least one significant name token.

    Returns a DataFrame with fixed columns ["id_a", "id_b"], regardless of
    what the id columns are actually called in your source files.
    """
    index_b = build_inverted_index(source_b, b_id_col, b_name_col, b_country_col)

    pairs = set()
    for _, row in source_a.iterrows():
        a_id = row[a_id_col]
        country = normalize_country(row[a_country_col])
        tokens = name_tokens(row[a_name_col])
        candidate_b_ids = set()
        for token in tokens:
            if len(token) >= MIN_TOKEN_LENGTH:
                candidate_b_ids.update(index_b.get((country, token), []))
        for b_id in candidate_b_ids:
            pairs.add((a_id, b_id))

    return pd.DataFrame(sorted(pairs), columns=["id_a", "id_b"])


def group_ids_by_source1(pairs_df: pd.DataFrame, all_source1_ids: list, output_col_name: str) -> pd.DataFrame:
    """
    Turn a flat (id_a, id_b) pairs table into the challenge's required
    one-row-per-Source-1-entity shape: source1_entity_id, comma-separated
    id list. Used for BOTH matching_results.tsv (output_col_name=
    "matched_entity_ids") and candidate_pairs.tsv (output_col_name=
    "candidate_entity_ids"), since the two files share the same shape.

    all_source1_ids must include every Source 1 test entity, even ones with
    zero pairs -- they still need a row with an empty string.
    """
    if len(pairs_df) > 0:
        grouped = pairs_df.groupby("id_a")["id_b"].apply(lambda ids: sorted(set(ids)))
    else:
        grouped = pd.Series(dtype=object)

    rows = []
    for s1_id in all_source1_ids:
        matched_ids = grouped.get(s1_id, [])
        rows.append({
            "source1_entity_id": s1_id,
            output_col_name: ",".join(matched_ids),
        })
    return pd.DataFrame(rows)
