"""
Implements the official macro F_0.5 metric so you can score yourself on a
held-out validation split -- the challenge gives you NO ground truth for the
real test set, so this is the only way to know if a change actually helped
before spending a leaderboard submission on it.

F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
Computed PER Source 1 entity, then averaged across all entities (macro).
Singletons (true empty match list) score 1.0 if predicted empty, 0.0 if
you predict anything for them.
"""


def f0_5_for_one_entity(predicted_ids: set, true_ids: set) -> float:
    """F0.5 for a single Source 1 entity's predicted vs true match set."""
    if len(true_ids) == 0:
        # Singleton: full credit for an empty prediction, zero for any guess.
        return 1.0 if len(predicted_ids) == 0 else 0.0

    if len(predicted_ids) == 0:
        # Missed every true match: precision is undefined, recall is 0 -> F is 0.
        return 0.0

    true_positives = len(predicted_ids & true_ids)
    precision = true_positives / len(predicted_ids)
    recall = true_positives / len(true_ids)

    if precision == 0 and recall == 0:
        return 0.0

    beta_sq = 0.5 ** 2
    denominator = (beta_sq * precision) + recall
    if denominator == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denominator


def macro_f0_5(predictions: dict, ground_truth: dict, all_source1_ids: list) -> float:
    """
    predictions / ground_truth: {source1_entity_id: set_of_matched_ids}
    all_source1_ids: every Source 1 id that must be scored (missing keys in
    either dict are treated as an empty prediction/truth set).
    Returns the macro-averaged F0.5 across all_source1_ids.
    """
    scores = []
    for s1_id in all_source1_ids:
        predicted = predictions.get(s1_id, set())
        true = ground_truth.get(s1_id, set())
        scores.append(f0_5_for_one_entity(predicted, true))
    return sum(scores) / len(scores) if scores else 0.0
