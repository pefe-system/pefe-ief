import numpy as np
from numpy import ndarray

def _stats_with_threshold(probs, labels, threshold):
    # type: (ndarray, ndarray, float) -> tuple[int, int,    int, int,    int,     int, int]
    frame_output = (probs >= threshold).astype(int)

    hits = np.sum(frame_output == labels)
    misses = np.sum(frame_output != labels)
    tp = np.sum((frame_output == 1) & (labels == 1))
    fp = np.sum((frame_output == 1) & (labels == 0))
    fn = np.sum((frame_output == 0) & (labels == 1))
    tn = np.sum((frame_output == 0) & (labels == 0))

    total_count = len(probs)
    num_labelled_1 = np.sum((labels == 1))
    num_labelled_0 = np.sum((labels == 0))

    result = (
        hits, misses,
        num_labelled_1, num_labelled_0,
        total_count,
        tp, tn, fp, fn
    )

    try:
        assert hits + misses == total_count
        assert tp + fn == num_labelled_1
        assert tn + fp == num_labelled_0
        assert num_labelled_1 + num_labelled_0 == total_count

    except AssertionError as e:
        raise Exception(
            result,
            e
        )
    
    return result

def stats_per_thresholds(probs, labels, thresholds):
    # type: (ndarray, ndarray, list[float]) -> dict[str, dict[str, int|float]]

    stats = []

    TOTAL_COUNT = 0
    TOTAL_TRUE = 0
    TOTAL_FALSE = 0

    for current_threshold in thresholds:
        (
            hits, misses,
            num_true, num_false,
            total_count,
            tp, tn, fp, fn
        ) = _stats_with_threshold(probs, labels, current_threshold)

        if TOTAL_COUNT == 0:
            assert TOTAL_TRUE == 0
            assert TOTAL_FALSE == 0
            TOTAL_COUNT = total_count
            TOTAL_TRUE = num_true
            TOTAL_FALSE = num_false
        else:
            assert TOTAL_COUNT == total_count
            assert TOTAL_TRUE == num_true
            assert TOTAL_FALSE == num_false

        accuracy = hits / total_count if total_count > 0 else 0.0
        precision = tp / num_true if num_true > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        stat = {
            "threshold": current_threshold,
            "total_hits": hits,
            "total_misses": misses,
            "accuracy": accuracy,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        stats.append(stat)

    return {
        "dataset": {
            "total_count": TOTAL_COUNT,
            "malware_count": TOTAL_TRUE,
            "benign_count": TOTAL_FALSE,
        },

        "stats_per_thresholds": stats,
    }
