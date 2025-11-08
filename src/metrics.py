from __future__ import annotations
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

def threshold_scores(scores: np.ndarray, thr: float):
    return (scores >= thr).astype(int)

def evaluate_pointwise(y_true, scores, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    y_pred = threshold_scores(scores, thr)

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        roc = roc_auc_score(y_true, scores)
    except ValueError:
        roc = float("nan")
    try:
        ap = average_precision_score(y_true, scores)
    except ValueError:
        ap = float("nan")

    far = (((y_pred==1) & (y_true==0))).sum() / max((y_true==0).sum(), 1)
    return {"precision":float(p), "recall":float(r), "f1":float(f1), "roc_auc":float(roc), "pr_auc":float(ap), "far":float(far)}

def interval_hits(y_true, y_pred):
    # y_true/pred are binary sequences. Each contiguous block of 1s is an event.
    def blocks(y):
        y = np.asarray(y).astype(int)
        edges = np.diff(np.r_[0, y, 0])
        starts = np.where(edges == 1)[0]
        ends   = np.where(edges == -1)[0] - 1
        return list(zip(starts, ends))
    GT = blocks(y_true); PD = blocks(y_pred)
    hits = 0
    for s,e in GT:
        if any(not (pe < s or ps > e) for ps,pe in PD):
            hits += 1
    return hits, len(GT), len(PD)
