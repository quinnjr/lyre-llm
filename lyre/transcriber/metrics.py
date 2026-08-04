import numpy as np


def frame_f1(pred, target, threshold=0.5):
    pred = np.asarray(pred) >= threshold
    target = np.asarray(target) >= 0.5
    if pred.shape != target.shape:
        raise ValueError("pred/target shape mismatch")
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}