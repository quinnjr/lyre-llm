import numpy as np


def _dilate_time(mask, tolerance_frames):
    """Widen a boolean mask by +/- tolerance_frames along the FRAME axis (axis 0).

    Works for 1-D (T,) and N-D (T, ...) masks. No wrap-around.
    """
    if tolerance_frames <= 0:
        return mask
    out = mask.copy()
    for shift in range(1, int(tolerance_frames) + 1):
        if shift >= mask.shape[0]:
            break
        out[shift:] |= mask[:-shift]
        out[:-shift] |= mask[shift:]
    return out


def _prf(pred, target, tolerance_frames):
    """Precision/recall/F1 with a symmetric frame tolerance.

    A predicted active frame counts as a hit if the target is active within
    +/- tolerance_frames; a target active frame counts as recalled if a
    prediction is active within the same window. With tolerance_frames == 0
    this is exactly the element-wise micro F1.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")
    target_wide = _dilate_time(target, tolerance_frames)
    pred_wide = _dilate_time(pred, tolerance_frames)
    tp_p = int(np.logical_and(pred, target_wide).sum())
    fp = int(np.logical_and(pred, ~target_wide).sum())
    tp_r = int(np.logical_and(target, pred_wide).sum())
    fn = int(np.logical_and(target, ~pred_wide).sum())
    precision = tp_p / (tp_p + fp) if tp_p + fp else 0.0
    recall = tp_r / (tp_r + fn) if tp_r + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def frame_f1(pred, target, threshold=0.5, tolerance_frames=0):
    """Frame-level multi-pitch F1.

    ``pred``/``target`` are (frames, pitches) or (batch, frames, pitches).
    ``tolerance_frames`` dilates along the frame axis (axis 0 for 2-D input,
    which for a batched (B, T, P) array is the batch axis -- pass unbatched
    arrays when using a tolerance).
    """
    pred = np.asarray(pred) >= threshold
    target = np.asarray(target) >= 0.5
    return _prf(pred, target, tolerance_frames)


def onset_f1(pred_onset, target_onset, threshold=0.5, tolerance_frames=0):
    """Onset F1 over the per-frame onset curve.

    Accepts (frames,) or (frames, 1); trailing singleton axes are squeezed so
    the tolerance always applies to the frame axis.
    """
    pred = np.asarray(pred_onset)
    target = np.asarray(target_onset)
    if pred.ndim > 1 and pred.shape[-1] == 1:
        pred = pred[..., 0]
    if target.ndim > 1 and target.shape[-1] == 1:
        target = target[..., 0]
    return _prf(pred >= threshold, target >= 0.5, tolerance_frames)
