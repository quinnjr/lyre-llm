# SPDX-License-Identifier: Elastic-2.0
# Copyright (c) 2026 Joseph R. Quinn

import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from lyre.config import load_config
from lyre.errors import LyreError
from lyre.reporting import warn
from lyre.scripts._common import device_type, resolve_device
from lyre.transcriber.dataset import StemDataset
from lyre.transcriber.index import load_index
from lyre.transcriber.metrics import frame_f1
from lyre.transcriber.model import MultiPitchNet

# Checkpoint payload format. Bumped whenever the saved keys or the module layout
# they describe change, so a stale file is rejected by name instead of surfacing
# as a torch "Missing key(s) in state_dict" dump that reads like a code bug.
CHECKPOINT_ARCH = 2


def load_checkpoint(path, map_location="cpu"):
    """Load a training checkpoint, refusing anything this build cannot restore.

    Every consumer (train, evaluate, export) goes through here so that an
    incompatible file produces one sentence naming the file and the two format
    numbers, rather than a state-dict key diff from inside torch.
    """
    if not os.path.exists(path):
        raise LyreError(f"checkpoint not found: {path}")
    state = torch.load(path, map_location=map_location)
    if not isinstance(state, dict) or "model" not in state:
        raise LyreError(
            f"{path}: not a lyre checkpoint (expected a dict with a 'model' key)"
        )
    arch = state.get("arch")
    if arch != CHECKPOINT_ARCH:
        raise LyreError(
            f"{path}: checkpoint format {arch!r}, but this build writes and reads "
            f"format {CHECKPOINT_ARCH}. The model architecture changed, so the "
            "weights cannot be loaded; retrain, or check out the revision that "
            "produced this file."
        )
    return state


def _atomic_save(obj, path):
    """Write a checkpoint so an interrupted save cannot leave a truncated file.

    Converter refuses to start without a checkpoint, so a half-written best.pt
    surfaces as a raw traceback from deep inside torch.load. Writing to a
    temporary file and renaming means the visible path is either the previous
    checkpoint or the complete new one.

    The temporary is removed on failure too: on the full filesystem that is the
    usual cause, a leftover .tmp consumes exactly the space the retry needs.
    """
    tmp = path + ".tmp"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _checkpoint_state(model, opt, epoch, step, best_f1):
    """The full training state, so a resume continues rather than restarts.

    ``epoch`` is the index of the epoch that COMPLETED. Saving the optimizer,
    the global step and the best score is what makes a resume equivalent to an
    uninterrupted run: without ``step`` the LR schedule re-enters warmup, and
    without ``best_f1`` the first evaluation after a resume overwrites best.pt
    with a worse model.
    """
    # The scalars are coerced to Python builtins: frame_f1 returns numpy floats,
    # and a numpy scalar in the payload makes the checkpoint unreadable under
    # torch.load's default weights_only=True.
    return {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "best_f1": float(best_f1),
        "arch": CHECKPOINT_ARCH,
    }


def _skip_count(dataset):
    """How many items the dataset skipped, whatever shape it counts them in."""
    value = getattr(dataset, "skipped", 0)
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return len(value)
    except TypeError:
        return 0


def _weighted_sampler(entries, mix_weights, seed, notes=None):
    if not mix_weights:
        return None
    counts = {}
    for e in entries:
        counts[e["source"]] = counts.get(e["source"], 0) + 1
    weights = []
    for e in entries:
        w = mix_weights.get(e["source"], 1.0)
        weights.append(w / counts.get(e["source"], 1))

    # An unmatched key is silent otherwise: the source keeps the 1.0 fallback and
    # trains at several times its intended share, which is exactly how a
    # `piano`/`maestro` key typo went unnoticed.
    for key in sorted(set(mix_weights) - set(counts)):
        warn(
            f"data.mix_weights['{key}'] matches no entry in the index; its weight "
            f"is ignored. Index sources are: {', '.join(sorted(counts)) or '(none)'}",
            notes,
        )
    for source in sorted(set(counts) - set(mix_weights)):
        warn(
            f"index source '{source}' has no data.mix_weights entry, so it samples "
            f"at the fallback weight 1.0 ({counts[source]} entries)",
            notes,
        )
    gen = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights=weights, num_samples=len(entries), generator=gen, replacement=True)


def _validate(model, loader, device, n_notes=128):
    model.eval()
    acc = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}
    with torch.no_grad():
        for mel, pitch, onset, _ in loader:
            mel = mel.unsqueeze(1).to(device)
            logits, _ = model(mel)
            pred = torch.sigmoid(logits).cpu().numpy()
            target = pitch.numpy()
            m = frame_f1(pred, target)
            acc["precision"] += m["precision"]
            acc["recall"] += m["recall"]
            acc["f1"] += m["f1"]
            acc["n"] += 1
    model.train()
    if acc["n"]:
        for k in ("precision", "recall", "f1"):
            acc[k] /= acc["n"]
    return acc


def main(config_path, checkpoint=None, devices="auto", epochs=None, resume=None):
    config = load_config(config_path)
    failures = []
    notes = []
    device = resolve_device(devices)
    dev_type = device_type(device)
    seed = config["run"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    feats = config["features"]
    data = config["data"]
    run_cfg = config["run"]
    out_dir = run_cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    train_entries = load_index(os.path.join(data["index_dir"], data["train_index"]))
    val_entries = load_index(os.path.join(data["index_dir"], data["val_index"]))

    ds_kwargs = dict(
        window_frames=feats["window_frames"],
        n_mels=feats["n_mels"],
        n_fft=feats["n_fft"],
        f_min=feats["f_min"],
        f_max=feats["f_max"],
        n_notes=config["model"]["n_notes"],
        sample_rate=config["audio"]["sample_rate"],
        hop_ms=feats["hop_ms"],
        # Without these the dataset falls back to its own defaults, so the
        # configured pitch range is ignored and training targets disagree with
        # the eval targets built from the same keys.
        min_note=feats["min_note"],
        max_note=feats["max_note"],
    )
    if data.get("cache_bytes") is not None:
        ds_kwargs["cache_bytes"] = data["cache_bytes"]

    # augment= is passed to the train split only. The val split is train=False,
    # which disables augmentation internally; handing it a config would only
    # suggest otherwise. Passing it here is what makes an "augment.enabled:
    # false" ablation actually turn augmentation off.
    train_ds = StemDataset(train_entries, train=True, augment=config.get("augment"), **ds_kwargs)
    val_ds = StemDataset(val_entries, train=False, **ds_kwargs)

    sampler = _weighted_sampler(train_entries, data.get("mix_weights"), seed, notes)
    train_loader = DataLoader(train_ds, batch_size=config["train"]["batch_size"], sampler=sampler, num_workers=data.get("num_workers", 0))
    val_loader = DataLoader(val_ds, batch_size=config["train"]["batch_size"], num_workers=0)

    model = MultiPitchNet(n_mels=feats["n_mels"], n_notes=config["model"]["n_notes"], channels=config["model"]["channels"]).to(device)

    # --checkpoint and --resume are different operations and must not be
    # conflated. --checkpoint seeds a NEW run from someone else's weights (the
    # phase-2 fine-tune), so it starts at epoch 0 under the new schedule.
    # --resume continues THIS run, so it picks up after the last completed epoch
    # and restores the optimizer, the LR step and the best score with it.
    start_epoch = 0
    resumed = False
    resume_step = 0
    resume_best = -1.0
    state = None
    if resume is None:
        resume = config["train"].get("resume")
    if resume:
        state = load_checkpoint(resume, device)
        model.load_state_dict(state["model"])
        start_epoch = int(state.get("epoch", -1)) + 1
        resume_step = int(state.get("step", 0) or 0)
        resume_best = float(state.get("best_f1", -1.0))
        resumed = True
    elif checkpoint:
        state = load_checkpoint(checkpoint, device)
        model.load_state_dict(state["model"])

    opt = torch.optim.AdamW(model.parameters(), lr=config["train"]["lr"], weight_decay=config["train"]["weight_decay"])
    if resumed and state.get("opt"):
        # Adam's moment estimates are part of the training state; dropping them
        # gives the resumed run a few hundred steps of effectively random
        # step sizes on a model that was already converging.
        opt.load_state_dict(state["opt"])
    warmup = config["train"]["warmup_epochs"]
    total_epochs = epochs or config["train"]["epochs"]
    if start_epoch >= total_epochs:
        raise LyreError(
            f"nothing to train: starting at epoch {start_epoch + 1} but train.epochs "
            f"is {total_epochs}. A resumed run is already finished; raise "
            "train.epochs (or --epochs), or use --checkpoint instead of --resume "
            "to fine-tune from these weights under a fresh schedule."
        )
    total_steps = total_epochs * len(train_loader)
    lr = config["train"]["lr"]
    onset_w = config["train"]["loss_onset_weight"]
    grad_clip = config["train"]["grad_clip"]
    amp = bool(config["train"]["amp"]) and dev_type != "cpu"
    scaler = torch.amp.GradScaler(dev_type, enabled=amp)

    bce = nn.BCEWithLogitsLoss()
    log_path = os.path.join(out_dir, "metrics.csv")
    # Truncating on resume destroys the metric history of every epoch trained so
    # far, which is not recoverable from the checkpoints.
    append = resumed and os.path.exists(log_path)
    best_f1 = resume_best

    def lr_at(step):
        # float(), not the numpy scalar np.cos returns: the value is stored in
        # opt.param_groups and lands in the checkpoint, where a numpy scalar
        # makes the file unloadable under torch.load's default weights_only=True.
        if step < warmup * len(train_loader):
            return float((step + 1) / (warmup * len(train_loader)) * lr)
        progress = (step - warmup * len(train_loader)) / max(1, total_steps - warmup * len(train_loader))
        return float(lr * 0.5 * (1 + np.cos(np.pi * progress)))

    model.train()
    step = resume_step
    # `with` rather than a close() at the end: an exception anywhere in the epoch
    # loop would otherwise leak the handle and drop unflushed rows.
    with open(log_path, "a" if append else "w", newline="") as metrics_fh:
        writer = csv.writer(metrics_fh)
        if not append:
            writer.writerow(["epoch", "step", "loss", "val_f1"])
            metrics_fh.flush()
        for epoch in range(start_epoch, total_epochs):
            epoch_loss = 0.0
            for mel, pitch, onset, _ in train_loader:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                mel = mel.unsqueeze(1).to(device)
                pitch = pitch.to(device)
                onset = onset.to(device)
                opt.zero_grad()
                with torch.autocast(device_type=dev_type, enabled=amp):
                    pitch_logits, onset_logits = model(mel)
                    loss = bce(pitch_logits, pitch) + onset_w * bce(onset_logits, onset.unsqueeze(-1))
                scaler.scale(loss).backward()
                # Gradients must be un-scaled before clipping, otherwise grad_clip
                # is applied to fp16-scaled gradients and does nothing meaningful.
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
                epoch_loss += loss.item()
                step += 1
            epoch_loss /= len(train_loader)
            tag = f"epoch {epoch + 1} loss {epoch_loss:.4f}"
            val_f1 = -1.0
            if (epoch + 1) % config["train"]["eval_every"] == 0:
                val = _validate(model, val_loader, device, config["model"]["n_notes"])
                val_f1 = val["f1"]
                tag += f" val_f1 {val_f1:.4f}"
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    _atomic_save(
                        _checkpoint_state(model, opt, epoch, step, best_f1),
                        os.path.join(out_dir, "best.pt"),
                    )
            writer.writerow([epoch + 1, step, round(epoch_loss, 4), round(float(val_f1), 4)])
            metrics_fh.flush()
            print(tag, flush=True)
            _atomic_save(
                _checkpoint_state(model, opt, epoch, step, best_f1),
                os.path.join(out_dir, "last.pt"),
            )

    # A dataset that silently skipped items trained on less than the configured
    # corpus, which is a failed run, not an advisory: the resulting model is not
    # the one the config describes.
    skipped = _skip_count(train_ds) + _skip_count(val_ds)
    if skipped:
        warn(
            f"{skipped} dataset item(s) were skipped because their audio or MIDI "
            "could not be read; the run did not train on the full index",
            failures,
        )

    print(f"done -> {out_dir} (best val_f1 {best_f1:.4f})")
    return {
        "out_dir": out_dir,
        "best_f1": best_f1,
        "epochs": total_epochs,
        "start_epoch": start_epoch,
        "skipped": skipped,
        "failures": failures,
        "notes": notes,
    }
