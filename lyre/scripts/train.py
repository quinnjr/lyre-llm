import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml

from lyre.transcriber.dataset import StemDataset
from lyre.transcriber.index import load_index
from lyre.transcriber.metrics import frame_f1
from lyre.transcriber.model import MultiPitchNet


def _device_from(devices):
    if devices in ("auto", None):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return devices


def _weighted_sampler(entries, mix_weights, seed):
    if not mix_weights:
        return None
    counts = {}
    for e in entries:
        counts[e["source"]] = counts.get(e["source"], 0) + 1
    weights = []
    for e in entries:
        w = mix_weights.get(e["source"], 1.0)
        weights.append(w / counts.get(e["source"], 1))
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
    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    device = _device_from(devices)
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

    train_ds = StemDataset(train_entries, window_frames=feats["window_frames"], n_mels=feats["n_mels"], n_notes=config["model"]["n_notes"], train=True)
    val_ds = StemDataset(val_entries, window_frames=feats["window_frames"], n_mels=feats["n_mels"], n_notes=config["model"]["n_notes"], train=False)

    sampler = _weighted_sampler(train_entries, data.get("mix_weights"), seed)
    train_loader = DataLoader(train_ds, batch_size=config["train"]["batch_size"], sampler=sampler, num_workers=data.get("num_workers", 0))
    val_loader = DataLoader(val_ds, batch_size=config["train"]["batch_size"], num_workers=0)

    model = MultiPitchNet(n_mels=feats["n_mels"], n_notes=config["model"]["n_notes"], channels=config["model"]["channels"]).to(device)
    start_epoch = 0
    if resume or (checkpoint and os.path.exists(checkpoint)):
        src = resume or checkpoint
        state = torch.load(src, map_location=device)
        model.load_state_dict(state["model"])
        start_epoch = state.get("epoch", 0) or 0

    opt = torch.optim.AdamW(model.parameters(), lr=config["train"]["lr"], weight_decay=config["train"]["weight_decay"])
    warmup = config["train"]["warmup_epochs"]
    total_epochs = epochs or config["train"]["epochs"]
    total_steps = total_epochs * len(train_loader)
    lr = config["train"]["lr"]
    onset_w = config["train"]["loss_onset_weight"]
    grad_clip = config["train"]["grad_clip"]
    amp = config["train"]["amp"] and device != "cpu"

    bce = nn.BCEWithLogitsLoss()
    log_path = os.path.join(out_dir, "metrics.csv")
    metrics_fh = open(log_path, "w", newline="")
    writer = csv.writer(metrics_fh)
    writer.writerow(["epoch", "step", "loss", "val_f1"])
    best_f1 = -1.0

    def lr_at(step):
        if step < warmup * len(train_loader):
            return (step + 1) / (warmup * len(train_loader)) * lr
        progress = (step - warmup * len(train_loader)) / max(1, total_steps - warmup * len(train_loader))
        return lr * 0.5 * (1 + np.cos(np.pi * progress))

    model.train()
    step = 0
    for epoch in range(start_epoch, total_epochs):
        epoch_loss = 0.0
        for mel, pitch, onset, _ in train_loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            mel = mel.unsqueeze(1).to(device)
            pitch = pitch.to(device)
            onset = onset.to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=amp):
                pitch_logits, onset_logits = model(mel)
                loss = bce(pitch_logits, pitch) + onset_w * bce(onset_logits, onset.unsqueeze(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
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
                torch.save({"model": model.state_dict(), "epoch": epoch}, os.path.join(out_dir, "best.pt"))
        writer.writerow([epoch + 1, step, round(epoch_loss, 4), round(float(val_f1), 4)])
        metrics_fh.flush()
        print(tag, flush=True)
        torch.save({"model": model.state_dict(), "epoch": epoch}, os.path.join(out_dir, "last.pt"))
    metrics_fh.close()
    print(f"done -> {out_dir} (best val_f1 {best_f1:.4f})")