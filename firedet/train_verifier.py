"""Verifier training: EfficientNet-B0 (ImageNet-pretrained) on crops fire / smoke / not_fire.
Prints per-epoch train + validation loss/accuracy, progress %, ETA, and plots curves."""
import json
import math
import time
from pathlib import Path

import numpy as np

from .common import fmt_secs


def plot_verifier_curves(history, out_png=None, show=True):
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(ep, [h["train_loss"] for h in history], "o-", label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], "s-", label="val")
    ax[0].set_title("loss")
    ax[1].plot(ep, [h["train_acc"] for h in history], "o-", label="train")
    ax[1].plot(ep, [h["val_acc"] for h in history], "s-", label="val")
    ax[1].set_title("accuracy")
    ax[2].plot(ep, [h["val_event_recall"] for h in history], "o-", label="fire+smoke kept (recall)")
    ax[2].plot(ep, [h["val_notfire_rejected"] for h in history], "s-", label="false positives rejected")
    ax[2].set_title("val @ p_event >= 0.5")
    ax[2].set_ylim(0, 1.02)
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=110)
    if show:
        plt.show()
    plt.close(fig)


def train_verifier(crop_dir, out_weights, epochs=8, batch=128, lr=5e-4, imgsz=224, workers=8, device="cuda:0",
                   pad=0.3, label_smoothing=0.05, show_plots=True, seed=0):
    import torch
    import torchvision
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    from tqdm.auto import tqdm

    torch.manual_seed(seed)
    crop_dir = Path(crop_dir)
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf_train = T.Compose([
        T.RandomResizedCrop(imgsz, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.35, 0.35, 0.25, 0.02)], p=0.8),
        T.RandomGrayscale(0.05),
        T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.2),
        T.ToTensor(), norm])
    tf_val = T.Compose([T.Resize((imgsz, imgsz)), T.ToTensor(), norm])
    ds_tr = torchvision.datasets.ImageFolder(crop_dir / "train", tf_train)
    ds_va = torchvision.datasets.ImageFolder(crop_dir / "val", tf_val)
    if ds_tr.classes != ds_va.classes:
        raise ValueError(f"class folders differ: {ds_tr.classes} vs {ds_va.classes}")
    classes = ds_tr.classes
    counts = np.bincount([y for _, y in ds_tr.samples], minlength=len(classes))
    print(f"Verifier classes {classes} | train {dict(zip(classes, counts.tolist()))} | val {len(ds_va):,} crops")
    dl_tr = DataLoader(ds_tr, batch, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True,
                       persistent_workers=workers > 0)
    dl_va = DataLoader(ds_va, batch, shuffle=False, num_workers=workers, pin_memory=True)

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    try:
        model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not download ImageNet weights ({type(e).__name__}) - training from scratch")
        model = torchvision.models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.to(dev)
    w = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(classes), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)
    crit_val = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * len(dl_tr)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    use_amp = dev.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    ev_idx = [i for i, c in enumerate(classes) if c != "not_fire"]
    nf_idx = classes.index("not_fire") if "not_fire" in classes else None

    history, best = [], float("inf")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        te = time.time()
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for x, y in tqdm(dl_tr, desc=f"epoch {ep}/{epochs} train", leave=False, mininterval=2):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=use_amp):
                out = model(x)
                loss = crit(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * len(y)
            tc += (out.argmax(1) == y).sum().item()
            tn += len(y)
        model.eval()
        vl, vn, probs, ys = 0.0, 0, [], []
        with torch.inference_mode():
            for x, y in tqdm(dl_va, desc=f"epoch {ep}/{epochs} val", leave=False, mininterval=2):
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                with torch.autocast(dev.type, dtype=amp_dtype, enabled=use_amp):
                    out = model(x)
                vl += crit_val(out.float(), y).item() * len(y)
                vn += len(y)
                probs.append(torch.softmax(out.float(), 1).cpu().numpy())
                ys.append(y.cpu().numpy())
        P, Y = np.concatenate(probs), np.concatenate(ys)
        pred = P.argmax(1)
        pe = P[:, ev_idx].sum(1)
        is_ev = np.isin(Y, ev_idx)
        h = {"epoch": ep, "train_loss": tl / tn, "train_acc": tc / tn, "val_loss": vl / vn,
             "val_acc": float((pred == Y).mean()),
             "val_event_recall": float((pe[is_ev] >= 0.5).mean()) if is_ev.any() else float("nan"),
             "val_notfire_rejected": float((pe[~is_ev] < 0.5).mean()) if (~is_ev).any() else float("nan"),
             "per_class_recall": {c: float((pred[Y == i] == i).mean()) if (Y == i).any() else None for i, c in enumerate(classes)},
             "lr": opt.param_groups[0]["lr"], "epoch_time_s": time.time() - te}
        history.append(h)
        el = time.time() - t0
        pct = 100 * ep / epochs
        bar = "#" * int(pct // 4) + "-" * (25 - int(pct // 4))
        print(f"[{bar}] {pct:5.1f}% | epoch {ep}/{epochs} | elapsed {fmt_secs(el)} | ETA {fmt_secs(el / ep * (epochs - ep))}")
        print(f"   train loss {h['train_loss']:.4f} acc {h['train_acc']:.3f} | val loss {h['val_loss']:.4f} acc {h['val_acc']:.3f} | "
              f"val: fire/smoke kept {h['val_event_recall']:.3f}, false positives rejected {h['val_notfire_rejected']:.3f} | "
              f"per-class recall {{{', '.join(f'{k}: {v:.3f}' for k, v in h['per_class_recall'].items() if v is not None)}}}")
        if h["val_loss"] < best:
            best = h["val_loss"]
            Path(out_weights).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "classes": classes,
                        "imgsz": imgsz, "pad": pad, "arch": "efficientnet_b0", "epoch": ep, "val": h}, out_weights)
            print(f"   saved best verifier -> {out_weights}")
    hist_path = Path(out_weights).with_suffix(".history.json")
    hist_path.write_text(json.dumps(history, indent=2))
    plot_verifier_curves(history, Path(out_weights).with_suffix(".curves.png"), show=show_plots)
    print(f"Verifier training done in {fmt_secs(time.time() - t0)}")
    return history
