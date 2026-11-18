"""Detector training (Ultralytics YOLO26) with a per-epoch progress / loss printer and curve plots."""
import csv
import math
import time
from pathlib import Path

from .common import fmt_secs


def _in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None and "IPKernelApp" in get_ipython().config
    except Exception:  # noqa: BLE001
        return False


def read_results_csv(path):
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with open(p) as f:
        for r in csv.DictReader(f):
            rows.append({k.strip(): float(v) for k, v in r.items() if v not in (None, "")})
    return rows


def plot_detector_curves(results_csv, out_png=None, show=True, title="Detector training"):
    """Train vs validation loss per epoch + mAP / precision / recall."""
    import matplotlib.pyplot as plt
    rows = read_results_csv(results_csv)
    if not rows:
        print("no results yet")
        return None
    ep = [r.get("epoch", i + 1) for i, r in enumerate(rows)]
    loss_names = sorted({k.split("/", 1)[1] for r in rows for k in r if k.startswith("train/")})
    n = len(loss_names) + 1
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6))
    for ax, ln in zip(axes, loss_names):
        ax.plot(ep, [r.get(f"train/{ln}", math.nan) for r in rows], "o-", ms=3, label="train")
        ax.plot(ep, [r.get(f"val/{ln}", math.nan) for r in rows], "s-", ms=3, label="val")
        ax.set_title(ln)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    ax = axes[-1]
    for k, lab in (("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95"),
                   ("metrics/precision(B)", "precision"), ("metrics/recall(B)", "recall")):
        ax.plot(ep, [r.get(k, math.nan) for r in rows], "o-", ms=3, label=lab)
    ax.set_title("validation metrics")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=110)
    if show:
        plt.show()
    plt.close(fig)
    return out_png


class EpochPrinter:
    """Ultralytics callback: after every epoch prints progress %, ETA, train + val losses and metrics,
    and every `plot_every` epochs draws the loss curves inline."""

    def __init__(self, plot_every=5, show_plots=True):
        self.plot_every = plot_every
        self.show_plots = show_plots and _in_notebook()
        self.t0 = None

    def on_train_start(self, trainer):
        self.t0 = time.time()
        hrs = trainer.args.time
        print("=" * 100)
        print(f"TRAINING START | model={Path(str(trainer.args.model)).name} | imgs/epoch="
              f"{len(trainer.train_loader.dataset):,} | batch={trainer.batch_size} | "
              + (f"time budget={hrs} h" if hrs else f"epochs={trainer.epochs}"))
        print("=" * 100)

    def on_fit_epoch_end(self, trainer):
        ep, total = trainer.epoch + 1, max(trainer.epochs, trainer.epoch + 1)
        el = time.time() - (self.t0 or time.time())
        if trainer.args.time:
            budget = trainer.args.time * 3600
            pct = min(100.0, 100 * el / budget)
            eta = max(0.0, budget - el)
        else:
            pct = 100 * ep / total
            eta = el / ep * (total - ep)
        tr = trainer.label_loss_items(trainer.tloss, prefix="train") if trainer.tloss else {}
        m = trainer.metrics or {}
        va = {k: v for k, v in m.items() if k.startswith("val/")}
        if not va:   # final re-validation of best.pt after training (no losses) -> short summary only
            print(f"\nFINAL VALIDATION of best.pt: P={m.get('metrics/precision(B)', 0):.3f}  "
                  f"R={m.get('metrics/recall(B)', 0):.3f}  mAP50={m.get('metrics/mAP50(B)', 0):.3f}  "
                  f"mAP50-95={m.get('metrics/mAP50-95(B)', 0):.3f}")
            return
        bar = "#" * int(pct // 4) + "-" * (25 - int(pct // 4))
        print(f"\n[{bar}] {pct:5.1f}% | epoch {ep}/{total} | elapsed {fmt_secs(el)} | ETA {fmt_secs(eta)} "
              f"| epoch time {fmt_secs(getattr(trainer, 'epoch_time', 0) or 0)}")
        names = sorted({k.split('/', 1)[1] for k in list(tr) + list(va)})
        print("   loss       " + "  ".join(f"{n:>9s}" for n in names) + "      total")
        print("   train      " + "  ".join(f"{tr.get('train/' + n, float('nan')):9.4f}" for n in names)
              + f"  {sum(tr.values()):9.4f}")
        print("   val        " + "  ".join(f"{va.get('val/' + n, float('nan')):9.4f}" for n in names)
              + f"  {sum(va.values()) if va else float('nan'):9.4f}")
        print(f"   val metrics: P={m.get('metrics/precision(B)', 0):.3f}  R={m.get('metrics/recall(B)', 0):.3f}  "
              f"mAP50={m.get('metrics/mAP50(B)', 0):.3f}  mAP50-95={m.get('metrics/mAP50-95(B)', 0):.3f}  "
              f"| best fitness so far={float(trainer.best_fitness or 0):.4f}")
        if self.show_plots and self.plot_every and ep % self.plot_every == 0:
            try:
                plot_detector_curves(Path(trainer.save_dir) / "results.csv", show=True, title=f"after epoch {ep}")
            except Exception as e:  # noqa: BLE001
                print("plot failed:", e)

    def on_train_end(self, trainer):
        print(f"\nTRAINING DONE in {fmt_secs(time.time() - (self.t0 or time.time()))}. "
              f"Best weights: {trainer.best}")


def train_detector(data_yaml, project, name, model="yolo26m.pt", hours=2.0, epochs=200, imgsz=640,
                   batch=64, workers=8, device=0, resume=True, plot_every=5, **overrides):
    """Train (or resume) a detector. Returns (best_weights_path, save_dir)."""
    from ultralytics import YOLO
    project = Path(project).resolve()      # absolute, otherwise Ultralytics nests it under runs/detect/
    save_dir = project / name
    last = save_dir / "weights" / "last.pt"
    printer = EpochPrinter(plot_every=plot_every)

    def _attach(yolo):
        yolo.add_callback("on_train_start", printer.on_train_start)
        yolo.add_callback("on_fit_epoch_end", printer.on_fit_epoch_end)
        yolo.add_callback("on_train_end", printer.on_train_end)
        return yolo

    if resume and last.exists():
        import torch
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if ck.get("epoch", -1) == -1:
            print(f"{name}: already finished -> skipping training (delete {save_dir} to retrain)")
            return save_dir / "weights" / "best.pt", save_dir
        print(f"{name}: resuming from {last} (epoch {ck.get('epoch', 0) + 1})")
        _attach(YOLO(str(last))).train(resume=True)
    else:
        args = dict(data=str(data_yaml), project=str(project), name=name, exist_ok=True, imgsz=imgsz,
                    epochs=epochs, time=hours, batch=batch, workers=workers, device=device, patience=0,
                    cos_lr=True, close_mosaic=3, amp=True, cache=False, plots=True, seed=0, deterministic=False,
                    # augmentation tuned for fire: small hue shift so red lights don't turn "orange"
                    hsv_h=0.005, hsv_s=0.6, hsv_v=0.45, degrees=0.0, translate=0.1, scale=0.5,
                    fliplr=0.5, mosaic=1.0, mixup=0.05)
        args.update(overrides)
        _attach(YOLO(model)).train(**args)
    best = save_dir / "weights" / "best.pt"
    return (best if best.exists() else last), save_dir
