"""
Stage 5: Full training script for the spectral decomposition model.

Features:
  - YAML config + CLI overrides
  - On-the-fly synthetic data generation (infinite IterableDataset)
  - Checkpoint save/resume (survives Run:AI preemptions)
  - SIGTERM handler for graceful shutdown
  - Mixed precision (AMP) support
  - Cosine LR scheduler
  - CSV + TensorBoard logging
  - Validation with fixed batch (holdout chemicals)

Usage:
    cd /gpfs0/bgu-rgilad/users/orelgr/deep2
    python -m src.train --config configs/base.yaml --run_id smoke_test
    python -m src.train --config configs/base.yaml --run_id smoke_test --max_epochs 2
"""

from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.synth_mixtures import ChemicalPool, SynthConfig, SyntheticMixtures, make_fixed_batch
from src.model.decompose import DecomposeModel, collate_decompose
from src.model.loss import DecomposeLoss


# ── Graceful shutdown ────────────────────────────────────────────────
_STOP_REQUESTED = False


def _sigterm_handler(signum, frame):
    global _STOP_REQUESTED
    print(f"\n[SIGNAL] Received signal {signum} — will save checkpoint and exit after current step.")
    _STOP_REQUESTED = True


signal.signal(signal.SIGTERM, _sigterm_handler)
signal.signal(signal.SIGINT, _sigterm_handler)


# ── Config ───────────────────────────────────────────────────────────
def load_config(config_path: str, cli_overrides: dict) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides (flat keys like --max_epochs map to training.max_epochs)
    override_map = {
        "max_epochs": ("training", "max_epochs"),
        "batch_size": ("data", "batch_size"),
        "lr": ("optimizer", "lr"),
        "run_id": None,  # handled separately
        "config": None,
        "resume": None,
    }
    for key, val in cli_overrides.items():
        if val is None:
            continue
        if key in override_map:
            path = override_map[key]
            if path is not None:
                section, field = path
                cfg[section][field] = val
        # else: ignore unknown keys

    return cfg


# ── Checkpoint ───────────────────────────────────────────────────────
def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    step: int,
    best_metric: float,
    config: dict,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
        "config": config,
        "rng_torch": torch.random.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.rename(path)
    print(f"  [CKPT] Saved: {path}")


def load_checkpoint(path: Path, model, optimizer, scheduler, scaler, device):
    print(f"  [CKPT] Loading: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    if scaler and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    torch.random.set_rng_state(state["rng_torch"])
    if state.get("rng_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["rng_cuda"])
    return state["epoch"], state["step"], state["best_metric"]


def find_latest_checkpoint(ckpt_dir: Path, run_id: str) -> Path | None:
    run_dir = ckpt_dir / run_id
    if not run_dir.exists():
        return None
    last = run_dir / "last.pt"
    if last.exists():
        return last
    return None


def rotate_checkpoints(run_dir: Path, keep_n: int):
    """Remove old epoch checkpoints, keeping the most recent `keep_n`."""
    epoch_ckpts = sorted(run_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime)
    for old in epoch_ckpts[:-keep_n]:
        old.unlink()
        print(f"  [CKPT] Rotated: {old.name}")


# ── Validation ───────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, val_batch, criterion, device):
    model.eval()
    y = val_batch["y"].to(device)
    R = val_batch["R"].to(device)
    c_true = val_batch["c"].to(device)
    baseline_true = val_batch["baseline"].to(device)
    mask = val_batch["mask"].to(device)
    ref_mask = val_batch["ref_mask"].to(device)

    c_pred, b_pred = model(y, R, ref_mask)
    _, detail = criterion(c_pred, c_true, b_pred, baseline_true, y, R, mask, ref_mask)
    return detail


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train spectral decomposition model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--run_id", type=str, required=True, help="Run identifier")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Auto-resume from latest checkpoint (default: True)")
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    args = parser.parse_args()

    cfg = load_config(args.config, vars(args))
    run_id = args.run_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Run ID: {run_id}")
    print(f"Config: {args.config}")

    # ── Data ──────────────────────────────────────────────────────────
    print("Loading ChemicalPool...")
    pool = ChemicalPool.load()
    dcfg = cfg["data"]
    train_pool, holdout_pool = pool.split(
        holdout_frac=dcfg["holdout_frac"], seed=dcfg["holdout_seed"]
    )
    print(f"  Train chemicals: {len(train_pool.chemicals)}, Holdout: {len(holdout_pool.chemicals)}")

    # Training: infinite on-the-fly generator
    train_ds = SyntheticMixtures(train_pool, SynthConfig())
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=dcfg["batch_size"],
        collate_fn=collate_decompose,
        num_workers=dcfg["num_workers"],
        pin_memory=(device == "cuda"),
    )

    # Validation: fixed batch from holdout chemicals
    n_val = dcfg["batch_size"] * cfg["training"]["val_steps"]
    n_val = min(n_val, 512)  # cap at 512 samples for speed
    print(f"  Building fixed validation set: {n_val} samples from holdout chemicals...")
    val_samples = make_fixed_batch(holdout_pool, n=n_val, seed=99)
    val_batch = collate_decompose(val_samples)

    # ── Model ─────────────────────────────────────────────────────────
    mcfg = cfg["model"]
    model = DecomposeModel(
        d_model=mcfg["d_model"],
        n_transformer_layers=mcfg["n_transformer_layers"],
        n_heads=mcfg["n_heads"],
        dropout=mcfg["dropout"],
        poly_order=mcfg["poly_order"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # ── Loss / Optimizer / Scheduler ──────────────────────────────────
    lcfg = cfg["loss"]
    criterion = DecomposeLoss(
        lambda_c=lcfg["lambda_c"],
        lambda_r=lcfg["lambda_r"],
        lambda_b=lcfg["lambda_b"],
        lambda_l1=lcfg["lambda_l1"],
        lambda_neg=lcfg["lambda_neg"],
    )

    ocfg = cfg["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=ocfg["lr"], weight_decay=ocfg["weight_decay"]
    )

    tcfg = cfg["training"]
    max_epochs = tcfg["max_epochs"]
    steps_per_epoch = tcfg["steps_per_epoch"]

    # Scheduler
    scfg = cfg["scheduler"]
    scheduler = None
    if scfg["type"] == "cosine":
        T_max = (scfg["T_max_epochs"] or max_epochs) * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=scfg["eta_min"]
        )

    # AMP scaler
    use_amp = tcfg["amp"] and device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"  AMP: {'enabled' if use_amp else 'disabled'}")

    # ── Checkpoint resume ─────────────────────────────────────────────
    ccfg = cfg["checkpoint"]
    ckpt_dir = ROOT / ccfg["dir"]
    run_dir = ckpt_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    global_step = 0
    best_metric = float("inf")

    if args.resume:
        latest = find_latest_checkpoint(ckpt_dir, run_id)
        if latest:
            start_epoch, global_step, best_metric = load_checkpoint(
                latest, model, optimizer, scheduler, scaler, device
            )
            start_epoch += 1  # resume from next epoch
            print(f"  Resuming from epoch {start_epoch}, step {global_step}, best={best_metric:.6f}")
        else:
            print("  No checkpoint found — starting fresh.")

    # ── Logging ───────────────────────────────────────────────────────
    logcfg = cfg["logging"]
    out_dir = ROOT / "outputs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "metrics.csv"
    csv_file = None
    csv_writer = None
    if logcfg["csv"]:
        csv_exists = csv_path.exists()
        csv_file = open(csv_path, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if not csv_exists:
            csv_writer.writerow([
                "epoch", "step", "lr",
                "train_loss", "train_loss_c", "train_loss_r", "train_loss_b",
                "val_loss", "val_loss_c", "val_loss_r", "val_loss_b",
            ])

    tb_writer = None
    if logcfg["tensorboard"]:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=str(out_dir / "tb"))
        except ImportError:
            print("  [WARN] tensorboard not installed — skipping TB logging")

    # Save config to output dir
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # ── Training loop ─────────────────────────────────────────────────
    grad_clip = tcfg["grad_clip"]
    print_every = logcfg["print_every_n_steps"]

    print(f"\n{'='*60}")
    print(f"Training: {max_epochs} epochs × {steps_per_epoch} steps/epoch")
    print(f"Batch size: {dcfg['batch_size']}, LR: {ocfg['lr']}")
    print(f"{'='*60}\n")

    train_iter = iter(train_loader)

    for epoch in range(start_epoch, max_epochs):
        if _STOP_REQUESTED:
            break

        model.train()
        epoch_losses = {k: 0.0 for k in ["loss", "loss_c", "loss_r", "loss_b"]}
        t0 = time.time()

        for step_in_epoch in range(1, steps_per_epoch + 1):
            if _STOP_REQUESTED:
                break

            global_step += 1

            batch = next(train_iter)
            y = batch["y"].to(device, non_blocking=True)
            R = batch["R"].to(device, non_blocking=True)
            c_true = batch["c"].to(device, non_blocking=True)
            b_true = batch["baseline"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            ref_mask = batch["ref_mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    c_pred, b_pred = model(y, R, ref_mask)
                    loss, detail = criterion(c_pred, c_true, b_pred, b_true, y, R, mask, ref_mask)
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                c_pred, b_pred = model(y, R, ref_mask)
                loss, detail = criterion(c_pred, c_true, b_pred, b_true, y, R, mask, ref_mask)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            if scheduler:
                scheduler.step()

            for k in epoch_losses:
                epoch_losses[k] += detail[k]

            if step_in_epoch % print_every == 0 or step_in_epoch == 1:
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"  [{epoch+1:3d}/{max_epochs}] step {step_in_epoch:4d}/{steps_per_epoch}  "
                    f"loss={detail['loss']:.5f}  c={detail['loss_c']:.5f}  "
                    f"r={detail['loss_r']:.6f}  b={detail['loss_b']:.6f}  "
                    f"lr={lr_now:.2e}"
                )

            if tb_writer:
                tb_writer.add_scalar("train/loss", detail["loss"], global_step)
                tb_writer.add_scalar("train/loss_c", detail["loss_c"], global_step)
                tb_writer.add_scalar("train/loss_r", detail["loss_r"], global_step)
                tb_writer.add_scalar("train/loss_b", detail["loss_b"], global_step)

        # End of epoch
        dt = time.time() - t0
        avg = {k: v / steps_per_epoch for k, v in epoch_losses.items()}
        print(f"\n  Epoch {epoch+1} done in {dt:.1f}s — "
              f"avg loss={avg['loss']:.5f}  c={avg['loss_c']:.5f}  "
              f"r={avg['loss_r']:.6f}  b={avg['loss_b']:.6f}")

        # ── Validation ────────────────────────────────────────────────
        val_detail = validate(model, val_batch, criterion, device)
        print(f"  Val: loss={val_detail['loss']:.5f}  c={val_detail['loss_c']:.5f}  "
              f"r={val_detail['loss_r']:.6f}  b={val_detail['loss_b']:.6f}")

        if tb_writer:
            for k, v in val_detail.items():
                tb_writer.add_scalar(f"val/{k}", v, global_step)

        if csv_writer:
            lr_now = optimizer.param_groups[0]["lr"]
            csv_writer.writerow([
                epoch + 1, global_step, f"{lr_now:.6e}",
                f"{avg['loss']:.6f}", f"{avg['loss_c']:.6f}",
                f"{avg['loss_r']:.6f}", f"{avg['loss_b']:.6f}",
                f"{val_detail['loss']:.6f}", f"{val_detail['loss_c']:.6f}",
                f"{val_detail['loss_r']:.6f}", f"{val_detail['loss_b']:.6f}",
            ])
            csv_file.flush()

        # ── Checkpointing ─────────────────────────────────────────────
        is_best = val_detail["loss"] < best_metric
        if is_best:
            best_metric = val_detail["loss"]
            save_checkpoint(
                run_dir / "best.pt", model, optimizer, scheduler, scaler,
                epoch, global_step, best_metric, cfg,
            )

        if (epoch + 1) % ccfg["save_every_n_epochs"] == 0 or _STOP_REQUESTED:
            save_checkpoint(
                run_dir / f"epoch_{epoch+1:04d}.pt", model, optimizer, scheduler, scaler,
                epoch, global_step, best_metric, cfg,
            )
            rotate_checkpoints(run_dir, ccfg["keep_last_n"])

        # Always save "last" checkpoint
        save_checkpoint(
            run_dir / "last.pt", model, optimizer, scheduler, scaler,
            epoch, global_step, best_metric, cfg,
        )

        if _STOP_REQUESTED:
            print("\n[STOP] Graceful shutdown complete.")
            break

        print()

    # ── Cleanup ───────────────────────────────────────────────────────
    if csv_file:
        csv_file.close()
    if tb_writer:
        tb_writer.close()

    print(f"\nTraining complete. Best val loss: {best_metric:.6f}")
    print(f"Checkpoints: {run_dir}")
    print(f"Metrics: {csv_path}")


if __name__ == "__main__":
    main()
