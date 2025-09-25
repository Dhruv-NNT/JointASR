"""ASR training entry point.

Wires together:
- config.Config (and sub-configs)
- dataset.ASRDataset + collate_fn
- features.HFRobustExtractor (for robust='hf' batches)
- model.JointASR
- training.{WarmupCosineScheduler, LossComputer, Trainer, compute_ctc_weight}
- evaluation.Evaluator (for WER + optional val loop)
- checkpoint save/resume
"""
import os, gc, sys
import torch.nn as nn
gc.collect()
if "--gpus" in sys.argv:
    i = sys.argv.index("--gpus")
    if i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i+1]
import time
import math
import argparse
import random
import numpy as np
from typing import Tuple
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import (
    Config, DataConfig, ModelConfig, TrainingConfig,
    MelConfig, HFRobustConfig
)
from tokenizer import CharTokenizer, build_tokenizer
from dataset import ASRDataset, collate_fn
from features import HFRobustExtractor
from model import JointASR
from training import (
    WarmupCosineScheduler, LossComputer, Trainer, compute_ctc_weight
)
from evaluation import (
    Evaluator,
    batch_wer_and_one_sample
)

# -----------------------------
# Utilities
# -----------------------------

def banner(msg: str):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80 + "\n")

def verify_feature_mapping(cfg):
    import pandas as pd, os
    try:
        df = pd.read_csv(cfg.data.csv_path)
        if len(df) == 0:
            print("[verify] Train CSV is empty.")
            return
        raw = df.iloc[0]["audio_filename"]

        root = cfg.feature_root
        if not root.startswith("features/"):
            root = f"features/{root}"
        feat = (
            raw
            .replace("audios", f"{root}/{cfg.robust_layer}")
            .replace("wav/arctic_", "")
            .replace(".wav", ".npy")
        )

        exists = os.path.isfile(feat)
        print("[verify] Raw audio:", raw)
        print("[verify] Mapped feature:", feat)
        print("[verify] Exists on disk:", exists)
        if not exists:
            print("[verify][WARNING] Feature file not found – check feature_root/robust_layer or mapping rules.")
    except Exception as e:
        print(f"[verify][ERROR] Verification failed: {e}")


def ckpt_path_epoch(save_dir: str, epoch: int) -> str:
    return os.path.join(save_dir, f"epoch{epoch:02d}.pt")

def find_latest_checkpoint(save_dir: str) -> str:
    last = os.path.join(save_dir, "last.pt")
    if os.path.isfile(last):
        return last
    eps = []
    if not os.path.isdir(save_dir):
        return ""
    for fn in os.listdir(save_dir):
        if fn.startswith("epoch") and fn.endswith(".pt"):
            try:
                ep = int(fn.replace("epoch", "").replace(".pt", ""))
                eps.append((ep, fn))
            except ValueError:
                pass
    if not eps:
        return ""
    eps.sort()
    return os.path.join(save_dir, eps[-1][1])

def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    epoch: int,
    best_val: float,
    global_step: int,
    run_dir: str,
    cfg: Config
):
    rng = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()

    payload = {
    # save the real module weights if DataParallel is used
    "model": model.state_dict() if not isinstance(model, torch.nn.DataParallel) else model.module.state_dict(),
    "optimizer": optimizer.state_dict(),

    # keep your existing scheduler state
    "scheduler": scheduler.state_dict(),

    # add reference-style scheduler meta (helps resume LR exactly)
    "scheduler_step": getattr(scheduler, "step_num", 0),
    "scheduler_cfg": {
        "warmup": scheduler.warmup_steps,
        "max_lr": scheduler.max_lr,
        "total":  scheduler.total_steps,
    },

    "epoch": int(epoch),
    "best_val": float(best_val),
    "global_step": int(global_step),

    # RNG snapshot for deterministic resume
    "rng": {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "random": random.getstate(),
        **({"cuda": torch.cuda.get_rng_state_all()} if torch.cuda.is_available() else {}),
    },

    # keep what you already stored
    "cfg": {
        "feature_mode": cfg.feature_mode,
        "robust_source": cfg.robust_source,
        "robust_layer": cfg.robust_layer,
        "feature_root": cfg.feature_root,
    },
    "tb_run_dir": run_dir,
    }

    torch.save(payload, path)

def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    device: str
) -> Tuple[int, float, int, str]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if "scheduler_step" in ckpt:
        scheduler.step_num = int(ckpt["scheduler_step"])
    if "scheduler_cfg" in ckpt:
        sc = ckpt["scheduler_cfg"]
        scheduler.warmup_steps = sc.get("warmup", scheduler.warmup_steps)
        scheduler.max_lr       = sc.get("max_lr",   scheduler.max_lr)
        scheduler.total_steps  = sc.get("total",    scheduler.total_steps)

    # Restore RNG (best effort)
    try:
        torch.set_rng_state(ckpt["rng"]["torch"])
        np.random.set_state(ckpt["rng"]["numpy"])
        random.setstate(ckpt["rng"]["random"])
        if torch.cuda.is_available() and "cuda" in ckpt["rng"]:
            torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
    except Exception:
        pass

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val = float(ckpt.get("best_val", float("inf")))
    global_step = int(ckpt.get("global_step", 0))
    run_dir = ckpt.get("tb_run_dir", "")
    return start_epoch, best_val, global_step, run_dir


# -----------------------------
# HF helper (convert raw-audio batch -> feature batch)
# -----------------------------
def maybe_hf_to_features(batch, hf_extractor: HFRobustExtractor, device: str):
    """
    If the batch contains raw audio tuples (HF mode), convert to standard
    (X, x_len, Y_attn, y_attn_len, Y_ctc, y_ctc_len); otherwise return as-is.
    """
    if hf_extractor is None:
        return batch

    # HF raw-audio batch is shaped like: (wavs, srs, Y_attn, y_attn_len, Y_ctc, y_ctc_len)
    if isinstance(batch[0], list):  # wavs list
        wavs, srs, Y_attn, y_attn_len, Y_ctc, y_ctc_len = batch
        feats, x_len = hf_extractor.extract_batch(wavs, srs)  # [B, T', 1024], [B]
        # Move to device later in Trainer
        return (feats, x_len, Y_attn, y_attn_len, Y_ctc, y_ctc_len)
    return batch


# -----------------------------
# Argument parsing
# -----------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Train Joint CTC-Attention ASR")
    # Data
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv",   type=str, required=True)
    p.add_argument("--vocab_json", type=str, required=True)
    p.add_argument("--save_dir",   type=str, required=True)

    # Feature mode
    p.add_argument("--feature_mode", choices=["mel", "robust"], default="mel")
    p.add_argument("--robust_source", choices=["disk", "hf"], default="disk")
    # Feature mode
    p.add_argument("--feature_root", choices=["robust-ft", "xls-r-300m", "xlsr300m-ft"], default="xls-r-300m", help="Name of the robust feature family folder")
    p.add_argument("--robust_layer", type=str, default="24")

    # Mel config overrides
    p.add_argument("--n_mels", type=int, default=80)

    # Training
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--peak_lr", type=float, default=1e-3)
    p.add_argument("--warmup_steps", type=int, default=13450)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume_path", type=str, default="")
    p.add_argument("--gpus", type=str, default=None, help="CUDA_VISIBLE_DEVICES string, e.g. '5' or '5,6'")
    return p


# -----------------------------
# Main
# -----------------------------
def main():
    args = build_argparser().parse_args()
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # Build Config
    data_cfg = DataConfig(
        csv_path=args.train_csv,
        val_csv_path=args.val_csv,
        vocab_json=args.vocab_json,
        save_dir=args.save_dir,
        resume=args.resume,
        resume_path=args.resume_path,
    )
    model_cfg = ModelConfig(input_dim=80)  # will be overwritten by update_for_mode
    train_cfg = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        peak_lr=args.peak_lr,
        warmup_steps=args.warmup_steps,
        num_workers=args.num_workers,
        device=args.device,
    )
    mel_cfg = MelConfig(n_mels=args.n_mels) if args.feature_mode == "mel" else None
    hf_cfg = HFRobustConfig() if (args.feature_mode == "robust" and args.robust_source == "hf") else None

    cfg = Config(
        data=data_cfg,
        model=model_cfg,
        training=train_cfg,
        mel=mel_cfg,
        hf_robust=hf_cfg,
        feature_mode=args.feature_mode,
        robust_source=args.robust_source,
        feature_root=args.feature_root,
        robust_layer=args.robust_layer,
    )
    cfg.update_for_mode()

    # Banner
    if cfg.feature_mode == "robust":
        if cfg.robust_source == "hf":
            bmsg = (
                "TRAINING START\n"
                f"Mode: ROBUST (HF on-the-fly)\n"
                f"Input dim: {cfg.model.input_dim}\n"
                f"Layer: {cfg.robust_layer}\n"
                f"Train CSV: {cfg.data.csv_path}\n"
                f"Val CSV:   {cfg.data.val_csv_path}\n"
                f"Save dir:  {cfg.data.save_dir}"
            )
        else:
            bmsg = (
                "TRAINING START\n"
                f"Mode: ROBUST (disk)\n"
                f"Input dim: {cfg.model.input_dim}\n"
                f"Layer: {cfg.robust_layer}\n"
                f"Root:  {cfg.feature_root}\n"
                f"Train CSV: {cfg.data.csv_path}\n"
                f"Val CSV:   {cfg.data.val_csv_path}\n"
                f"Save dir:  {cfg.data.save_dir}"
            )
    else:
        bmsg = (
            "TRAINING START\n"
            f"Mode: MEL\n"
            f"Input dim: {cfg.model.input_dim} (n_mels={cfg.mel.n_mels if cfg.mel else 'NA'})\n"
            f"Train CSV: {cfg.data.csv_path}\n"
            f"Val CSV:   {cfg.data.val_csv_path}\n"
            f"Save dir:  {cfg.data.save_dir}"
        )
    banner(bmsg)
    # Verify mapping sanity before training
    verify_feature_mapping(cfg)

    # Dirs + TensorBoard
    os.makedirs(cfg.data.save_dir, exist_ok=True)
    tb_root = os.path.join(cfg.data.save_dir, "tb")
    os.makedirs(tb_root, exist_ok=True)
    run_dir = ""
    writer = None
    global_step = 0

    # Tokenizer
    tokenizer = build_tokenizer(cfg.data.vocab_json)
    assert tokenizer.pad_id == tokenizer.blank_id, "CTC expects blank==pad"
    assert tokenizer.sos_id != tokenizer.eos_id

    # Datasets + Loaders
    train_set = ASRDataset(cfg.data.csv_path, tokenizer, cfg)
    val_set   = ASRDataset(cfg.data.val_csv_path, tokenizer, cfg)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=cfg.training.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=cfg.training.num_workers > 0,
    )

    # HF extractor (if needed)
    hf_extractor = None
    if cfg.feature_mode == "robust" and cfg.robust_source == "hf":
        assert cfg.model.input_dim == 1024
        hf_cfg = cfg.hf_robust or HFRobustConfig()
        # Respect layer argument (string/int or negative)
        try:
            layer_idx = int(cfg.robust_layer)
        except ValueError:
            layer_idx = hf_cfg.layer
        hf_extractor = HFRobustExtractor(
            HFRobustConfig(
                model_name=hf_cfg.model_name,
                layer=layer_idx,
                sample_rate=hf_cfg.sample_rate,
                device=cfg.training.device,
            )
        ).eval()

    # Model
    model = JointASR(cfg.model, tokenizer.vocab_size, tokenizer.pad_id, tokenizer.sos_id, tokenizer.eos_id)
    model = model.to(cfg.training.device)

    # --- Multi-GPU (DataParallel) like requested ---
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and cfg.training.device.startswith("cuda"):
        print(f"[GPU] Using DataParallel over {torch.cuda.device_count()} visible GPUs")
        model = nn.DataParallel(model)  # scatters to all visible GPUs

    # Optimizer + Scheduler
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.peak_lr, betas=(0.9, 0.98), eps=1e-9)
    steps_per_epoch = len(train_loader)
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=cfg.training.warmup_steps,
        max_lr=cfg.training.peak_lr,
        total_steps=max(1, cfg.training.epochs * max(1, steps_per_epoch)),
    )
    loss_comp = LossComputer(tokenizer)
    trainer = Trainer(model, optimizer, scheduler, loss_comp, cfg.training)

    # Resume
    best_val = float("inf")
    start_epoch = 1
    if cfg.data.resume:
        resume_from = cfg.data.resume_path or find_latest_checkpoint(cfg.data.save_dir)
        if resume_from and os.path.isfile(resume_from):
            print(f"[RESUME] Loading checkpoint: {resume_from}")
            start_epoch, best_val, global_step, tb_prev = load_checkpoint(
                resume_from, model, optimizer, scheduler, cfg.training.device
            )
            # print current LR like the reference
            try:
                curr_lr = optimizer.param_groups[0]["lr"]
                print(f"[RESUME] start_epoch={start_epoch} best_val={best_val:.6f} global_step={global_step}")
                print(f"[RESUME] current LR = {curr_lr:.6g}")
            except Exception:
                pass

            # Reuse previous TB run dir if it exists, else try the latest under tb_root, else new
            if tb_prev and os.path.isdir(tb_prev):
                run_dir = tb_prev
            else:
                subdirs = [os.path.join(tb_root, d) for d in os.listdir(tb_root)
                        if os.path.isdir(os.path.join(tb_root, d))]
                subdirs.sort()
                run_dir = subdirs[-1] if subdirs else ""
                if not run_dir:
                    run_dir = os.path.join(tb_root, time.strftime("%Y%m%d-%H%M%S"))
        else:
            print("[RESUME] No checkpoint found; training from scratch.")

    if not run_dir:
        run_dir = os.path.join(tb_root, time.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=run_dir, purge_step=global_step)
    trainer.writer = writer  # attach for logging

    # Evaluator for WER (optional)
    evaluator = Evaluator(model, tokenizer, device=cfg.training.device)

    print(f"TensorBoard logdir: {run_dir}")
    print(f"warmup {cfg.training.warmup_steps} steps to LR {cfg.training.peak_lr} over {cfg.training.epochs} epochs ({steps_per_epoch} steps/epoch)")

    # -----------------------------
    # Train Loop
    # -----------------------------
    # for epoch in range(start_epoch, cfg.training.epochs + 1):
    #     # Epoch-wise CTC weight
    #     ctc_w = compute_ctc_weight(epoch, cfg.training.epochs, cfg.training)
    #     if writer:
    #         writer.add_scalar("ctc_weight/epoch", ctc_w, epoch)

    #     # Train
    #     model.train()
    #     epoch_loss, epoch_ctc, epoch_attn = 0.0, 0.0, 0.0

    #     for batch in train_loader:
    #         batch = maybe_hf_to_features(batch, hf_extractor, cfg.training.device)
    #         stats = trainer.train_step(batch, ctc_w)
    #         epoch_loss += stats["total"]
    #         epoch_ctc  += stats["ctc"]
    #         epoch_attn += stats["attn"]

    #     n_steps = max(1, len(train_loader))
    #     tr_loss = epoch_loss / n_steps
    #     tr_ctc  = epoch_ctc  / n_steps
    #     tr_attn = epoch_attn / n_steps

    #     # Validate (compute loss + optional WER sample)
    #     val_loss, val_ctc, val_attn = 0.0, 0.0, 0.0
    #     model.eval()
    #     with torch.no_grad():
    #         for batch in val_loader:
    #             batch = maybe_hf_to_features(batch, hf_extractor, cfg.training.device)
    #             # Forward for loss
    #             X, x_len, Y_attn, y_attn_len, Y_ctc, y_ctc_len = batch
    #             X = X.to(cfg.training.device); x_len = x_len.to(cfg.training.device)
    #             Y_attn = Y_attn.to(cfg.training.device); Y_ctc = Y_ctc.to(cfg.training.device)
    #             ys_in = Y_attn[:, :-1]
    #             outputs = model(X, x_len, ys_in)
    #             targets = (Y_attn, y_attn_len, Y_ctc, y_ctc_len)
    #             loss, parts = loss_comp.compute(outputs, targets, ctc_w)
    #             val_loss += float(loss.item()); val_ctc += parts["ctc"]; val_attn += parts["attn"]

    #     n_val = max(1, len(val_loader))
    #     val_loss /= n_val; val_ctc /= n_val; val_attn /= n_val

    #     # Log epoch aggregates
    #     if writer:
    #         writer.add_scalar("train_epoch/loss", tr_loss, epoch)
    #         writer.add_scalar("train_epoch/ctc_loss", tr_ctc, epoch)
    #         writer.add_scalar("train_epoch/attn_loss", tr_attn, epoch)
    #         writer.add_scalar("val/loss", val_loss, epoch)
    #         writer.add_scalar("val/ctc_loss", val_ctc, epoch)
    #         writer.add_scalar("val/attn_loss", val_attn, epoch)

    #     print(f"[{epoch}] ctc_weight={ctc_w:.3f}")
    #     print(f"[{epoch}] train loss={tr_loss:.4f} ctc={tr_ctc:.4f} attn={tr_attn:.4f} | "
    #           f"val loss={val_loss:.4f} ctc={val_ctc:.4f} attn={val_attn:.4f}")

    #     # Save checkpoints
    #     if val_loss < best_val:
    #         best_val = val_loss
    #         save_checkpoint(os.path.join(cfg.data.save_dir, "best.pt"),
    #                         model, optimizer, scheduler, epoch, best_val, trainer.global_step, run_dir, cfg)
    #     # epoch & last
    #     save_checkpoint(ckpt_path_epoch(cfg.data.save_dir, epoch),
    #                     model, optimizer, scheduler, epoch, best_val, trainer.global_step, run_dir, cfg)
    #     save_checkpoint(os.path.join(cfg.data.save_dir, "last.pt"),
    #                     model, optimizer, scheduler, epoch, best_val, trainer.global_step, run_dir, cfg)

    #     # (Optional) Quick WER probe every 10 epochs
    #     # --- Reference-style: every 10 epochs, compute batch WER (CTC & Attn) on ONE random batch ---
    #     # --- before the epoch loop (already present) ---
    for epoch in range(start_epoch, cfg.training.epochs + 1):
        # epoch-wise CTC weight
        ctc_w = compute_ctc_weight(epoch, cfg.training.epochs, cfg.training)
        if writer:
            writer.add_scalar("ctc_weight/epoch", ctc_w, epoch)

        model.train()
        epoch_loss = epoch_ctc = epoch_attn = 0.0

        # WER sampling setup — deterministic per epoch
        do_print = (epoch % 10 == 0)
        steps_per_epoch = len(train_loader)
        rng_state = random.getstate()
        random.seed(epoch)  # deterministic batch pick for reproducibility
        print_batch_idx = random.randint(1, steps_per_epoch) if do_print else None
        random.setstate(rng_state)

        for batch_idx, batch in enumerate(train_loader, 1):
            # (hf conversion if needed)
            batch = maybe_hf_to_features(batch, hf_extractor, cfg.training.device)

            # ---- normal training step
            stats = trainer.train_step(batch, ctc_w)
            epoch_loss += stats["total"]; epoch_ctc += stats["ctc"]; epoch_attn += stats["attn"]

            # ---- one-time train-batch WER probe (side-effect free)
            if do_print and batch_idx == print_batch_idx:
                try:
                    # clone lightweight views for safety; keep labels on CPU
                    X, x_len, Y_attn, y_attn_len, Y_ctc, y_ctc_len = batch
                    X_wer    = X.detach().clone()
                    xlen_wer = x_len.detach().clone()

                    model_for_eval = model.module if isinstance(model, torch.nn.DataParallel) else model
                    wer_ctc, wer_attn, sample = batch_wer_and_one_sample(
                        model_for_eval,
                        (X_wer, xlen_wer, Y_attn, y_attn_len, Y_ctc, y_ctc_len),
                        tokenizer,
                        device=cfg.training.device,
                        max_len=128
                    )
                    print(f"[E{epoch:02d} B{batch_idx:04d}] Batch WER — CTC: {wer_ctc*100:.2f}% | Attn: {wer_attn*100:.2f}%")
                    print("   ├─ REF :", sample['ref'])
                    print("   ├─ CTC :", sample['ctc'])
                    print("   └─ ATTN:", sample['attn'])
                    if writer:
                        writer.add_scalar("train/batch_wer_ctc",  wer_ctc,  trainer.global_step)
                        writer.add_scalar("train/batch_wer_attn", wer_attn, trainer.global_step)
                        writer.add_text("train/sample",
                                        f"REF: {sample['ref']}\nCTC: {sample['ctc']}\nATTN:{sample['attn']}",
                                        trainer.global_step)
                except Exception as e:
                    print(f"[warn] train-batch WER probe failed: {e}")

        # ===== keep the end-of-epoch summary & validation just like before =====
        n_steps = max(1, len(train_loader))
        tr_loss = epoch_loss / n_steps
        tr_ctc  = epoch_ctc  / n_steps
        tr_attn = epoch_attn / n_steps

        # --- validation (unchanged)
        val_loss, val_ctc, val_attn = 0.0, 0.0, 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = maybe_hf_to_features(batch, hf_extractor, cfg.training.device)
                X, x_len, Y_attn, y_attn_len, Y_ctc, y_ctc_len = batch
                X = X.to(cfg.training.device); x_len = x_len.to(cfg.training.device)
                Y_attn = Y_attn.to(cfg.training.device); Y_ctc = Y_ctc.to(cfg.training.device)
                ys_in = Y_attn[:, :-1]
                outputs = model(X, x_len, ys_in)
                targets = (Y_attn, y_attn_len, Y_ctc, y_ctc_len)
                loss, parts = loss_comp.compute(outputs, targets, ctc_w)
                val_loss += float(loss.item()); val_ctc += parts["ctc"]; val_attn += parts["attn"]

        n_val = max(1, len(val_loader))
        val_loss /= n_val; val_ctc /= n_val; val_attn /= n_val

        if writer:
            writer.add_scalar("train_epoch/loss", tr_loss, epoch)
            writer.add_scalar("train_epoch/ctc_loss", tr_ctc, epoch)
            writer.add_scalar("train_epoch/attn_loss", tr_attn, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/ctc_loss", val_ctc, epoch)
            writer.add_scalar("val/attn_loss", val_attn, epoch)

        print(f"[{epoch}] ctc_weight={ctc_w:.3f}")
        print(f"[{epoch}] train loss={tr_loss:.4f} ctc={tr_ctc:.4f} attn={tr_attn:.4f} | "
            f"val loss={val_loss:.4f} ctc={val_ctc:.4f} attn={val_attn:.4f}")

        # Save checkpoints
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(os.path.join(cfg.data.save_dir, "best.pt"),
                            model, optimizer, scheduler, epoch, best_val, trainer.global_step, run_dir, cfg)
        # epoch & last
        save_checkpoint(ckpt_path_epoch(cfg.data.save_dir, epoch),
                        model, optimizer, scheduler, epoch, best_val, trainer.global_step, run_dir, cfg)
        save_checkpoint(os.path.join(cfg.data.save_dir, "last.pt"),
                        model, optimizer, scheduler, epoch, best_val, trainer.global_step, run_dir, cfg)

        # (Optional) Quick WER probe every 10 epochs
        # --- Reference-style: every 10 epochs, compute batch WER (CTC & Attn) on ONE random batch ---
        # --- before the epoch loop (already present) ---


if __name__ == "__main__":
    main()
