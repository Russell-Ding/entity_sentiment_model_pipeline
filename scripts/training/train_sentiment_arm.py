#!/usr/bin/env python3
"""Second-phase sentiment-head retrain — one ablation arm (standalone .py).

Extracted from the launcher notebook so the training logic lives in a tested,
reviewable script and Colab only *calls* it (cheaper; survives disconnects).

Loads the v2.0 checkpoint, applies per-arm encoder freezing + loss recipe, trains
(sentiment objective only), selects the best epoch by **CCC** (not MSE — MSE rewards
timidity / under-polarization), and saves best_model.pt + history.json.

Arms (see docs/second_phase_retrain_plan.md):
  armA    frozen encoder, legacy MSE+(1-Pearson)        -> data effect (control)
  control frozen encoder, CCC + weighted-Huber          -> recipe effect, head-only
  armC    top-2 encoder layers unfrozen, CCC+Huber       -> discrimination lever
  armD    frozen, weighted-Huber only (no CCC/sign)      -> isolate magnitude weighting
  armE    frozen, CCC+Huber, no sign penalty             -> isolate sign penalty

Usage (Colab):
  python scripts/training/train_sentiment_arm.py --arm control \
      --project /content/drive/.../entity_sentiment_model_pipeline
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from datetime import datetime
from pathlib import Path

ARMS = {
    "armA":    dict(unfreeze_top=0, loss="mse_pearson", loss_kw={}),
    "control": dict(unfreeze_top=0, loss="ccc_huber",  loss_kw=dict(ccc_weight=1.0, sign_weight=0.5, mag_alpha=3.0)),
    "armC":    dict(unfreeze_top=2, loss="ccc_huber",  loss_kw=dict(ccc_weight=1.0, sign_weight=0.5, mag_alpha=3.0)),
    "armD":    dict(unfreeze_top=0, loss="ccc_huber",  loss_kw=dict(ccc_weight=0.0, sign_weight=0.0, mag_alpha=3.0)),
    "armE":    dict(unfreeze_top=0, loss="ccc_huber",  loss_kw=dict(ccc_weight=1.0, sign_weight=0.0, mag_alpha=3.0)),
}
N_ENCODER_LAYERS = 24  # longformer-large-4096


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--project", default=str(Path(__file__).resolve().parents[2]),
                    help="project root (so models/ & training/ import)")
    ap.add_argument("--checkpoint", default=None, help="start checkpoint (default v2.0 model.pt)")
    ap.add_argument("--train-file", default=None)
    ap.add_argument("--val-file", default=None)
    ap.add_argument("--ckpt-dir", default=None, help="default checkpoints/retrain_<run_id>")
    ap.add_argument("--tag", default=None,
                    help="run label so re-runs don't overwrite each other (default: timestamp)")
    ap.add_argument("--local-ckpt", default="/content/best.pt", help="fast local save before Drive copy")
    ap.add_argument("--resume", choices=["auto", "off"], default="auto",
                    help="auto: continue from <ckpt_dir>/last.pt if present (default); off: fresh")
    ap.add_argument("--encoder-name", default="allenai/longformer-large-4096")
    ap.add_argument("--hidden-size", type=int, default=1024)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--encoder-lr", type=float, default=2e-6)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--gradient-clip", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main():
    args = build_args()
    proj = Path(args.project).resolve()
    sys.path.insert(0, str(proj))
    os.chdir(proj)

    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from models.pipeline import FinancialEntitySentimentModel
    from training.preprocessing import LABEL_TO_ID, DataPreprocessor
    from training.dataset import create_data_loaders
    from training.trainer import compute_sentiment_metrics

    arm = ARMS[args.arm]
    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.arm}__{tag}"
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                          cwd=str(proj), text=True).strip()
    except Exception:
        git_sha = "unknown"
    ckpt = args.checkpoint or str(proj / "trained_model/v2.0_20260517/model.pt")
    train_file = args.train_file or str(proj / "data/labeled/deepseek_t1/splits/train.jsonl")
    val_file = args.val_file or str(proj / "data/labeled/deepseek_t1/splits/val.jsonl")
    ckpt_dir = Path(args.ckpt_dir or proj / f"checkpoints/retrain_{run_id}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for p in (ckpt, train_file, val_file):
        assert os.path.exists(p), f"missing: {p}"
    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")
    use_amp = device.type == "cuda"
    print(f"run_id={run_id} (git {git_sha}) | {arm} | device={device} | ckpt_dir={ckpt_dir}")

    # --- model: load v2.0, freeze encoder + NER, optionally unfreeze top-k layers ---
    model = FinancialEntitySentimentModel(
        encoder_name=args.encoder_name, hidden_size=args.hidden_size,
        num_ner_labels=len(LABEL_TO_ID), use_ner_head=True, use_crf_ner=True,
        ner_label_to_id=LABEL_TO_ID, max_length=args.max_length, device="cpu")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.freeze_encoder()
    for n, p in model.named_parameters():
        if n.startswith("ner_head."):
            p.requires_grad = False
    if arm["unfreeze_top"] > 0:
        layers = list(range(N_ENCODER_LAYERS - arm["unfreeze_top"], N_ENCODER_LAYERS))
        nm = 0
        for n, p in model.named_parameters():
            if n.startswith("encoder.") and any(f".layer.{i}." in n for i in layers):
                p.requires_grad = True
                nm += p.numel()
        print(f"unfroze top {arm['unfreeze_top']} encoder layers {layers}: {nm:,} params")
    model.sentiment_head.configure_loss(arm["loss"], **arm["loss_kw"])

    # --- discriminative-LR optimizer ---
    enc = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    groups = [{"params": head, "lr": args.head_lr}]
    if enc:
        groups.append({"params": enc, "lr": args.encoder_lr})
    optimizer = AdamW(groups, weight_decay=args.weight_decay)
    print(f"trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- data ---
    pre = DataPreprocessor(model_name=args.encoder_name, max_length=args.max_length,
                           use_expanded_sentiment=True)
    train_loader, val_loader = create_data_loaders(
        train_files=train_file, val_files=val_file, preprocessor=pre, batch_size=args.batch_size)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * max(len(train_loader), 1))
    scaler = torch.amp.GradScaler(enabled=use_amp)

    def run_epoch(train: bool):
        model.train() if train else model.eval()
        loader = train_loader if train else val_loader
        preds, tgts, tot, nb = [], [], 0.0, 0
        for batch in loader:
            ii = batch["input_ids"].to(device); am = batch["attention_mask"].to(device)
            em = batch["entity_masks"].to(device); st = batch["sentiment_scores"].to(device)
            ev = batch["entity_mask_valid"].to(device)
            with torch.set_grad_enabled(train), torch.amp.autocast(device_type=device.type, enabled=use_amp):
                _, sp = model(input_ids=ii, attention_mask=am, entity_masks=em)
            vm = ev.bool(); vp = sp[vm].float(); vt = st[vm].float()
            if vp.numel() == 0:
                continue
            loss = model.sentiment_head.compute_loss(vp, vt)
            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                scaler.step(optimizer); scaler.update(); scheduler.step()
            tot += loss.item(); nb += 1
            preds += vp.detach().cpu().tolist(); tgts += vt.detach().cpu().tolist()
        m = compute_sentiment_metrics(preds, tgts)
        m["loss"] = tot / max(nb, 1)  # average over batches that actually had valid entities
        return m

    # ---- resume support: full-state last.pt written EVERY epoch (model + optimizer +
    # scheduler + scaler + epoch + best_ccc + hist), so an interrupted run continues
    # exactly where it stopped instead of restarting from scratch on costly Colab. ----
    last_path = ckpt_dir / "last.pt"
    start_epoch, best_ccc, patience, hist, base_ccc = 0, float("-inf"), 0, [], None
    if args.resume == "auto" and last_path.exists():
        r = torch.load(last_path, map_location=device, weights_only=False)
        # Refuse to continue a run whose arm OR exact loss config differs from this
        # invocation — otherwise the two halves train under different objectives.
        # Also reject pre-feature checkpoints missing the fields we need to do this
        # safely (arm, config, base_ccc); start those fresh with --resume off.
        if r.get("arm") != args.arm or r.get("config") != arm:
            raise SystemExit(
                f"resume config mismatch: last.pt arm={r.get('arm')} config={r.get('config')} "
                f"but this run is arm={args.arm} config={arm}. Use --resume off for a fresh run, "
                f"or pass the matching --arm.")
        if "base_ccc" not in r or r["base_ccc"] is None:
            raise SystemExit("resume checkpoint lacks base_ccc (untrained baseline); it predates "
                             "resume support. Start fresh with --resume off.")
        # Adopt the original run's identity so ledger/metadata stay consistent.
        run_id = r.get("run_id", run_id); tag = r.get("tag", tag)
        model.load_state_dict(r["model_state_dict"])
        optimizer.load_state_dict(r["optimizer_state_dict"])
        scheduler.load_state_dict(r["scheduler_state_dict"])
        if r.get("scaler_state_dict") and use_amp:
            scaler.load_state_dict(r["scaler_state_dict"])
        start_epoch = r["epoch"] + 1
        best_ccc = r["best_ccc"]; patience = r["patience"]; hist = r["history"]
        base_ccc = r.get("base_ccc")
        # If the saved run had already early-stopped (or finished all epochs), don't
        # train another epoch — go straight to finalize with the existing best.
        if patience >= args.patience or start_epoch >= args.epochs:
            print(f"RESUMED but run already complete (epoch {start_epoch}/{args.epochs}, "
                  f"patience {patience}/{args.patience}); finalizing without more training.")
            start_epoch = args.epochs
        else:
            print(f"RESUMED from {last_path} @ epoch {start_epoch} (best CCC {best_ccc:.4f}, "
                  f"{len(hist)} epochs done, patience {patience})", flush=True)

    if base_ccc is None:
        base = run_epoch(train=False)
        print("BASELINE val:", {k: round(v, 4) for k, v in base.items() if k != "loss"})
        base_ccc = base["sentiment_ccc"]
    for epoch in range(start_epoch, args.epochs):
        tr = run_epoch(train=True)
        va = run_epoch(train=False)
        hist.append({"epoch": epoch + 1, "train": tr, "val": va})
        print(f"epoch {epoch+1}: train_loss={tr['loss']:.4f} | val corr={va['sentiment_corr']:.4f} "
              f"ccc={va['sentiment_ccc']:.4f} std_ratio={va['sentiment_std_ratio']:.3f} "
              f"sign_flip={va['sentiment_sign_flip']:.3f} vneg_f1={va['sentiment_very_neg_f1']:.2f} "
              f"vpos_f1={va['sentiment_very_pos_f1']:.2f}", flush=True)
        if va["sentiment_ccc"] > best_ccc:
            best_ccc = va["sentiment_ccc"]; patience = 0
            payload = {"model_state_dict": model.state_dict(), "val_metrics": va,
                       "epoch": epoch, "arm": args.arm, "run_id": run_id, "tag": tag,
                       "git_sha": git_sha, "config": arm, "history": hist}
            torch.save(payload, args.local_ckpt)
            try:
                shutil.copy2(args.local_ckpt, ckpt_dir / "best_model.pt")
            except Exception as e:
                print("drive copy warn:", e)
            print(f"  -> new best CCC={best_ccc:.4f}")
        else:
            patience += 1
        # Resume point — write EVERY epoch (local fast, then mirror to Drive) so a
        # disconnect after this epoch resumes here, not from scratch.
        resume_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "epoch": epoch, "best_ccc": best_ccc, "patience": patience,
            "base_ccc": base_ccc, "history": hist, "arm": args.arm,
            "run_id": run_id, "tag": tag, "git_sha": git_sha, "config": arm,
        }
        local_last = Path(args.local_ckpt).with_name("last.pt")
        torch.save(resume_payload, local_last)
        try:
            shutil.copy2(local_last, last_path)
        except Exception as e:
            print("drive copy warn (last.pt):", e)
        if patience >= args.patience:
            print("early stop"); break
    (ckpt_dir / "history.json").write_text(json.dumps(hist, indent=2))
    (ckpt_dir / f"history_{run_id}.json").write_text(json.dumps(hist, indent=2))
    beat = best_ccc > base_ccc
    best_epoch = (max(range(len(hist)), key=lambda i: hist[i]["val"]["sentiment_ccc"]) + 1
                  if hist else 0)
    # Append-only ledger across ALL runs/arms — never overwritten, so you can always
    # tell runs apart (run_id, time, git, config, baseline vs best CCC).
    ledger = proj / "outputs" / "retrain_runs.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "run_id": run_id, "arm": args.arm, "tag": tag,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "git_sha": git_sha,
        "loss": arm["loss"], "loss_kw": arm["loss_kw"], "unfreeze_top": arm["unfreeze_top"],
        "batch_size": args.batch_size, "head_lr": args.head_lr, "encoder_lr": args.encoder_lr,
        "base_val_ccc": round(base_ccc, 4), "best_val_ccc": round(best_ccc, 4),
        "best_epoch": best_epoch, "beat_baseline": bool(beat),
        "ckpt": str(ckpt_dir / "best_model.pt"),
    }
    with ledger.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    # Run finished cleanly — drop last.pt so a later re-run of this id starts fresh.
    try:
        last_path.unlink(missing_ok=True)
    except Exception:
        pass
    print(f"DONE run_id={run_id} best_val_CCC={best_ccc:.4f} "
          f"({'BEAT' if beat else 'did NOT beat'} untrained baseline {base_ccc:.4f}) "
          f"-> {ckpt_dir/'best_model.pt'}  | ledger: outputs/retrain_runs.jsonl")


if __name__ == "__main__":
    main()
