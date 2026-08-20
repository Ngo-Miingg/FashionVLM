
import argparse
import contextlib
import json
import math
import os
import random
import shutil
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather as all_gather_autograd
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoProcessor, AutoModel, get_cosine_schedule_with_warmup
from transformers.utils import logging as hf_logging


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["standard_eval", "dryrun", "train"])
    p.add_argument("--model_dir", required=True)
    p.add_argument("--manifest")
    p.add_argument("--train_manifest")
    p.add_argument("--val_manifest")
    p.add_argument("--baseline_val_metrics")
    p.add_argument("--output_dir", required=True)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=7)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--min_epochs_before_stop", type=int, default=4)

    # COMMON benchmark LR: same for every backbone in Experiment I.
    p.add_argument("--vision_lr", type=float, default=1e-5)
    p.add_argument("--text_lr", type=float, default=1e-6)
    p.add_argument("--other_lr", type=float, default=1e-5)

    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.08)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--eval_batch_size", type=int, default=32)

    p.add_argument("--same_cat_neg_weight", type=float, default=0.25)
    p.add_argument("--p_long", type=float, default=2/3)
    p.add_argument("--max_text_length", type=int, default=64)

    # COMMON mixed-precision runtime policy for all Experiment-I backbones.
    p.add_argument("--amp_init_scale", type=float, default=4096.0)
    p.add_argument("--amp_growth_factor", type=float, default=2.0)
    p.add_argument("--amp_backoff_factor", type=float, default=0.5)
    p.add_argument("--amp_growth_interval", type=int, default=2000)
    p.add_argument("--max_consecutive_amp_skips", type=int, default=4)

    # Dry-run must obtain real optimizer updates. AMP overflow is allowed
    # to self-calibrate by skipping the update and reducing the loss scale.
    p.add_argument("--dryrun_required_success_steps", type=int, default=2)
    p.add_argument("--dryrun_max_windows", type=int, default=6)

    # Common benchmark head, independent of native pretraining objective.
    p.add_argument("--init_temperature", type=float, default=0.07)
    p.add_argument("--init_logit_bias", type=float, default=0.0)
    return p.parse_args()


# ============================================================
# Runtime
# ============================================================

def setup_ddp():
    hf_logging.disable_progress_bar()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    kwargs = {"backend": "nccl", "timeout": timedelta(minutes=30)}
    try:
        dist.init_process_group(device_id=device, **kwargs)
    except TypeError:
        dist.init_process_group(**kwargs)
    return dist.get_rank(), dist.get_world_size(), local_rank, device


def seed_all(seed, rank=0):
    s = int(seed) + 1009 * int(rank)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def pooled(x):
    if torch.is_tensor(x):
        return x
    if hasattr(x, "pooler_output"):
        return x.pooler_output
    if isinstance(x, (tuple, list)):
        return x[0]
    raise TypeError(f"Unsupported feature output: {type(x)}")



def load_backbone_processor(model_dir, device=None, dtype=None, log_prefix="[load]"):
    hf_logging.disable_progress_bar()
    print(f"{log_prefix} processor...", flush=True)
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
        use_fast=False,
    )
    print(f"{log_prefix} processor OK {time.time()-t0:.1f}s", flush=True)

    # Native text contract probe. FixRes SigLIP2 checkpoints can use the
    # SigLIP-compatible processor/model stack and may omit attention_mask.
    probe = proc(
        text=["processor contract probe"],
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    if "input_ids" not in probe:
        raise RuntimeError(f"Processor text contract missing input_ids; keys={list(probe.keys())}")
    if tuple(probe["input_ids"].shape) != (1, 64):
        raise RuntimeError(
            f"Processor text length drift: expected (1,64), got {tuple(probe['input_ids'].shape)}"
        )
    print(
        f"{log_prefix} text contract OK | keys={list(probe.keys())} | "
        f"attention_mask={'present' if 'attention_mask' in probe else 'native-none'}",
        flush=True,
    )

    kw = {"local_files_only": True}
    if dtype is not None:
        kw["dtype"] = dtype

    print(f"{log_prefix} SigLIP2 AutoModel materialization...", flush=True)
    t0 = time.time()
    model = AutoModel.from_pretrained(model_dir, **kw)
    print(f"{log_prefix} SigLIP2 CPU OK {time.time()-t0:.1f}s", flush=True)

    # Experiment I uses a common retrieval scale/bias; native SigLIP head is not optimized.
    if hasattr(model, "logit_scale") and isinstance(model.logit_scale, torch.nn.Parameter):
        model.logit_scale.requires_grad_(False)
    if hasattr(model, "logit_bias") and isinstance(model.logit_bias, torch.nn.Parameter):
        model.logit_bias.requires_grad_(False)

    if device is not None:
        torch.cuda.set_device(device)
        print(f"{log_prefix} moving to {device}...", flush=True)
        t0 = time.time()
        model.to(device)
        torch.cuda.synchronize(device)
        print(
            f"{log_prefix} CUDA OK {time.time()-t0:.1f}s | "
            f"VRAM={torch.cuda.memory_allocated(device)/1024**3:.3f} GiB",
            flush=True,
        )
    return model, proc

def make_grad_scaler(args):
    return torch.amp.GradScaler(
        "cuda",
        init_scale=float(args.amp_init_scale),
        growth_factor=float(args.amp_growth_factor),
        backoff_factor=float(args.amp_backoff_factor),
        growth_interval=int(args.amp_growth_interval),
    )


def amp_optimizer_step(scaler, optimizer, scheduler=None):
    """
    GradScaler is the source of truth for FP16 overflow handling.
    scaler.step() skips optimizer.step() when inf/NaN gradients were
    recorded by unscale_(); scaler.update() then reduces the scale.
    """
    old_scale = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    new_scale = float(scaler.get_scale())
    stepped = new_scale >= old_scale
    if stepped and scheduler is not None:
        scheduler.step()
    return stepped, old_scale, new_scale


def gradient_finiteness(module):
    bad_names = []
    total_with_grad = 0
    for name, p in module.named_parameters():
        if p.grad is None:
            continue
        total_with_grad += 1
        if not torch.isfinite(p.grad).all():
            bad_names.append(name)
    return len(bad_names) == 0, total_with_grad, bad_names


def gather_grad(x):
    return torch.cat(tuple(all_gather_autograd(x)), dim=0)


def gather_nograd(x):
    out = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(out, x.contiguous())
    return torch.cat(out, dim=0)


# ============================================================
# Common benchmark wrapper / objective
# ============================================================


class RetrievalBenchmarkModel(nn.Module):
    """SigLIP2 adapter on top of the same Experiment-I retrieval objective."""
    def __init__(self, backbone, init_temperature=0.07, init_bias=0.0):
        super().__init__()
        self.backbone = backbone
        self.benchmark_logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / float(init_temperature)), dtype=torch.float32)
        )
        self.benchmark_logit_bias = nn.Parameter(
            torch.tensor(float(init_bias), dtype=torch.float32)
        )

    def forward(self, pixel_values, input_ids, attention_mask=None):
        # Transformers 5 may return a Tensor or BaseModelOutputWithPooling.
        image = pooled(self.backbone.get_image_features(pixel_values=pixel_values))
        text_kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            text_kwargs["attention_mask"] = attention_mask
        text = pooled(self.backbone.get_text_features(**text_kwargs))
        return (
            F.normalize(image.float(), dim=-1),
            F.normalize(text.float(), dim=-1),
        )

def weighted_pairwise_logistic(I, T, C, W, scale_param, bias_param, same_cat_neg_weight):
    scale = scale_param.float().exp().clamp(max=100.0)
    bias = bias_param.float()
    logits = (T @ I.t()) * scale + bias

    n = logits.size(0)
    eye = torch.eye(n, device=logits.device, dtype=torch.bool)
    signs = torch.where(eye, torch.ones_like(logits), -torch.ones_like(logits))
    pair_loss = -F.logsigmoid(signs * logits)

    # Same-category non-matching products are ambiguous negatives,
    # so they receive a lower weight under the COMMON benchmark protocol.
    same_cat = C[:, None].eq(C[None, :]) & (~eye)
    pair_weight = torch.where(
        same_cat,
        torch.full_like(pair_loss, float(same_cat_neg_weight)),
        torch.ones_like(pair_loss),
    )

    row_norm = float(n) / pair_weight.sum(1).clamp_min(1.0)
    col_norm = float(n) / pair_weight.sum(0).clamp_min(1.0)

    row_loss = (pair_loss * pair_weight).sum(1) * row_norm
    col_loss = (pair_loss * pair_weight).sum(0) * col_norm

    W = W.float()
    W = W / W.mean().clamp_min(1e-6)

    return 0.5 * (
        (row_loss * W).sum() / W.sum().clamp_min(1e-6)
        + (col_loss * W).sum() / W.sum().clamp_min(1e-6)
    )


def weighted_mean(x, w):
    w = w.float()
    return (x * w).sum() / w.sum().clamp_min(1e-6)


def build_optimizer(wrapper, args):
    groups = {}
    for name, p in wrapper.named_parameters():
        if not p.requires_grad:
            continue
        lo = name.lower()

        if "backbone.vision_model" in lo or "backbone.visual" in lo:
            lr, branch = args.vision_lr, "vision"
        elif "backbone.text_model" in lo or "backbone.text" in lo:
            lr, branch = args.text_lr, "text"
        else:
            lr, branch = args.other_lr, "other"

        no_decay = (
            name.endswith(".bias")
            or "layernorm" in lo
            or "layer_norm" in lo
            or lo.endswith(".norm.weight")
            or "benchmark_logit_" in lo
        )
        wd = 0.0 if no_decay else args.weight_decay
        groups.setdefault((branch, lr, wd), []).append(p)

    return torch.optim.AdamW([
        {"params": ps, "lr": lr, "weight_decay": wd}
        for (_, lr, wd), ps in groups.items()
    ])


# ============================================================
# Datasets / collators
# ============================================================

class TrainDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return {
            "manifest_id": int(r["manifest_id"]),
            "image_path": str(r["image_path"]),
            "short": str(r["short_text"]).lower(),
            "long": str(r["long_text"]).lower(),
            "category_id": int(r["category_id"]),
            "class_weight": float(r["class_weight"]),
        }


class TrainCollator:
    def __init__(self, processor, p_long=2/3, max_text_length=64):
        self.processor = processor
        self.p_long = float(p_long)
        self.max_text_length = int(max_text_length)

    def __call__(self, batch):
        images, texts = [], []
        for r in batch:
            with Image.open(r["image_path"]) as im:
                images.append(im.convert("RGB").copy())
            use_long = bool(torch.rand(1).item() < self.p_long)
            texts.append(r["long"] if use_long else r["short"])

        enc = self.processor(
            images=images,
            text=texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        return {
            "manifest_id": torch.tensor([r["manifest_id"] for r in batch], dtype=torch.long),
            "category_id": torch.tensor([r["category_id"] for r in batch], dtype=torch.long),
            "class_weight": torch.tensor([r["class_weight"] for r in batch], dtype=torch.float32),
            "pixel_values": enc["pixel_values"],
            "input_ids": enc["input_ids"],
            "attention_mask": enc.get("attention_mask"),
        }


class EvalDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return {
            "manifest_id": int(r["manifest_id"]),
            "image_path": str(r["image_path"]),
            "short": str(r["short_text"]).lower(),
            "long": str(r["long_text"]).lower(),
            "category": str(r["category"]),
            "freq_bin": str(r["freq_bin"]),
        }


class EvalCollator:
    def __init__(self, processor, max_text_length=64):
        self.processor = processor
        self.max_text_length = int(max_text_length)

    def __call__(self, batch):
        images = []
        for r in batch:
            with Image.open(r["image_path"]) as im:
                images.append(im.convert("RGB").copy())

        p_img = self.processor(images=images, return_tensors="pt")
        p_short = self.processor(
            text=[r["short"] for r in batch],
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        p_long = self.processor(
            text=[r["long"] for r in batch],
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        return {
            "manifest_id": torch.tensor([r["manifest_id"] for r in batch], dtype=torch.long),
            "pixel_values": p_img["pixel_values"],
            "short_ids": p_short["input_ids"],
            "short_mask": p_short.get("attention_mask"),
            "long_ids": p_long["input_ids"],
            "long_mask": p_long.get("attention_mask"),
            "category": [r["category"] for r in batch],
            "freq_bin": [r["freq_bin"] for r in batch],
        }


# ============================================================
# Embeddings / metrics — reliable fast path
# ============================================================

def encode_image(backbone, pixel_values):
    out = backbone.get_image_features(pixel_values=pixel_values)
    return F.normalize(pooled(out).float(), dim=-1)


def encode_text(backbone, input_ids, attention_mask=None):
    kw = {"input_ids": input_ids}
    if attention_mask is not None:
        kw["attention_mask"] = attention_mask
    out = backbone.get_text_features(**kw)
    return F.normalize(pooled(out).float(), dim=-1)


@torch.inference_mode()
def embed_manifest(backbone, processor, df, device, batch_size, num_workers, max_text_length):
    ds = EvalDataset(df)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=EvalCollator(processor, max_text_length),
    )

    mids, imgs, shorts, longs, cats, bins = [], [], [], [], [], []
    backbone.eval()
    dtype = next(backbone.parameters()).dtype
    t0 = time.time()

    for step, b in enumerate(dl, 1):
        px = b["pixel_values"].to(device, dtype=dtype, non_blocking=True)
        si = b["short_ids"].to(device, non_blocking=True)
        li = b["long_ids"].to(device, non_blocking=True)

        sm = b.get("short_mask")
        lm = b.get("long_mask")
        if sm is not None:
            sm = sm.to(device, non_blocking=True)
        if lm is not None:
            lm = lm.to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.float16):
            im = encode_image(backbone, px)
            st = encode_text(backbone, si, sm)
            lt = encode_text(backbone, li, lm)

        mids.append(b["manifest_id"].numpy())
        imgs.append(im.cpu().numpy().astype(np.float32))
        shorts.append(st.cpu().numpy().astype(np.float32))
        longs.append(lt.cpu().numpy().astype(np.float32))
        cats.extend(b["category"])
        bins.extend(b["freq_bin"])

        if step == 1 or step % 20 == 0 or step == len(dl):
            done = min(step * batch_size, len(ds))
            elapsed = max(time.time() - t0, 1e-6)
            print(
                f"[eval-encode] {step}/{len(dl)} batches | "
                f"{done}/{len(ds)} rows | {done/elapsed:.1f} rows/s",
                flush=True,
            )

    z = {
        "manifest_id": np.concatenate(mids),
        "image": np.concatenate(imgs),
        "short": np.concatenate(shorts),
        "long": np.concatenate(longs),
        "category": np.asarray(cats, dtype=object),
        "freq_bin": np.asarray(bins, dtype=object),
    }

    order = np.argsort(z["manifest_id"], kind="stable")
    for k in z:
        z[k] = z[k][order]
    return z


def metrics_from_ranks(ranks):
    ranks = np.asarray(ranks, dtype=np.int32)
    if len(ranks) == 0:
        return {
            "n": 0, "R@1": 0.0, "R@5": 0.0, "R@10": 0.0,
            "MRR@10": 0.0, "median_rank": None,
        }
    return {
        "n": int(len(ranks)),
        "R@1": float(np.mean(ranks <= 1)),
        "R@5": float(np.mean(ranks <= 5)),
        "R@10": float(np.mean(ranks <= 10)),
        "MRR@10": float(np.mean(np.where(ranks <= 10, 1.0 / ranks, 0.0))),
        "median_rank": float(np.median(ranks)),
    }


@torch.inference_mode()
def exact_ranks_gpu(query, gallery, category_codes=None, chunk=512, label="retrieval"):
    """
    Exact aligned-product rank without full argsort.

    rank = 1 + count(similarity > target_similarity)

    For within-category retrieval the same similarity chunk is reused and the
    count is restricted to the query's category. This is mathematically the
    same rank definition for non-tied similarities, while avoiding the slow
    CPU argsort path that looked like a hang in the previous notebook.
    """
    device = torch.device("cuda", 0)
    q = torch.from_numpy(np.asarray(query, dtype=np.float32)).to(device)
    g = torch.from_numpy(np.asarray(gallery, dtype=np.float32)).to(device)

    cats = None
    if category_codes is not None:
        cats = torch.from_numpy(np.asarray(category_codes, dtype=np.int64)).to(device)

    standard = []
    within = []
    t0 = time.time()

    for st in range(0, len(q), chunk):
        en = min(len(q), st + chunk)
        sim = q[st:en] @ g.t()

        target_idx = torch.arange(st, en, device=device)
        target_sim = sim[torch.arange(en - st, device=device), target_idx]

        better = sim > target_sim[:, None]
        r = 1 + better.sum(dim=1)
        standard.append(r.cpu())

        if cats is not None:
            same = cats[st:en, None].eq(cats[None, :])
            rw = 1 + (better & same).sum(dim=1)
            within.append(rw.cpu())

        print(
            f"[eval-rank] {label} {en}/{len(q)} | elapsed={time.time()-t0:.1f}s",
            flush=True,
        )

    std = torch.cat(standard).numpy()
    win = torch.cat(within).numpy() if within else None

    del q, g, cats
    torch.cuda.empty_cache()
    return std, win


def evaluate_embeddings(z):
    cats = np.asarray(z["category"], dtype=object)
    cat_names, cat_codes = np.unique(cats, return_inverse=True)
    cat_sizes = np.bincount(cat_codes)
    within_eligible = cat_sizes[cat_codes] >= 5

    pairs = [
        ("short_T2I", z["short"], z["image"]),
        ("short_I2T", z["image"], z["short"]),
        ("long_T2I", z["long"], z["image"]),
        ("long_I2T", z["image"], z["long"]),
    ]

    result = {
        "rows": int(len(z["image"])),
        "within_category": {
            "protocol": "same-category exact retrieval; gallery size >=5"
        },
        "by_frequency": {},
    }
    standard_ranks = {}

    for name, query, gallery in pairs:
        print(f"[eval-metrics] START {name}", flush=True)
        r_std, r_within = exact_ranks_gpu(
            query, gallery, category_codes=cat_codes,
            chunk=512, label=name,
        )
        standard_ranks[name] = r_std
        result[name] = metrics_from_ranks(r_std)
        wm = metrics_from_ranks(r_within[within_eligible])
        wm.update({
            "groups": int(np.sum(cat_sizes >= 5)),
            "min_gallery": 5,
            "mean_gallery_size": float(np.mean(cat_sizes[cat_sizes >= 5])),
            "median_gallery_size": float(np.median(cat_sizes[cat_sizes >= 5])),
        })
        result["within_category"][name] = wm
        print(
            f"[eval-metrics] DONE {name} "
            f"R1={result[name]['R@1']:.6f} "
            f"within_R1={wm['R@1']:.6f}",
            flush=True,
        )

    bins = np.asarray(z["freq_bin"], dtype=object)
    for freq in ["tail", "mid", "head"]:
        mask = bins == freq
        result["by_frequency"][freq] = {
            "n": int(mask.sum()),
            "short_T2I": metrics_from_ranks(standard_ranks["short_T2I"][mask]),
            "long_T2I": metrics_from_ranks(standard_ranks["long_T2I"][mask]),
        }

    std = float(np.mean([
        result["short_T2I"]["R@1"],
        result["short_I2T"]["R@1"],
        result["long_T2I"]["R@1"],
        result["long_I2T"]["R@1"],
    ]))
    within = float(np.mean([
        result["within_category"]["short_T2I"]["R@1"],
        result["within_category"]["short_I2T"]["R@1"],
        result["within_category"]["long_T2I"]["R@1"],
        result["within_category"]["long_I2T"]["R@1"],
    ]))
    mrr = float(np.mean([
        result["short_T2I"]["MRR@10"],
        result["short_I2T"]["MRR@10"],
        result["long_T2I"]["MRR@10"],
        result["long_I2T"]["MRR@10"],
    ]))

    result["selection"] = {
        "standard_r1_mean": std,
        "within_category_r1_mean": within,
        "standard_mrr_mean": mrr,
        "primary_selection_metric": "standard_r1_mean",
    }
    print("[eval-metrics] COMPLETE", json.dumps(result["selection"]), flush=True)
    return result


@torch.inference_mode()
def evaluate_model(backbone, processor, manifest_path, device, batch_size, num_workers, max_text_length):
    df = pd.read_csv(manifest_path)
    print(f"[eval] rows={len(df)} batch={batch_size} workers={num_workers}", flush=True)
    z = embed_manifest(
        backbone, processor, df, device,
        batch_size, num_workers, max_text_length
    )
    return evaluate_embeddings(z)

def atomic_json_dump(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    t0 = time.time()
    print(f"[checkpoint] saving {path.name} ...", flush=True)
    torch.save(obj, tmp)
    os.replace(tmp, path)
    print(
        f"[checkpoint] saved {path.name} "
        f"size={path.stat().st_size/1024**3:.2f}GiB "
        f"elapsed={time.time()-t0:.1f}s",
        flush=True,
    )


def repair_best_checkpoint_dirs(out_dir):
    out_dir = Path(out_dir)
    best = out_dir / "best_model"
    tmp = out_dir / "best_model_tmp"
    backup = out_dir / "best_model_backup"

    if not best.exists():
        if backup.exists() and (backup / "model.safetensors").exists():
            os.replace(backup, best)
        elif tmp.exists() and (tmp / "model.safetensors").exists():
            os.replace(tmp, best)

    if tmp.exists():
        shutil.rmtree(tmp)
    if backup.exists():
        shutil.rmtree(backup)


# ============================================================
# Mode: standard eval
# ============================================================


def mode_standard_eval(args):
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    seed_all(args.seed, 0)
    backbone, proc = load_backbone_processor(
        args.model_dir,
        device=device,
        dtype=None,
        log_prefix="[eval-load]",
    )
    metrics = evaluate_model(
        backbone, proc, args.manifest, device,
        args.batch_size, args.num_workers, args.max_text_length
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(metrics, out / "metrics.json")
    print("STANDARD_EVAL", json.dumps(metrics["selection"]), flush=True)

# ============================================================
# Training helpers
# ============================================================

def save_best(wrapper, processor, out_dir, epoch, score):
    out_dir = Path(out_dir)
    best = out_dir / "best_model"
    tmp = out_dir / "best_model_tmp"
    backup = out_dir / "best_model_backup"

    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    wrapper.backbone.save_pretrained(tmp, safe_serialization=True)
    processor.save_pretrained(tmp)

    atomic_json_dump(
        {
            "epoch": int(epoch),
            "standard_r1_mean": float(score),
            "benchmark_logit_scale": float(wrapper.benchmark_logit_scale.detach().cpu()),
            "benchmark_logit_bias": float(wrapper.benchmark_logit_bias.detach().cpu()),
        },
        tmp / "benchmark_head.json",
    )

    weight = tmp / "model.safetensors"
    if not weight.exists() or weight.stat().st_size <= 0:
        raise RuntimeError("Temporary best checkpoint is incomplete")

    if backup.exists():
        shutil.rmtree(backup)
    if best.exists():
        os.replace(best, backup)
    os.replace(tmp, best)
    if backup.exists():
        shutil.rmtree(backup)


def build_train_components(args, device, rank, world_size):
    train_df = pd.read_csv(args.train_manifest)
    val_df = pd.read_csv(args.val_manifest)

    # Training keeps FP32 master weights and uses autocast+GradScaler.
    # This is safer than optimizing FP16 parameters directly.
    if rank > 0:
        time.sleep(1.0 * rank)
    backbone, proc = load_backbone_processor(
        args.model_dir,
        device=device,
        dtype=None,
        log_prefix=f"[rank{rank}-load]",
    )

    if hasattr(backbone, "logit_scale") and backbone.logit_scale.requires_grad:
        raise RuntimeError("Native SigLIP2 logit_scale must be frozen")

    # V4 protocol lock: gradient checkpointing is OFF.
    # Transformers 5 warns that composite CLIPModel does not expose
    # input embeddings for the checkpointing hook. For full fine-tuning
    # this makes token-embedding gradient flow harder to audit.
    if hasattr(backbone, "gradient_checkpointing_disable"):
        backbone.gradient_checkpointing_disable()
    if rank == 0:
        print("GRADIENT_CHECKPOINTING=OFF (protocol lock)", flush=True)

    wrapper = RetrievalBenchmarkModel(
        backbone,
        init_temperature=args.init_temperature,
        init_bias=args.init_logit_bias,
    ).to(device)

    ddp = DDP(
        wrapper,
        device_ids=[device.index],
        output_device=device.index,
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )

    ds = TrainDataset(train_df)
    sampler = DistributedSampler(
        ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=TrainCollator(proc, args.p_long, args.max_text_length),
    )

    optimizer = build_optimizer(ddp.module, args)
    return train_df, val_df, proc, ddp, dl, sampler, optimizer


def train_micro_loss(ddp, batch, device, args):
    cat = batch["category_id"].to(device, non_blocking=True)
    weight = batch["class_weight"].to(device, non_blocking=True)
    px = batch["pixel_values"].to(
        device,
        dtype=next(ddp.module.backbone.parameters()).dtype,
        non_blocking=True,
    )
    ids = batch["input_ids"].to(device, non_blocking=True)
    mask = batch.get("attention_mask")
    if mask is not None:
        mask = mask.to(device, non_blocking=True)

    with torch.autocast("cuda", dtype=torch.float16):
        I, T = ddp(
            pixel_values=px,
            input_ids=ids,
            attention_mask=mask,
        )

    Ig = gather_grad(I)
    Tg = gather_grad(T)
    Cg = gather_nograd(cat)
    Wg = gather_nograd(weight)

    factual = weighted_pairwise_logistic(
        Ig, Tg, Cg, Wg,
        ddp.module.benchmark_logit_scale,
        ddp.module.benchmark_logit_bias,
        args.same_cat_neg_weight,
    )
    return factual


def mode_dryrun(args):
    rank, world_size, _, device = setup_ddp()
    seed_all(args.seed, rank)

    _, _, proc, ddp, dl, sampler, optimizer = build_train_components(
        args, device, rank, world_size
    )
    scaler = make_grad_scaler(args)
    sampler.set_epoch(0)

    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)

    required_success = int(args.dryrun_required_success_steps)
    max_windows = int(args.dryrun_max_windows)
    max_microsteps = min(len(dl), max_windows * args.grad_accum)

    run = torch.zeros(2, device=device, dtype=torch.float64)
    success_steps = 0
    attempted_steps = 0
    overflow_skips = 0
    unused_checked = False
    tested_microsteps = 0

    initial_scale = float(scaler.get_scale())
    min_scale = initial_scale
    max_scale = initial_scale
    t0 = time.time()

    for micro, b in enumerate(dl):
        if micro >= max_microsteps or success_steps >= required_success:
            break

        tested_microsteps += 1
        is_update = ((micro + 1) % args.grad_accum == 0)
        ctx = contextlib.nullcontext() if is_update else ddp.no_sync()

        with ctx:
            loss = train_micro_loss(ddp, b, device, args)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite dryrun loss at microstep {micro}"
                )
            scaler.scale(loss / args.grad_accum).backward()

        run[0] += float(loss.detach())
        run[1] += 1.0

        if not is_update:
            continue

        attempted_steps += 1
        scaler.unscale_(optimizer)

        # Structural DDP correctness gate: every trainable parameter
        # must receive a gradient at the update boundary.
        if not unused_checked:
            unused = [
                n for n, p in ddp.module.named_parameters()
                if p.requires_grad and p.grad is None
            ]
            if unused:
                raise RuntimeError(
                    "Trainable parameters without gradient at DDP update boundary: "
                    + ", ".join(unused[:30])
                )
            unused_checked = True

        finite, _, bad_names = gradient_finiteness(ddp.module)

        if finite:
            grad_norm = clip_grad_norm_(
                ddp.parameters(),
                args.max_grad_norm,
                error_if_nonfinite=True,
            )
            stepped, old_scale, new_scale = amp_optimizer_step(
                scaler, optimizer
            )
            if stepped:
                success_steps += 1
                status = "STEP_OK"
            else:
                overflow_skips += 1
                status = "SCALER_SKIP"
        else:
            # IMPORTANT: do not raise here. PyTorch AMP is designed to
            # skip this optimizer update and lower the scale.
            grad_norm = float("inf")
            stepped, old_scale, new_scale = amp_optimizer_step(
                scaler, optimizer
            )
            if stepped:
                raise RuntimeError(
                    "Optimizer stepped despite non-finite gradients: "
                    + ", ".join(bad_names[:10])
                )
            overflow_skips += 1
            status = "AMP_OVERFLOW_SKIP"

        min_scale = min(min_scale, float(new_scale))
        max_scale = max(max_scale, float(new_scale))
        optimizer.zero_grad(set_to_none=True)

        if rank == 0:
            print(
                f"[dryrun] window={attempted_steps}/{max_windows} "
                f"micro={micro+1} status={status} "
                f"success={success_steps}/{required_success} "
                f"loss={float(loss.detach()):.5f} "
                f"grad_norm={float(grad_norm):.4f} "
                f"amp_scale={old_scale:.1f}->{new_scale:.1f} "
                f"bad_grads={len(bad_names)}",
                flush=True,
            )
            if bad_names:
                print(
                    "[dryrun] nonfinite gradient preview: "
                    + ", ".join(bad_names[:8]),
                    flush=True,
                )

    dist.all_reduce(run)

    success_tensor = torch.tensor(
        [success_steps], device=device, dtype=torch.int64
    )
    attempt_tensor = torch.tensor(
        [attempted_steps], device=device, dtype=torch.int64
    )
    overflow_tensor = torch.tensor(
        [overflow_skips], device=device, dtype=torch.int64
    )

    dist.all_reduce(success_tensor, op=dist.ReduceOp.MIN)
    dist.all_reduce(attempt_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(overflow_tensor, op=dist.ReduceOp.MAX)

    success_steps = int(success_tensor.item())
    attempted_steps = int(attempt_tensor.item())
    overflow_skips = int(overflow_tensor.item())

    mean_loss = float((run[0] / run[1].clamp_min(1)).cpu())
    peak = torch.cuda.max_memory_allocated(device) / 1024**3

    report = {
        "batch_per_gpu": int(args.batch_size),
        "world_size": int(world_size),
        "global_contrastive_batch": int(args.batch_size * world_size),
        "grad_accum": int(args.grad_accum),
        "tested_microsteps": int(tested_microsteps),
        "attempted_optimizer_steps": int(attempted_steps),
        "optimizer_steps": int(success_steps),
        "overflow_skips": int(overflow_skips),
        "loss": mean_loss,
        "peak_GB": float(peak),
        "amp_step": bool(success_steps >= 1),
        "initial_amp_scale": float(initial_scale),
        "final_amp_scale": float(scaler.get_scale()),
        "min_amp_scale": float(min_scale),
        "max_amp_scale": float(max_scale),
        "gradient_checkpointing": False,
        "unused_gradient_gate_passed": bool(unused_checked),
        "elapsed_sec": float(time.time() - t0),
        "ddp_multi_iteration_tested": bool(
            attempted_steps >= required_success
        ),
        "required_success_steps": int(required_success),
        "max_windows": int(max_windows),
    }

    if success_steps < required_success:
        raise RuntimeError(
            "AMP/DDP dry-run could not obtain the required number of real "
            f"optimizer steps: success={success_steps}, "
            f"attempts={attempted_steps}, overflows={overflow_skips}, "
            f"final_scale={scaler.get_scale()}"
        )

    if rank == 0:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(report, out / "dryrun.json")
        print("DRYRUN", json.dumps(report), flush=True)

    dist.barrier()

def mode_train(args):
    rank, world_size, _, device = setup_ddp()
    seed_all(args.seed, rank)

    train_df, val_df, proc, ddp, dl, sampler, optimizer = build_train_components(
        args, device, rank, world_size
    )

    out = Path(args.output_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        repair_best_checkpoint_dirs(out)
        stale_state_tmp = out / "last_state.pt.tmp"
        if stale_state_tmp.exists():
            stale_state_tmp.unlink()

    dist.barrier()

    complete_path = out / "training_complete.json"
    last_path = out / "last_state.pt"

    if complete_path.exists():
        if rank == 0:
            print("TRAINING_ALREADY_COMPLETE", complete_path.read_text(), flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return

    baseline = json.load(open(args.baseline_val_metrics))
    base_score = float(baseline["selection"]["standard_r1_mean"])
    best_score = base_score
    best_epoch = -1
    bad_epochs = 0
    update_step = 0

    if rank == 0 and not last_path.exists():
        # epoch=-1 is the untouched base safety checkpoint.
        save_best(ddp.module, proc, out, -1, base_score)
        atomic_json_dump(
            {
                "protocol": "COMMON_BACKBONE_BENCHMARK_V4_STABLE",
                "base_standard_r1": base_score,
                "batch_per_gpu": args.batch_size,
                "world_size": world_size,
                "global_contrastive_batch": args.batch_size * world_size,
                "grad_accum": args.grad_accum,
                "effective_optimizer_batch": args.batch_size * world_size * args.grad_accum,
                "vision_lr": args.vision_lr,
                "text_lr": args.text_lr,
                "other_lr": args.other_lr,
                "epochs": args.epochs,
                "patience": args.patience,
                "min_epochs_before_stop": args.min_epochs_before_stop,
                "same_cat_neg_weight": args.same_cat_neg_weight,
                "p_long": args.p_long,
                "max_text_length": args.max_text_length,
                "gradient_checkpointing": False,
                "amp_init_scale": args.amp_init_scale,
                "amp_growth_factor": args.amp_growth_factor,
                "amp_backoff_factor": args.amp_backoff_factor,
                "amp_growth_interval": args.amp_growth_interval,
                "max_consecutive_amp_skips": args.max_consecutive_amp_skips,
                "primary_selection_metric": "validation standard_r1_mean",
            },
            out / "run_config.json",
        )
        print("INITIAL_BEST=BASE", base_score, flush=True)

    dist.barrier()

    total_updates = math.ceil(len(dl) / args.grad_accum) * args.epochs
    warmup = int(round(total_updates * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup,
        num_training_steps=total_updates,
    )
    scaler = make_grad_scaler(args)

    history_path = out / "history.json"
    history = []
    start_epoch = 0

    if history_path.exists():
        try:
            history = json.load(open(history_path))
        except Exception:
            history = []

    dist.barrier()
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        ddp.module.load_state_dict(state["wrapper"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        best_score = float(state["best_score"])
        best_epoch = int(state["best_epoch"])
        bad_epochs = int(state["bad_epochs"])
        update_step = int(state["update_step"])
        amp_overflow_skips = int(state.get("amp_overflow_skips", 0))
        start_epoch = int(state["epoch"]) + 1
        if rank == 0:
            print(f"RESUME_FROM_EPOCH={start_epoch}", flush=True)

    if not last_path.exists():
        amp_overflow_skips = 0
    consecutive_amp_skips = 0

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        ddp.train()
        optimizer.zero_grad(set_to_none=True)

        loss_sum = torch.zeros(4, device=device, dtype=torch.float64)
        t0 = time.time()

        for step, b in enumerate(dl):
            window_start = (step // args.grad_accum) * args.grad_accum
            window_end = min(window_start + args.grad_accum, len(dl))
            window_size = window_end - window_start
            is_update = (step + 1 == window_end)
            sync_ctx = contextlib.nullcontext() if is_update else ddp.no_sync()

            with sync_ctx:
                total = train_micro_loss(ddp, b, device, args)
                if not torch.isfinite(total):
                    raise RuntimeError(
                        f"Non-finite training loss at epoch={epoch} step={step}"
                    )
                scaler.scale(total / window_size).backward()

            loss_sum[0] += float(total.detach())
            loss_sum[3] += 1.0

            step_status = None
            grad_norm = float("nan")
            old_scale = float(scaler.get_scale())
            new_scale = old_scale

            if is_update:
                scaler.unscale_(optimizer)
                finite_grads, _, bad_names = gradient_finiteness(ddp.module)

                if finite_grads:
                    grad_norm = clip_grad_norm_(
                        ddp.parameters(),
                        args.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                    stepped, old_scale, new_scale = amp_optimizer_step(
                        scaler, optimizer, scheduler
                    )
                else:
                    grad_norm = float("inf")
                    stepped, old_scale, new_scale = amp_optimizer_step(
                        scaler, optimizer, scheduler
                    )
                    if stepped:
                        raise RuntimeError(
                            "Optimizer stepped despite non-finite gradients: "
                            + ", ".join(bad_names[:10])
                        )

                optimizer.zero_grad(set_to_none=True)

                if stepped:
                    update_step += 1
                    consecutive_amp_skips = 0
                    step_status = "STEP_OK"
                else:
                    amp_overflow_skips += 1
                    consecutive_amp_skips += 1
                    step_status = "AMP_OVERFLOW_SKIP"

                    if consecutive_amp_skips > args.max_consecutive_amp_skips:
                        raise RuntimeError(
                            "Too many consecutive AMP overflow skips: "
                            f"{consecutive_amp_skips}; scale={new_scale}; "
                            f"bad_grad_preview={bad_names[:10]}"
                        )

            if rank == 0 and (
                step == 0
                or (step + 1) % 100 == 0
                or step + 1 == len(dl)
                or step_status == "AMP_OVERFLOW_SKIP"
            ):
                print(
                    f"[train] epoch={epoch} step={step+1}/{len(dl)} "
                    f"loss={float(total.detach()):.5f} "
                    f"updates={update_step} "
                    f"status={step_status or 'ACCUM'} "
                    f"grad_norm={float(grad_norm):.4f} "
                    f"amp_scale={old_scale:.1f}->{new_scale:.1f} "
                    f"overflow_skips={amp_overflow_skips} "
                    f"vram={torch.cuda.memory_allocated(device)/1024**3:.2f}GiB",
                    flush=True,
                )

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        means = (loss_sum[:1] / loss_sum[3].clamp_min(1.0)).cpu().tolist()

        dist.barrier()

        epoch_score = None
        epoch_metrics = None
        improved = False
        stop = False

        if rank == 0:
            ddp.module.backbone.eval()
            epoch_metrics = evaluate_model(
                ddp.module.backbone,
                proc,
                args.val_manifest,
                device,
                args.eval_batch_size,
                args.num_workers,
                args.max_text_length,
            )
            epoch_score = float(epoch_metrics["selection"]["standard_r1_mean"])

            val_dir = out / "val_history"
            val_dir.mkdir(parents=True, exist_ok=True)
            atomic_json_dump(
                epoch_metrics,
                val_dir / f"epoch_{epoch}.json",
            )

            improved = epoch_score > best_score + 1e-12
            if improved:
                best_score = epoch_score
                best_epoch = epoch
                bad_epochs = 0
                save_best(ddp.module, proc, out, best_epoch, best_score)
            else:
                bad_epochs += 1

            row = {
                "epoch": int(epoch),
                "train_loss": float(means[0]),
                "val_standard_r1": float(epoch_score),
                "best_standard_r1": float(best_score),
                "best_epoch": int(best_epoch),
                "improved": bool(improved),
                "bad_epochs": int(bad_epochs),
                "elapsed_sec": float(time.time() - t0),
                "update_step": int(update_step),
                "amp_overflow_skips": int(amp_overflow_skips),
                "amp_scale": float(scaler.get_scale()),
            }
            history.append(row)
            atomic_json_dump(history, out / "history.json")
            print("BENCHMARK_EPOCH", json.dumps(row), flush=True)

            # Save resumable state after each completed epoch.
            atomic_torch_save(
                {
                    "epoch": epoch,
                    "wrapper": ddp.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "bad_epochs": bad_epochs,
                    "update_step": update_step,
                    "amp_overflow_skips": amp_overflow_skips,
                },
                last_path,
            )

            stop = (
                (epoch + 1) >= args.min_epochs_before_stop
                and bad_epochs >= args.patience
            )

        state = torch.zeros(4, device=device, dtype=torch.float64)
        if rank == 0:
            state[0] = best_score
            state[1] = best_epoch
            state[2] = bad_epochs
            state[3] = 1.0 if stop else 0.0
        dist.broadcast(state, src=0)
        best_score = float(state[0].item())
        best_epoch = int(state[1].item())
        bad_epochs = int(state[2].item())
        stop = bool(state[3].item() > 0.5)

        dist.barrier()
        if stop:
            if rank == 0:
                print("EARLY_STOP", {"epoch": epoch, "best_epoch": best_epoch})
            break

    if rank == 0:
        complete = {
            "completed": True,
            "best_epoch": int(best_epoch),
            "best_standard_r1": float(best_score),
            "base_standard_r1": float(base_score),
            "gain": float(best_score - base_score),
            "best_is_base": bool(best_epoch == -1),
            "amp_overflow_skips": int(amp_overflow_skips),
            "final_amp_scale": float(scaler.get_scale()),
            "best_model": str(out / "best_model"),
        }
        atomic_json_dump(complete, complete_path)
        print("BENCHMARK_COMPLETE", json.dumps(complete, indent=2))

    dist.barrier()
    dist.destroy_process_group()


def main():
    args = parse_args()
    try:
        if args.mode == "standard_eval":
            mode_standard_eval(args)
        elif args.mode == "dryrun":
            mode_dryrun(args)
        elif args.mode == "train":
            mode_train(args)
        else:
            raise ValueError(args.mode)
    finally:
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    main()
