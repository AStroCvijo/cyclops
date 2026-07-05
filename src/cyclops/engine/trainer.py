"""Training loop for one experiment.

Reads everything from the config: dataset, model, optimizer, schedule, loss.
Each epoch it trains over the train split, evaluates on the test split, logs to
W&B (if enabled), and saves `last.pt` plus `best.pt` (best abs_rel).
"""

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cyclops.data.datasets import build_dataset
from cyclops.engine.evaluator import evaluate
from cyclops.models.build import build_model
from cyclops.models.losses import SiLogLoss
from cyclops.utils.checkpoint import save_checkpoint
from cyclops.utils.device import dataloader_workers, pin_memory_for, resolve_device
from cyclops.utils.logging import get_logger, init_wandb

log = get_logger()


# Function for choosing which parameters train (all, or only named submodules)
def set_trainable(model, trainable):
    if trainable is None:
        return list(model.parameters())
    for p in model.parameters():
        p.requires_grad = False
    params = []
    for name in trainable:                       # e.g. ["decoder"] or ["fusion", "decoder"]
        module = getattr(model, name)
        for p in module.parameters():
            p.requires_grad = True
        params += list(module.parameters())
    return params


# Function for a linear-warmup then cosine-decay LR schedule (per epoch)
def make_scheduler(optimizer, epochs, warmup_epochs):
    def factor(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# Function for training one experiment end-to-end from its config
def train(cfg):
    device = resolve_device(cfg.get("device", "auto"))
    torch.manual_seed(cfg["seed"])
    tr = cfg["training"]
    workers = dataloader_workers(tr["num_workers"])
    pin_memory = pin_memory_for(device)
    log.info(f"device: {device}")

    train_set = build_dataset(cfg, "train")
    test_set = build_dataset(cfg, "test")
    train_loader = DataLoader(
        train_set, batch_size=tr["batch_size"], shuffle=True,
        num_workers=workers, pin_memory=pin_memory, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg["eval"]["batch_size"], shuffle=False,
        num_workers=workers, pin_memory=pin_memory,
    )

    model = build_model(cfg).to(device)
    params = set_trainable(model, tr["trainable"])
    log.info(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M")

    opt = tr["optimizer"]
    optimizer = torch.optim.AdamW(params, lr=opt["lr"], weight_decay=opt["weight_decay"])
    scheduler = make_scheduler(optimizer, tr["epochs"], tr["scheduler"]["warmup_epochs"])
    loss_fn = SiLogLoss(lam=tr["loss"]["lambda"])

    use_amp = tr["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    run = init_wandb(cfg)
    ckpt_dir = Path(cfg["checkpoint"]["dir"]) / cfg["experiment"]["name"] / "checkpoints"
    best = float("inf")
    step = 0

    for epoch in range(tr["epochs"]):
        model.train()
        for batch in train_loader:
            image = batch["image"].to(device)
            depth = batch["depth"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device.type, enabled=use_amp):
                pred = model(image)
                loss = loss_fn(pred, depth, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, tr["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            if step % cfg["logging"]["log_interval"] == 0:
                lr = scheduler.get_last_lr()[0]
                log.info(f"epoch {epoch} step {step} loss {loss.item():.4f} lr {lr:.2e}")
                if run:
                    run.log({"train/loss": loss.item(), "train/lr": lr}, step=step)
            step += 1

        scheduler.step()

        metrics = evaluate(model, test_loader, device, cfg)
        log.info(f"epoch {epoch} eval {metrics}")
        if run:
            run.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)

        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epoch, metrics)
        if metrics["abs_rel"] < best:
            best = metrics["abs_rel"]
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch, metrics)
            log.info(f"new best abs_rel {best:.4f}")

    if run:
        run.finish()
    log.info(f"done. best abs_rel {best:.4f}")
