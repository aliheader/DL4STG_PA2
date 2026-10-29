import json
import platform
import subprocess
import time

import torch

from .data import ROOT, WindowSampler, load_split, to_float
from .evaluate import get_device, per_horizon_mse, predict, sync, test_windows
from .models import D_CHOICE, PARAM_LIMIT_D, PROTOCOLS, build, ckpt_stem, n_params


def hardware_string(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    if device.type == "mps":
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        )
        return f"{chip.stdout.strip()} (MPS), torch {torch.__version__}"
    return platform.processor()


def masked_loss(pred, target, mask):
    per_h = ((pred - target) ** 2).flatten(2).mean(-1)  # (B, 5)
    return ((per_h * mask).sum(1) / mask.sum(1)).mean()


def train(model_name, seed, epochs=30, batch=32, lr=1e-3):
    device = get_device()
    speed_aug, sym_aug = PROTOCOLS[model_name]
    torch.manual_seed(seed)  # initialisation
    model = build("D" if model_name == "D" else "A").to(device)
    gen = torch.Generator().manual_seed(seed)
    P = n_params(model)
    if model_name == "D":
        assert P <= PARAM_LIMIT_D and model.reach_pixels() > 8, P
    else:
        assert P == 480_784, P
    name = f"{ckpt_stem(model_name)}_s{seed}"
    print(
        f"[{name}] params={P} reach={model.reach_pixels()}px speed_aug={speed_aug} sym_aug={sym_aug} "
        f"device={device}",
        flush=True,
    )

    train_u8 = load_split("train")
    Xv, Yv, _ = test_windows(1, 1, "val")
    sampler = WindowSampler(speed_aug, sym_aug, gen)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log = dict(
        name=name,
        model=model_name,
        arch=D_CHOICE if model_name == "D" else "A",
        config=model.config,
        seed=seed,
        params=P,
        reach_px=model.reach_pixels(),
        lr=lr,
        batch=batch,
        n_epochs=epochs,
        speed_aug=speed_aug,
        sym_aug=sym_aug,
        hardware=hardware_string(device),
        epochs=[],
    )
    for epoch in range(1, epochs + 1):
        model.train()
        sync(device)
        t0 = time.perf_counter()
        perm = torch.randperm(len(train_u8), generator=gen)
        losses = []
        for i in range(0, len(train_u8), batch):
            X, Y, mask, _ = sampler.sample(train_u8[perm[i : i + batch]])
            loss = masked_loss(
                model(to_float(X).to(device)), to_float(Y).to(device), mask.to(device)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sync(device)
        ep_time = time.perf_counter() - t0
        val_e5 = float(per_horizon_mse(predict(model, Xv, device), Yv).mean())
        log["epochs"].append(
            dict(
                epoch=epoch,
                train_loss=sum(losses) / len(losses),
                val_E5_1x=val_e5,
                time_s=ep_time,
            )
        )
        print(
            f"[{name}] epoch {epoch:2d} loss {sum(losses) / len(losses):.5f} val E5 {val_e5:.5f} "
            f"time {ep_time:.1f}s",
            flush=True,
        )

    log["train_time_per_epoch_s"] = sum(e["time_s"] for e in log["epochs"]) / epochs
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    (ROOT / "results" / "train_logs").mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            config=model.config,
            state_dict=model.state_dict(),
            model=model_name,
            seed=seed,
        ),
        ROOT / "checkpoints" / f"{name}.pt",
    )
    with open(ROOT / "results" / "train_logs" / f"{name}.json", "w") as f:
        json.dump(log, f, indent=1)
    print(
        f"[{name}] done. mean epoch time {log['train_time_per_epoch_s']:.1f}s",
        flush=True,
    )
    return log
