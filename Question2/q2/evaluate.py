import time

import numpy as np
import torch

from .data import load_split, to_float
from .transforms import mirror, rotate, window

TRANSFORMS = {"M": mirror, "R": lambda x: rotate(x, 1)}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    """Wait for queued GPU work, so wall-clock timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def predict(model, X, device, batch_size=32):
    """Forecasts for float windows X (N, 5, 1, 64, 64), as a CPU tensor."""
    model.eval()
    return torch.cat(
        [
            model(X[i : i + batch_size].to(device)).float().cpu()
            for i in range(0, len(X), batch_size)
        ]
    )


def per_horizon_mse(P, Y):
    """(N, 5) matrix of MSE_h over the 64x64 pixels."""
    return ((P - Y) ** 2).flatten(2).mean(-1).numpy()


def test_windows(k=1, t=1, split="test"):
    """X, Y and target mask of every video of a split, at speed k and start t."""
    return window(to_float(load_split(split)), t, k)


def E(mse, n_h):
    """E^(n): mean over the first n horizons, then over videos."""
    return float(mse[:, :n_h].mean())


def full_evaluation(model, device, split="test"):
    """Per-video arrays and the scalars of Tables 2-6 for one forecaster."""
    X, Y, _ = test_windows(1, 1, split)
    P = predict(model, X, device)
    out = {"mse_1x": per_horizon_mse(P, Y)}
    for name, T in TRANSFORMS.items():
        PT = predict(model, T(X), device)  # f(T(X))
        out[f"C_{name}_pv"] = per_horizon_mse(PT, T(P)).mean(1)
        out[f"E_{name}_pv"] = per_horizon_mse(PT, T(Y)).mean(1)
    for k in (2, 3):
        Xk, Yk, mask = test_windows(k, 1, split)
        out[f"mse_{k}x"] = per_horizon_mse(predict(model, Xk, device), Yk)[
            :, : int(mask.sum())
        ]

    scal = {
        "MSE_h": [float(v) for v in out["mse_1x"].mean(0)],
        "E5_1x": E(out["mse_1x"], 5),
        "E2_1x": E(out["mse_1x"], 2),
        "E2_2x": E(out["mse_2x"], 2),
        "E5_2x": E(out["mse_2x"], 5),
        "E2_3x": E(out["mse_3x"], 2),
    }
    for name in TRANSFORMS:
        scal[f"C_{name}"] = float(out[f"C_{name}_pv"].mean())
        scal[f"E_{name}"] = float(out[f"E_{name}_pv"].mean())
    return out, scal


@torch.no_grad()
def inference_time(model, device, batch_size=32, warmup=2):
    """Seconds to predict all 1,000 original (1x, t=1) test videos; each batch is on the device before the clock starts."""
    X, _, _ = test_windows(1, 1, "test")
    model.eval()
    batches = [X[i : i + batch_size].to(device) for i in range(0, len(X), batch_size)]
    for b in batches[:warmup]:  # untimed warm-up
        model(b)
    total = 0.0
    for b in batches:
        sync(device)
        t0 = time.perf_counter()
        model(b)
        sync(device)
        total += time.perf_counter() - t0
    return total


def fmt(x):
    """3 significant figures (keeping trailing zeros); scientific notation below 1e-3."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if x == 0:
        return "0"
    return f"{x:.2e}" if abs(x) < 1e-3 else f"{x:#.3g}"


def fmt_seeds(vals):
    """'mean (seed 0, seed 1)'."""
    vals = list(vals)
    if len(vals) == 1:
        return fmt(vals[0])
    return f"{fmt(float(np.mean(vals)))} ({', '.join(fmt(v) for v in vals)})"
