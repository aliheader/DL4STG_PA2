from functools import lru_cache
from pathlib import Path

import torch

from .transforms import G, allowed_starts, apply_g_per_sample, batch_windows

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {"train": (0, 8000), "val": (8000, 9000), "test": (9000, 10000)}


@lru_cache(maxsize=1)
def load_all():
    """All 10,000 videos as uint8, shape (10000, 20, 1, 64, 64)."""
    from torchvision.datasets import MovingMNIST

    data = MovingMNIST(root=str(ROOT / "data"), split=None, download=True).data
    if data.dim() == 4:  # (N, T, H, W) -> add the channel axis
        data = data.unsqueeze(2)
    assert tuple(data.shape) == (10000, 20, 1, 64, 64), data.shape
    return data.contiguous()


def load_split(name):
    """uint8 videos of one split (split by video index, never by frame)."""
    lo, hi = SPLITS[name]
    return load_all()[lo:hi]


def to_float(x):
    return x.float().div_(255.0)


class WindowSampler:
    """One random window per video, following the model's training protocol.

    speed_aug: k ~ U{1, 2, 3}, then t ~ U(allowed starts for k)   (C, D); otherwise k = 1, t ~ U{1..11}
    sym_aug:   g ~ U(G), applied to both input and target          (B)
    """

    def __init__(self, speed_aug, sym_aug, generator):
        self.speed_aug, self.sym_aug, self.gen = speed_aug, sym_aug, generator
        self.starts = {
            k: torch.tensor(allowed_starts(k, speed_aug))
            for k in ((1, 2, 3) if speed_aug else (1,))
        }

    def sample(self, videos_u8):
        """Returns X, Y, the target mask and the drawn (k, t, g) of every window."""
        B = len(videos_u8)
        ks = (
            torch.randint(0, 3, (B,), generator=self.gen) + 1
            if self.speed_aug
            else torch.ones(B, dtype=torch.long)
        )
        ts = torch.empty(B, dtype=torch.long)
        for k, starts in self.starts.items():
            sel = ks == k
            if sel.any():
                ts[sel] = starts[
                    torch.randint(0, len(starts), (int(sel.sum()),), generator=self.gen)
                ]
        X, Y, mask = batch_windows(videos_u8, ts, ks)
        g = None
        if self.sym_aug:
            g = torch.randint(0, len(G), (B,), generator=self.gen)
            X, Y = apply_g_per_sample(X, g), apply_g_per_sample(Y, g)
        return X, Y, mask, dict(k=ks, t=ts, g=g)
