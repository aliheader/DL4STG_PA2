import torch

N_FRAMES, N_IN, N_OUT = 20, 5, 5


def mirror(x):
    return x.flip(-1)


def rotate(x, a=1):
    return torch.rot90(x, k=a, dims=(-2, -1))


G = [(0, a) for a in range(4)] + [(1, a) for a in range(4)]
G_NAMES = [f"R{a}" for a in range(4)] + [f"MR{a}" for a in range(4)]


def apply_g(x, g):
    m, a = g
    x = rotate(x, a) if a else x
    return mirror(x) if m else x


def inverse_g(x, g):
    m, a = g
    if m:
        return apply_g(x, g)
    return rotate(x, -a) if a else x


def apply_g_per_sample(x, g_idx):
    out = x.clone()
    for gi in torch.unique(g_idx).tolist():
        sel = g_idx == gi
        out[sel] = apply_g(x[sel], G[gi])
    return out


def window_frames(t, k):
    inputs = [t + k * j for j in range(N_IN)]
    targets = [f for f in (t + k * (N_IN + j) for j in range(N_OUT)) if f <= N_FRAMES]
    return inputs, targets


def allowed_starts(k, augmented):
    need = 2 if augmented else N_OUT
    return [
        t
        for t in range(1, N_FRAMES + 1)
        if t + (N_IN - 1) * k <= N_FRAMES and len(window_frames(t, k)[1]) >= need
    ]


def window_indices(t, k):
    inputs, targets = window_frames(t, k)
    n = len(targets)
    targets = targets + [targets[-1]] * (N_OUT - n)
    return (
        [f - 1 for f in inputs],
        [f - 1 for f in targets],
        [1.0] * n + [0.0] * (N_OUT - n),
    )


def window(video, t, k=1):

    ii, ti, mask = window_indices(t, k)
    mask = torch.tensor(mask)
    X = video[..., ii, :, :, :]
    Y = video[..., ti, :, :, :].clone()
    Y[..., mask == 0, :, :, :] = 0
    return X, Y, mask


def batch_windows(videos, ts, ks):
    ii, ti, mask = zip(*[window_indices(int(t), int(k)) for t, k in zip(ts, ks)])
    rows = torch.arange(len(videos))[:, None]
    return (
        videos[rows, torch.tensor(ii)],
        videos[rows, torch.tensor(ti)],
        torch.tensor(mask),
    )
