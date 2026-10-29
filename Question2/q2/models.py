import torch
import torch.nn as nn
import torch.nn.functional as F

from .transforms import G, apply_g, inverse_g

N_IN, N_STEPS = 5, 9


class ConvLSTMCell(nn.Module):
    """ConvLSTM without peepholes (Eqs. 3-4): one convolution produces the gates [i, f, o, g]."""

    def __init__(self, in_ch, hidden, kernel=3, dilation=1):
        super().__init__()
        self.reach_cells = dilation * (kernel - 1) // 2
        self.conv = nn.Conv2d(
            in_ch + hidden,
            4 * hidden,
            kernel,
            padding=self.reach_cells,
            dilation=dilation,
        )

    def forward(self, u, state):
        h, c = state
        i, f, o, g = self.conv(torch.cat([u, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class ConvLSTMForecaster(nn.Module):
    """pixel_unshuffle(p) -> ConvLSTM 1 -> ConvLSTM 2 -> 1x1 conv -> pixel_shuffle(p) -> sigmoid,
    rolled out for 9 steps without teacher forcing. Model A is ConvLSTMForecaster()."""

    def __init__(self, patch=4, hidden=64, kernels=(3, 3), dilations=(1, 1)):
        super().__init__()
        self.patch, self.hidden = patch, hidden
        self.config = dict(
            patch=patch, hidden=hidden, kernels=list(kernels), dilations=list(dilations)
        )
        c = patch * patch  # channels of a folded 1-channel frame
        self.cell1 = ConvLSTMCell(c, hidden, kernels[0], dilations[0])
        self.cell2 = ConvLSTMCell(hidden, hidden, kernels[1], dilations[1])
        self.head = nn.Conv2d(hidden, c, kernel_size=1)

    def forward(self, X):
        """X: (B, 5, 1, H, W) -> predictions (B, 5, 1, H, W)."""
        B, T, _, H, W = X.shape
        assert T == N_IN
        z = X.new_zeros(B, self.hidden, H // self.patch, W // self.patch)
        s1 = s2 = (z, z)
        outs, pred = [], None
        for tau in range(1, N_STEPS + 1):
            u = (
                X[:, tau - 1] if tau <= N_IN else pred
            )  # own prediction fed back from step 6 on
            s1 = self.cell1(F.pixel_unshuffle(u, self.patch), s1)
            s2 = self.cell2(s1[0], s2)
            pred = torch.sigmoid(F.pixel_shuffle(self.head(s2[0]), self.patch))
            if tau >= N_IN:
                outs.append(pred)
        return torch.stack(outs, dim=1)

    def reach_pixels(self):
        """One-step reach (Q2.3 / Q2.12): two spatial convolutions in sequence, in original pixels."""
        return (self.cell1.reach_cells + self.cell2.reach_cells) * self.patch


class Blank(nn.Module):
    def forward(self, X):
        return torch.zeros_like(X)


class CopyLast(nn.Module):
    def forward(self, X):
        return X[:, -1:].expand_as(X).clone()


class Symmetrised(nn.Module):
    """f_bar(X) = 1/8 sum_g g^-1 f(g X)  (Eq. 7): exactly G-equivariant for any f."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, X):
        Pg = self.base(torch.cat([apply_g(X, g) for g in G])).split(
            len(X)
        )  # one batched pass
        return torch.stack([inverse_g(p, g) for p, g in zip(Pg, G)]).mean(0)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


D_CANDIDATES = {
    "D1": dict(patch=4, hidden=64, kernels=(3, 3), dilations=(1, 2)),
    "D2": dict(patch=4, hidden=53, kernels=(3, 5), dilations=(1, 1)),
    "D3": dict(patch=4, hidden=46, kernels=(5, 5), dilations=(1, 1)),
    "D4": dict(patch=8, hidden=64, kernels=(3, 3), dilations=(1, 1)),
}
D_CHOICE = "D3"
PARAM_LIMIT_D = int(1.5 * 480_784)

PROTOCOLS = {
    "A": (False, False),
    "B": (False, True),
    "C": (True, False),
    "D": (True, False),
}


def build(name):
    if name == "A":
        return ConvLSTMForecaster()
    return ConvLSTMForecaster(**D_CANDIDATES[D_CHOICE if name == "D" else name])


def ckpt_stem(model_name):

    return f"D_{D_CHOICE}" if model_name == "D" else model_name


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = ConvLSTMForecaster(**ck["config"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


def empirical_reach(model, H=64):
    model = model.double().eval()
    p, g = model.patch, H // model.patch
    torch.manual_seed(0)
    u = torch.rand(1, p * p, g, g, dtype=torch.double, requires_grad=True)
    h1, c1, h2, c2 = (
        torch.randn(1, model.hidden, g, g, dtype=torch.double, requires_grad=True)
        for _ in range(4)
    )
    nh1, _ = model.cell1(u, (h1, c1))
    nh2, _ = model.cell2(nh1, (h2, c2))
    ci = g // 2  # centre cell
    F.pixel_shuffle(model.head(nh2), p)[0, 0, ci * p, ci * p].backward()
    reach = {}
    for name, t in [
        ("u_tau", u),
        ("h1_prev", h1),
        ("c1_prev", c1),
        ("h2_prev", h2),
        ("c2_prev", c2),
    ]:
        nz = torch.nonzero(t.grad[0].abs().sum(0) > 0)
        reach[name] = (nz - ci).abs().max().item() * p
    return reach
