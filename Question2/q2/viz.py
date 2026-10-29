import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np

from .data import ROOT
from .evaluate import TRANSFORMS, get_device, predict
from .transforms import window

FIG_DIR = ROOT / "figures"
GIF_DIR = ROOT / "gifs"

_DILATE = np.ones((5, 5), np.uint8)


def n_digit_blobs(frame):
    m = cv2.dilate((np.asarray(frame) > 0.2).astype(np.uint8), _DILATE)
    return cv2.connectedComponents(m)[0] - 1


def figure5_frames():
    return sorted({1 + k * j for k in (1, 2, 3) for j in range(7) if 1 + k * j <= 20})


def first_clean_video(test_vids_float):
    for i in range(len(test_vids_float)):
        if all(
            n_digit_blobs(test_vids_float[i, f - 1, 0].numpy()) == 2
            for f in figure5_frames()
        ):
            return i
    raise RuntimeError("no non-overlapping test video found")


def _img(frame):
    return np.asarray(frame).squeeze()


def save_grid_figure(rows, col_titles, row_labels, title, out_path):
    n_rows, n_cols = len(rows), len(rows[0])
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 1.3, n_rows * 1.3 + 0.4),
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
        squeeze=False,
    )
    for r, row in enumerate(rows):
        for c, frame in enumerate(row):
            ax = axes[r][c]
            ax.imshow(_img(frame), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=8, pad=3)
            if c == 0:
                ax.set_ylabel(
                    row_labels[r], fontsize=8, rotation=0, labelpad=35, va="center"
                )
    fig.suptitle(title, fontsize=10, y=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_figure2(model, test_vids_float, video_indices, out_path=None):
    out_path = out_path or (FIG_DIR / "figure2_predictions.png")
    X, Y, _ = window(test_vids_float[video_indices], t=1, k=1)
    P = predict(model, X, get_device())
    blank = [np.zeros((64, 64), np.float32)] * 5
    rows, labels = [], []
    for i, v in enumerate(video_indices):
        rows += [list(X[i]) + blank, blank + list(P[i]), blank + list(Y[i])]
        labels += [f"V{v}\nInput", f"V{v}\nPred", f"V{v}\nTrue"]
    cols = [f"t={i}" for i in range(1, 6)] + [f"h={h}" for h in range(1, 6)]
    save_grid_figure(
        rows,
        cols,
        labels,
        "Figure 2: Input (t=1..5), Prediction (h=1..5), and Ground Truth (h=1..5) "
        "for 3 test videos",
        out_path,
    )


def make_figure3_panel(
    v_idx, X, Y, f_X, f_TX, T_name, out_path, gif_path=None, fig_name="Figure 3"
):

    T = TRANSFORMS[T_name]
    TX = T(X)
    rows = [
        list(X) + list(f_X),
        list(TX) + list(T(Y)),
        list(TX) + list(f_TX),
        list(TX) + list(T(f_X)),
    ]
    labels = [
        "X | f(X)",
        f"{T_name}(X) | {T_name}(Y)",
        f"{T_name}(X) | f({T_name}(X))",
        f"{T_name}(X) | {T_name}(f(X))",
    ]
    cols = [f"x{i}" for i in range(1, 6)] + [f"h{h}" for h in range(1, 6)]
    save_grid_figure(
        rows,
        cols,
        labels,
        f"{fig_name}: Test Video {v_idx} under transformation {T_name}",
        out_path,
    )
    if gif_path:
        _save_panel_gif(rows, labels, v_idx, T_name, fig_name, gif_path)


def _save_panel_gif(rows, labels, v_idx, T_name, fig_name, gif_path):
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for h in range(5):
        fig, axes = plt.subplots(
            4, 2, figsize=(4.5, 7.5), gridspec_kw={"wspace": 0.05, "hspace": 0.1}
        )
        for r, row in enumerate(rows):
            for c, frame in enumerate((row[h], row[5 + h])):
                axes[r][c].imshow(_img(frame), cmap="gray", vmin=0, vmax=1)
                axes[r][c].set_xticks([])
                axes[r][c].set_yticks([])
            axes[r][0].set_ylabel(
                labels[r], fontsize=7, rotation=0, labelpad=30, va="center"
            )
        axes[0][0].set_title(f"Input t={h + 1}", fontsize=8)
        axes[0][1].set_title(f"Forecast h={h + 1}", fontsize=8)
        fig.suptitle(
            f"{fig_name}: Video {v_idx} ({T_name}) Horizon h={h + 1}", fontsize=9
        )
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba()))
        plt.close(fig)
    imageio.mimsave(gif_path, frames, duration=0.8, loop=0)


def make_figure5(model, test_vids_float, v_idx, out_path=None, fig_name="Figure 5"):
    out_path = out_path or (FIG_DIR / "figure5_speeds.png")
    vid = test_vids_float[v_idx : v_idx + 1]
    rows = []
    for k in (1, 2, 3):
        X, Y, _ = window(vid, t=1, k=k)
        P = predict(model, X, get_device())
        rows.append(list(X[0]) + list(P[0, :2]) + list(Y[0, :2]))
    cols = [f"in_{i + 1}" for i in range(5)] + [
        "pred_h1",
        "pred_h2",
        "true_h1",
        "true_h2",
    ]
    save_grid_figure(
        rows,
        cols,
        ["1×", "2×", "3×"],
        f"{fig_name}: Video {v_idx} at 1×, 2×, and 3× speeds "
        "(inputs, first 2 preds, and ground truth)",
        out_path,
    )
