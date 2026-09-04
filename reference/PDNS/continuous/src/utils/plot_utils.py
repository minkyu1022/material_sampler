import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcol
from matplotlib.patches import Circle, Ellipse, Rectangle

import torch

# cm1 = cm.get_cmap("Greens")
cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["b","r"])

fontsize = 10
plt.rcParams.update({"font.size": fontsize})

# https://stackoverflow.com/a/42951885
plt.rcParams['axes.axisbelow'] = True

cpuize = lambda t: t.detach().cpu() if isinstance(t, torch.Tensor) else t

################################################################################################


def get_fig_axes(ncol, nrow=1, ax_length_in=2.0, lim=None):
    figsize = (ncol * ax_length_in, nrow * ax_length_in)
    fig = plt.figure(figsize=figsize)
    axes = fig.subplots(nrow, ncol)

    if lim is not None:
        axs = [axes] if nrow == 1 and ncol == 1 else axes.reshape(-1)
        for ax in axs:
            ax.set(xlim=[-lim, lim], ylim=[-lim, lim])

    return fig, axes


def save_fig(fn, pdf=False):
    plt.tight_layout()
    if pdf:
        plt.savefig(f"figs/{fn}.pdf")
    else:
        plt.savefig(f"figs/{fn}.png", dpi=300)
    plt.close()


def get_colors(n_snapshot):
    colors = cm1(np.linspace(0.2, 0.8, n_snapshot))
    return colors


def plot_scatter(ax, x, s=2, c=None, marker=None, title=None, alpha=1.0):
    """
    x: (B, 2)
    """
    x = cpuize(x)
    ax.scatter(x[:, 0], x[:, 1], s=s, c=c, marker=marker, alpha=alpha)
    if title:
        ax.set_title(title)
    ax.grid(True)


def plot_toy(ts, xs, target, lim=None):
    N, B, D = xs.shape
    assert ts.shape == (N,)
    if target is not None: assert target.shape == (B, D)

    ncol = N if target is None else (N + 1)
    fig, axs = get_fig_axes(ncol=ncol, ax_length_in=2.5, lim=lim)

    colors = get_colors(N)
    colors = np.repeat(colors[:, None], B, axis=1)
    assert colors.shape == (N, B, 4)

    for i, c in enumerate(colors):
        plot_scatter(axs[i], xs[i], c=c, title=r"$t$=" + f"{ts[i]:.2f}")

    if target is not None:
        plot_scatter(axs[-1], target, c="r", title="Target")
