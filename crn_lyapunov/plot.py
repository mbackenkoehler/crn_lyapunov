from collections.abc import Callable, Iterable
from functools import partial

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import tqdm.auto as tqdm

from .crn import ReactionNetwork
from .utils import device, get_drift


def plot_loss_traj(h_loss, h_dmax):
    fig, ax = plt.subplots()
    ax.plot(h_loss, label="loss", lw=1)
    # ax.set_yscale("log")
    ax.set_ylabel("Loss")
    tax = ax.twinx()
    tax.plot(h_dmax, label="max", c="r", lw=1)
    tax.set_yscale("log")
    tax.set_ylabel(r"$\max\, D(x)$")
    plt.legend()
    return fig, ax, tax


def plot_drift_1d(
    n_max: int,
    model: Callable,
    ref: Callable,
    net: ReactionNetwork,
    adv_population: None | Iterable[float] = None,
    figsize: tuple[int] = (18, 5),
):
    x_range = torch.arange(n_max).float().view(-1, 1).detach()
    x_dev = x_range.to(device)

    with torch.no_grad():
        drift_vals = get_drift(model, net, x_dev).detach().cpu().numpy()
        d_max = np.max(drift_vals)
        v_vals = model(x_dev).detach().cpu().numpy()
        alpha_vals = model._get_mixing_weight(ref(x_dev)).detach().cpu().numpy()
        ref_vals = ref(x_dev).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    x_plot = x_range.cpu().numpy()

    axes[0].plot(x_plot, v_vals, label="Learned V(x)", color="blue")
    axes[0].plot(
        x_plot,
        ref_vals,
        "--",
        label="Reference",
        color="gray",
        alpha=0.5,
    )
    axes[0].set_title("Lyapunov Function")
    axes[0].set_xlabel("Population x")
    axes[0].set_yscale("log")

    eps = 5e-1
    axes[1].plot(x_plot, eps * drift_vals / drift_vals.max(), color="red")
    axes[1].axhline(0, color="k", linestyle="--")
    axes[1].axhline(eps - 1, color="g", linestyle="--")
    axes[1].set_title("Drift")
    axes[1].set_xlabel("Population x")
    axes[1].set_ylim(-1.5, eps + 0.1)

    axes[2].plot(x_plot, alpha_vals, color="green")
    axes[2].set_title("Mixing Weight $\\gamma$")
    axes[2].set_xlabel("Population x")

    if adv_population is not None:
        for ax in axes:
            sns.rugplot(adv_population.cpu(), ax=ax, legend=False)

    return axes


@torch.no_grad()
def plot_level_set_comparison(
    model, network, reference_fn, x_max=600, min_level=-4, ax=None
):
    x_range = (
        torch.arange(start=0, end=x_max, step=max(1, x_max // 1000)).float().view(-1, 1)
    )
    x_dev = x_range.to(device)

    drift_aug = get_drift(model, network, x_dev).detach().cpu().numpy().flatten()
    d_max_aug = np.max(drift_aug)
    ratio_aug = drift_aug / d_max_aug

    class RefWrapper(torch.nn.Module):
        def __init__(self, f):
            super().__init__()
            self.f = f

        def forward(self, x):
            return self.f(x)

    drift_ref = (
        get_drift(RefWrapper(reference_fn), network, x_dev)
        .detach()
        .cpu()
        .numpy()
        .flatten()
    )
    d_max_ref = np.max(drift_ref)
    ratio_ref = drift_ref / d_max_ref

    eps_vals = np.logspace(min_level, 0, 300)
    X, E = np.meshgrid(x_range.cpu().numpy().flatten(), eps_vals)

    thresholds = 1.0 - (1.0 / E)

    C_aug = ratio_aug[None, :] > thresholds
    C_ref = ratio_ref[None, :] > thresholds

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    im1 = ax.pcolormesh(X, E, C_ref, cmap="Blues", shading="auto", alpha=0.8)
    im2 = ax.pcolormesh(X, E, C_aug, cmap="Reds", shading="auto", alpha=0.8)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\epsilon$ (Log Scale)")
    ax.set_xlabel("State Space (x)")

    ax.grid(True, which="both", ls="-", alpha=0.1)

    plt.tight_layout()
    return ax


def plot_drift_2d(
    model, net, x_max, y_max, adversary=None, num_points=300, log_drift=False
):
    x_range = torch.linspace(0, x_max, num_points)
    y_range = torch.linspace(0, y_max, num_points)

    x_mesh, y_mesh = torch.meshgrid(x_range, y_range, indexing="xy")

    x_grid = torch.stack([x_mesh.flatten(), y_mesh.flatten()], dim=1).float().to(device)

    with torch.no_grad():
        grid_drift = get_drift(model, net, x_grid).detach().cpu().numpy()

    drift_heatmap_data = grid_drift.reshape(num_points, num_points) / grid_drift.max()

    plt.figure(figsize=(10, 8))
    plt.imshow(
        np.sign(drift_heatmap_data) * np.log(np.abs(drift_heatmap_data))
        if log_drift
        else drift_heatmap_data,
        origin="lower",
        extent=[
            float(x_range.min()),
            float(x_range.max()),
            float(y_range.min()),
            float(y_range.max()),
        ],
        cmap="viridis",
        aspect="auto",
    )

    plt.colorbar(label="Drift Value")

    if adversary is not None:
        points_lv = adversary.detach().cpu().numpy()
        plt.scatter(
            points_lv[:, 0],
            points_lv[:, 1],
            c="red",
            s=5,
            alpha=0.5,
            label="Adversarial Points",
        )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()


def hist_2d(fun, x_range, y_range, min_eps, dmax):
    assert len(x_range) == len(y_range)
    num_points = len(x_range)

    x_mesh, y_mesh = torch.meshgrid(x_range, y_range, indexing="xy")
    x_grid = torch.stack([x_mesh.flatten(), y_mesh.flatten()], dim=1).float().to(device)

    batch_size = 2**15
    outputs = []

    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(x_grid), batch_size)):
            batch = x_grid[i : i + batch_size]

            # drift = get_drift(model, net, batch)
            drift = fun(batch)

            if drift.ndim == 1:
                drift = drift.unsqueeze(1)

            outputs.append(drift.cpu())

    drift_map = torch.cat(outputs, dim=0).numpy().reshape(num_points, num_points)
    drift_map = drift_map / dmax

    cum_prob = np.zeros_like(drift_map)

    eps_values = np.logspace(min_eps, 0, 1000)
    for eps in eps_values:
        mask = drift_map * eps > eps - 1
        cum_prob[mask] = eps

    return cum_prob


def plot_hist_2d_comp(
    model,
    net,
    x_max,
    y_max,
    dmax_aug,
    dmax_ref,
    show_lya=False,
    min_eps=-4,
    num_points=300,
    log_prob=False,
    size=5,
):
    import matplotlib.gridspec as gridspec

    x_range = torch.linspace(0, x_max, min(x_max, num_points))
    y_range = torch.linspace(0, y_max, min(y_max, num_points))

    rows = 2 if show_lya else 1

    fig = plt.figure(figsize=(2 * size, size * rows))
    gs = gridspec.GridSpec(
        rows,
        3,
        width_ratios=[1, 1, 0.05],
        height_ratios=[1] * rows,
        wspace=0.15,
        hspace=0.2,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    cax12 = fig.add_subplot(gs[0, 2])

    if show_lya:
        ax3 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)
        ax4 = fig.add_subplot(gs[1, 1], sharex=ax1, sharey=ax1)
        cax34 = fig.add_subplot(gs[1, 2])
    else:
        ax3 = ax4 = cax34 = None

    drift_plots = [
        (ax1, dmax_aug, partial(get_drift, model, net)),
        (ax2, dmax_ref, partial(get_drift, model.reference_g, net)),
    ]

    shared_im = None
    for ax, dmax, fun in drift_plots:
        cum_prob = hist_2d(fun, x_range, y_range, min_eps, dmax)
        data = np.log(cum_prob) if log_prob else cum_prob

        im = ax.imshow(
            data,
            origin="lower",
            extent=[
                float(x_range.min()),
                float(x_range.max()),
                float(y_range.min()),
                float(y_range.max()),
            ],
            cmap="viridis",
            aspect="equal",
        )
        shared_im = im

    fig.colorbar(
        shared_im, cax=cax12, label="log Probability" if log_prob else "Probability"
    )

    ax1.set_title("Augmented Histogram")
    ax2.set_title("Reference Histogram")
    ax1.set_ylabel("Y")

    if show_lya:
        x_mesh, y_mesh = torch.meshgrid(x_range, y_range, indexing="xy")
        x_grid = torch.stack([x_mesh.flatten(), y_mesh.flatten()], dim=1).float()

        lya_plots = [
            (ax3, model),
            (ax4, model.reference_g),
        ]

        shared_im_lya = None

        for ax, fun in lya_plots:
            with torch.no_grad():
                vals = fun(x_grid)
                if vals.ndim == 1:
                    vals = vals.unsqueeze(1)

            data = vals.reshape(len(x_range), len(y_range)).cpu().numpy()

            im = ax.imshow(
                np.log(data),
                origin="lower",
                extent=[
                    float(x_range.min()),
                    float(x_range.max()),
                    float(y_range.min()),
                    float(y_range.max()),
                ],
                cmap="viridis",
                aspect="equal",
            )
            shared_im_lya = im

        fig.colorbar(shared_im_lya, cax=cax34, label="log Lyapunov")

        ax3.set_title("Augmented Lyapunov")
        ax4.set_title("Reference Lyapunov")
        ax3.set_ylabel("Y")
        ax3.set_xlabel("X")
        ax4.set_xlabel("X")
    else:
        ax1.set_xlabel("X")
        ax2.set_xlabel("X")

    ax2.tick_params(left=False, labelleft=False)

    if show_lya:
        ax4.tick_params(left=False, labelleft=False)

    return fig, (ax1, ax2)


def plot_hist_2d(
    model,
    net,
    x_max,
    y_max,
    dmax,
    min_eps=-4,
    num_points=300,
    log_prob=False,
):
    x_range = torch.linspace(0, x_max, min(x_max, num_points))
    y_range = torch.linspace(0, y_max, min(y_max, num_points))
    fun = partial(get_drift, model, net)
    cum_prob = hist_2d(fun, x_range, y_range, min_eps, dmax)

    ax = plt.imshow(
        np.log(cum_prob) if log_prob else cum_prob,
        origin="lower",
        extent=[
            float(x_range.min()),
            float(x_range.max()),
            float(y_range.min()),
            float(y_range.max()),
        ],
        cmap="viridis",
        aspect="auto",
    )

    plt.colorbar(label="log Probability" if log_prob else "Probability")

    plt.xlabel("X")
    plt.ylabel("Y")
    return ax


def plot_performances(sizes: pd.DataFrame, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    (l1,) = ax.plot(
        sizes["epsilon"], sizes["size_ref"], "ob:", label="reference size", lw=1
    )
    (l2,) = ax.plot(
        sizes["epsilon"], sizes["size_aug"], "or:", label="augmented size", lw=1
    )
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel("Set size")
    ax.set_xlabel("Threshold $\\epsilon$")

    tax = ax.twinx()
    (l3,) = tax.plot(
        sizes["epsilon"],
        sizes["size_ref"] / sizes["size_aug"],
        "^k:",
        label=r"improvement ratio",
    )

    tax.set_xscale("log")
    tax.set_ylabel("improvement ratio")

    ax.legend(handles=[l1, l2, l3], loc="best")
    return ax
