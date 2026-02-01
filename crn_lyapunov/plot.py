from collections.abc import Callable, Iterable

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

from .crn import ReactionNetwork, get_drift


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def plot_loss_traj(h_loss, h_dmax):
    fig, ax = plt.subplots()
    ax.plot(h_loss, label="loss", lw=1)
    ax.set_yscale("log")
    ax.set_ylabel("Loss")
    tax = ax.twinx()
    tax.plot(h_dmax, label="max", c="r", lw=1)
    tax.set_yscale("log")
    tax.set_ylabel(r"$\max D(x)$")
    plt.legend()


def plot_drift_1d(
    n_max: int,
    model: Callable,
    ref: Callable,
    net: ReactionNetwork,
    adv_population: None | Iterable[float] = None,
):
    x_range = torch.arange(n_max).float().view(-1, 1).detach()
    with torch.no_grad():
        drift_vals = get_drift(model, net, x_range).detach().numpy()
        d_max = np.max(drift_vals)
        v_vals = model(x_range).numpy()
        alpha_vals = model._get_mixing_weight(ref(x_range)).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    x_range = x_range.detach()

    axes[0].plot(x_range, v_vals, label="Learned V(x)", color="blue")
    axes[0].plot(
        x_range,
        ref(x_range).numpy(),
        "--",
        label="Reference",
        color="gray",
        alpha=0.5,
    )
    axes[0].set_title("Lyapunov Function")
    axes[0].set_xlabel("Population x")
    axes[0].set_yscale("log")
    axes[0].legend()

    eps = 5e-1
    axes[1].plot(x_range, eps * drift_vals / drift_vals.max(), color="red")
    axes[1].axhline(0, color="k", linestyle="--")
    axes[1].axhline(eps - 1, color="g", linestyle="--")
    axes[1].set_title("Drift (Should be < 0)")
    axes[1].set_xlabel("Population x")
    axes[1].set_ylim(-1.5, eps + 0.1)

    axes[2].plot(x_range, alpha_vals, color="green")
    axes[2].set_title("Mixing Weight (alpha)")
    axes[2].set_xlabel("Population x")
    axes[2].set_ylabel("0 = Neural, 1 = Reference")

    if adv_population is not None:
        for ax in axes:
            sns.rugplot(adv_population, ax=ax)

    plt.tight_layout()


@torch.no_grad()
def plot_level_set_comparison(model, network, reference_fn, x_max=600, min_level=-4):
    x_range = (
        torch.arange(start=0, end=x_max, step=max(1, x_max // 1000)).float().view(-1, 1)
    )

    drift_aug = get_drift(model, network, x_range).detach().numpy().flatten()
    d_max_aug = np.max(drift_aug)
    ratio_aug = drift_aug / d_max_aug

    class RefWrapper(torch.nn.Module):
        def __init__(self, f):
            super().__init__()
            self.f = f

        def forward(self, x):
            return self.f(x)

    drift_ref = (
        get_drift(RefWrapper(reference_fn), network, x_range).detach().numpy().flatten()
    )
    d_max_ref = np.max(drift_ref)
    ratio_ref = drift_ref / d_max_ref

    eps_vals = np.logspace(min_level, 0, 300)
    X, E = np.meshgrid(x_range.numpy().flatten(), eps_vals)

    thresholds = 1.0 - (1.0 / E)

    C_aug = ratio_aug[None, :] > thresholds
    C_ref = ratio_ref[None, :] > thresholds

    # x_range

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))

    im1 = ax1.pcolormesh(X, E, C_ref, cmap="Blues", shading="auto", alpha=0.8)
    im2 = ax1.pcolormesh(X, E, C_aug, cmap="Reds", shading="auto", alpha=0.8)
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$\epsilon$ (Log Scale)")
    ax1.set_xlabel("State Space (x)")

    ax1.grid(True, which="both", ls="-", alpha=0.1)

    plt.tight_layout()
    plt.show()


def plot_drift_2d(
    model, net, x_max, y_max, adversary=None, num_points=300, log_drift=False
):
    x_range = torch.linspace(0, x_max, num_points)
    y_range = torch.linspace(0, y_max, num_points)

    x_mesh, y_mesh = torch.meshgrid(x_range, y_range, indexing="xy")

    x_grid = torch.stack([x_mesh.flatten(), y_mesh.flatten()], dim=1).to(device).float()

    with torch.no_grad():
        grid_drift = get_drift(model, net, x_grid).cpu().numpy()

    drift_heatmap_data = grid_drift.reshape(num_points, num_points) / grid_drift.max()

    plt.figure(figsize=(10, 8))
    plt.imshow(
        np.sign(drift_heatmap_data) * np.log(np.abs(drift_heatmap_data))
        if log_drift
        else drift_heatmap_data,
        origin="lower",
        extent=[
            x_range.min(),
            x_range.max(),
            y_range.min(),
            y_range.max(),
        ],
        cmap="viridis",
        aspect="auto",
    )

    plt.colorbar(label="Drift Value")

    if adversary is not None:
        points_lv = adversary.population.detach().cpu().numpy()
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
    plt.grid(True)
