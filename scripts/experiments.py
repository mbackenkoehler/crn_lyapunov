#!/usr/bin/env python

from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from crn_lyapunov.training import train_tight_sets
from crn_lyapunov.loss import TightLoss
from crn_lyapunov.plot import (
    plot_drift_1d,
    plot_loss_traj,
    plot_level_set_comparison,
    plot_drift_2d,
    plot_hist_2d,
    plot_performances,
)
from crn_lyapunov.utils import performance_table, get_drift, device
from crn_lyapunov.crn import (
    BirthDeath,
    Schloegl,
    ParBD,
    Competition,
    Toggle,
    ReactionNetwork,
)

sns.set_context("paper")
sns.set_style("white")

OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)


def savefig(model_dir: Path, name: str, dpi=300):
    ensure_dir(model_dir)
    plt.tight_layout()
    plt.savefig(model_dir / name, dpi=dpi)
    plt.close()


# -----------------------------------------------------------------------------
# birth–death
# -----------------------------------------------------------------------------
def run_birth_death():
    model_dir = OUTPUT / "birth_death"
    ensure_dir(model_dir)

    net = BirthDeath(200, 1)

    def quadratic_ref(x):
        return x**2

    model, adv, history, history_dmax = train_tight_sets(
        net,
        quadratic_ref,
        TightLoss(k=1),
        n_adv_samples=16,
        n_rand_samples=16,
        max_n=700,
        lr=1e-3,
        hidden_dim=128,
        steps_evolve=1,
        n_epochs=500,
        non_negative=True,
        output_path=model_dir,
    )

    plot_loss_traj(history, history_dmax)
    savefig(model_dir, "optimization.pdf")

    sizes = performance_table(
        model,
        net,
        quadratic_ref,
        [10_000_000],
        max_drift_ref=20300,
        min_eps=-9,
        output_dir=model_dir,
    )

    fig, ax = plt.subplots(figsize=(4, 3))
    plot_performances(sizes, ax=ax)
    ax.set_xlabel("Threshold $\\epsilon$")
    savefig(model_dir, "setsizes.pdf")

    axes = plot_drift_1d(
        1000, model, quadratic_ref, net, adv_population=adv.population, figsize=(8, 2)
    )
    axes[0].set_yscale("log")
    axes[1].set_ylim(-10, 0.7)
    axes[2].set_ylabel("")
    savefig(model_dir, "overview.pdf")

    fig, ax = plt.subplots(figsize=(5, 3))
    plot_level_set_comparison(
        model, net, quadratic_ref, min_level=-8, x_max=2000, ax=ax
    )
    savefig(model_dir, "levelsets.png", dpi=350)


# -----------------------------------------------------------------------------
# schloegl
# -----------------------------------------------------------------------------
def solve_schlogl_params(x_low, x_high, c2=0.001):
    x1 = x_low
    x3 = x_high
    x2 = (x1 + x3) / 2

    S = x1 + x2 + x3
    P = (x1 * x2) + (x2 * x3) + (x3 * x1)
    T = x1 * x2 * x3

    c1 = (c2 / 3) * S - c2
    c3 = (c2 / 6) * T
    c4 = (c2 / 6) * P - (c1 / 2) - (c2 / 3)

    return {"c1": c1, "c2": c2, "c3": c3, "c4": c4}


def run_schloegl():
    model_dir = OUTPUT / "schloegl"
    ensure_dir(model_dir)

    params = solve_schlogl_params(10, 100, c2=0.006)
    net = Schloegl(**params)

    def quadratic_ref(x):
        return x**2

    model, adv, history, history_dmax = train_tight_sets(
        net,
        quadratic_ref,
        TightLoss(),
        steps_evolve=5,
        n_adv_samples=8,
        n_rand_samples=8,
        max_n=500,
        lr=1e-3,
        hidden_dim=128,
        n_epochs=1000,
        non_negative=True,
        output_path=model_dir,
    )

    plot_loss_traj(history, history_dmax)
    savefig(model_dir, "optimization.pdf")

    sizes = performance_table(
        model, net, quadratic_ref, [1_000_000], min_eps=-10, output_dir=model_dir
    )

    plot_performances(sizes, ax=ax)
    savefig(model_dir, "setsizes.pdf")

    plot_drift_1d(
        600, model, quadratic_ref, net, adv_population=adv.population, figsize=(8, 2)
    )
    savefig(model_dir, "overview.pdf", dpi=350)

    fig, ax = plt.subplots(figsize=(8, 3))
    plot_level_set_comparison(model, net, quadratic_ref, min_level=-8, x_max=500, ax=ax)
    savefig(model_dir, "levelsets.png", dpi=350)


# -----------------------------------------------------------------------------
# parallel birth–death
# -----------------------------------------------------------------------------
def run_parbd(k=1):
    model_dir = OUTPUT / "parbd" / str(k)
    ensure_dir(model_dir)

    net_parbd = ParBD()

    def ref_g(x):
        X, Y = x[:, 0:1], x[:, 1:2]
        return (X - 0) ** 2 + (Y - 0) ** 2

    model_parbd, adv_parbd, h_loss, h_dmax = train_tight_sets(
        net_parbd,
        ref_g,
        TightLoss(k=0.1),
        steps_evolve=5,
        hidden_dim=128,
        n_adv_samples=2**8,
        n_rand_samples=2**8,
        max_n=300,
        n_epochs=10_000,
        lr=1e-3,
        output_path=model_dir,
    )

    plot_loss_traj(h_loss, h_dmax)
    savefig(model_dir, "loss.pdf")

    plot_drift_2d(model_parbd, net_parbd, 300, 300, log_drift=True, adversary=adv_parbd)
    savefig(model_dir, "drift2d.pdf")
    sizes = performance_table(
        model_parbd,
        net_parbd,
        ref_g,
        [10_000, 10_000],
        max_drift_ref=103.0,
        max_drift_aug=0,
        chunk_size=1_000_000,
        min_eps=-3,
        output_dir=model_dir,
    )
    fig, ax = plt.subplots(figsize=(4, 3))
    plot_performances(sizes, ax=ax)
    ax.set_xlabel("Threshold $\\epsilon$")
    savefig(model_dir, "setsizes.pdf")


# -----------------------------------------------------------------------------
# competition
# -----------------------------------------------------------------------------
def run_competition():
    model_dir = OUTPUT / "competition"
    ensure_dir(model_dir)

    net_comp = Competition(k3=0.00004)

    def ref_g(x):
        X, Y = x[:, 0:1], x[:, 1:2]
        return (X - 0) ** 2 + (Y - 0) ** 2

    model_comp, adv_comp, h_loss, h_dmax = train_tight_sets(
        net_comp,
        ref_g,
        TightLoss(k=2),
        steps_evolve=2,
        hidden_dim=512,
        n_adv_samples=2**10,
        n_rand_samples=2**13,
        max_n=1500,
        n_epochs=5000,
        lr=1e-3,
        output_path=model_dir,
    )

    plot_loss_traj(h_loss, h_dmax)
    savefig(model_dir, "loss.pdf")

    plot_hist_2d(
        model_comp,
        net_comp,
        1500,
        1500,
        4e2,
        min_eps=-8,
        num_points=100,
        log_prob=False,
    )
    savefig(model_dir, "hist2d.pdf")

    plot_drift_2d(model_comp, net_comp, 1500, 1500, log_drift=True, adversary=adv_comp)
    savefig(model_dir, "drift2d.pdf")


# -----------------------------------------------------------------------------
# toggle
# -----------------------------------------------------------------------------
def run_toggle():
    model_dir = OUTPUT / "toggle"
    ensure_dir(model_dir)

    net_toggle = Toggle(alpha=50, beta=0.1, k=10)

    def ref_g(x):
        X, Y = x[:, 0:1], x[:, 1:2]
        return (X - 0) ** 2 + (Y - 0) ** 2

    model_toggle, adv_toggle, h_loss, h_dmax = train_tight_sets(
        net_toggle,
        ref_g,
        TightLoss(k=1),
        steps_evolve=5,
        hidden_dim=2048,
        n_adv_samples=2**10,
        n_rand_samples=2**10,
        max_n=1000,
        n_epochs=1000,
        lr=1e-3,
        output_path=model_dir,
    )

    plot_loss_traj(h_loss, h_dmax)
    savefig(model_dir, "loss.pdf")

    sizes = performance_table(
        model_toggle,
        net_toggle,
        ref_g,
        [1_000, 1_000],
        min_eps=-3,
        chunk_size=10_000,
        output_dir=model_dir,
    )
    plot_performances(sizes)
    savefig(model_dir, "performance.pdf")

    plot_drift_2d(
        model_toggle, net_toggle, 750, 750, log_drift=True, adversary=adv_toggle
    )
    savefig(model_dir, "drift2d.pdf")


def run_p53():
    class P53Oscillator(ReactionNetwork):
        def __init__(self):
            # Species Order: [p53, pMdm2, Mdm2]
            # 0 -> p53           [1, 0, 0]
            # p53 -> 0           [-1, 0, 0]
            # p53 -> p53+pMdm2   [0, 1, 0]
            # p53 -> 0 (alpha4)  [-1, 0, 0]
            # pMdm2 -> Mdm2      [0, -1, 1]
            # Mdm2 -> 0          [0, 0, -1]
            S = torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 1.0],
                    [0.0, 0.0, -1.0],
                ]
            )
            super().__init__(S, self._propensities)

            # Parameters
            self.k1, self.k2, self.k3 = 90.0, 0.002, 1.7
            self.k4, self.k5, self.k6 = 1.1, 0.93, 0.96
            self.k7 = 0.01

        def _propensities(self, x):
            p53, pMdm2, Mdm2 = x[:, 0:1], x[:, 1:2], x[:, 2:3]

            return torch.cat(
                [
                    torch.full_like(p53, self.k1),  # k1
                    self.k2 * p53,  # k2
                    self.k4 * p53,  # k4
                    self.k3 * Mdm2 * (p53 / (p53 + self.k7)),
                    self.k5 * pMdm2,  # k5
                    self.k6 * Mdm2,  # k6
                ],
                dim=1,
            )

    def p53_reference_g(x):
        # g(x) = 120*p53 + 0.2*pMdm2 + 0.1*Mdm2
        weights = torch.tensor([120.0, 0.2, 0.1], device=x.device)
        return torch.sum(weights * x, dim=1, keepdim=True)

    model_dir = OUTPUT / "p53"
    ensure_dir(model_dir)

    net_p53 = P53Oscillator()

    model_p53, adv_p53, h_loss, h_dmax = train_tight_sets(
        net_p53,
        p53_reference_g,
        TightLoss(k=1),
        steps_evolve=10,
        hidden_dim=1024,
        n_adv_samples=2**14,
        n_rand_samples=2**14,
        max_n=1000,
        n_epochs=10000,
        lr=1e-3,
        output_path=model_dir,
    )
    plot_loss_traj(h_loss, h_dmax)
    savefig(model_dir, "loss.pdf")

    sizes = performance_table(
        model_p53,
        net_p53,
        p53_reference_g,
        [100_000, 100_000, 100_000],
        min_eps=-2,
        chunk_size=1_000_000,
        output_dir=model_dir,
    )
    plot_performances(sizes)
    savefig(model_dir, "performance.pdf")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    with torch.no_grad():
        drift = (
            get_drift(model_p53, net_p53, adv_p53.population.to(device)).cpu().numpy()
        )

    points = adv_p53.population.detach().numpy()

    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=drift.flatten() > 0,
        cmap="viridis",
        s=5,
    )
    dmax = drift.max()

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Drift")
    savefig(model_dir, "adversaries.pdf")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    n = 100000
    points = eval_xs = torch.rand(n, 3) * torch.tensor([2000, 2000, 2000])
    with torch.no_grad():
        drift = get_drift(model_p53, net_p53, points.to(device)).cpu().numpy()

    eps = 1e-1
    in_set = drift.flatten() / dmax * eps > eps - 1

    drift = drift[in_set]
    points = points[in_set]

    sc = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2], c=drift, cmap="viridis", s=1
    )

    savefig(model_dir, "bounded_0.1p.pdf")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(10 * "#" + " Birth-Death")
    run_birth_death()
    print(10 * "#" + " Schlögl")
    run_schloegl()
    print(10 * "#" + " Parallel BD")
    for k in [0.1, 1, 10, 100]:
        run_parbd(k)
    print(10 * "#" + " Competition")
    run_competition()
    print(10 * "#" + " Toggle")
    run_toggle()
    run_p53()
