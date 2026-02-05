import json
from collections.abc import Callable
from pathlib import Path

import torch
import torch.optim as optim
import numpy as np
import tqdm.auto as tqdm

from .smooth_lyapunov import SmoothLyapunov
from .crn import ReactionNetwork
from .utils import device, get_drift


class Adversary:
    def __init__(
        self,
        network,
        n_species,
        n_max,
        population_size=128,
        mutation_var=0.1,
    ):
        self.network = network
        self.n_species = n_species
        self.pop_size = population_size
        self.n_max = n_max
        self.mutation_var = mutation_var

        self.population = self._get_unique_random_points(self.pop_size)
        self.last_drift = torch.full((self.pop_size,), -float("inf"), device=device)

    def _get_unique_random_points(self, n_samples) -> torch.Tensor:
        return torch.randint(
            0, int(self.n_max) + 1, (n_samples, self.n_species), device=device
        ).float()

    def evolve(self, model, steps=5):
        for _ in range(steps):
            with torch.no_grad():
                self.last_drift = get_drift(model, self.network, self.population).view(
                    -1
                )

            k = max(self.pop_size // 4, 1)
            _, top_idx = torch.topk(self.last_drift, k)
            parents = self.population[top_idx]

            offspring_count = self.pop_size - k
            parents_expanded = parents.repeat((offspring_count // k) + 1, 1)[
                :offspring_count
            ]

            offspring = self.mutate(parents_expanded)

            self.population = torch.round(torch.cat([parents, offspring], dim=0))

        with torch.no_grad():
            self.last_drift = get_drift(model, self.network, self.population).view(-1)
        return self.population.detach()

    def mutate(self, parents):
        mutation = torch.randn_like(parents) * self.n_max * self.mutation_var
        return torch.clamp(parents + mutation, 0, self.n_max)

    def update_population(self, xs, ds):
        with torch.no_grad():
            combined_x = torch.cat([self.population, xs.detach()], dim=0)
            combined_d = torch.cat([self.last_drift, ds.view(-1).detach()], dim=0)

            _, top_idx = torch.topk(combined_d, self.pop_size)
            self.population = combined_x[top_idx]
            self.last_drift = combined_d[top_idx]


def sample_normal(max_n, n_rand_samples, n_species):
    return torch.randint(
        0, int(max_n) + 1, (n_rand_samples, n_species), device=device
    ).float()


def train_tight_sets(
    network: ReactionNetwork,
    ref_g: Callable,
    loss_fn: torch.nn.Module,
    n_adv_samples: int = 64,
    n_rand_samples: int = 64,
    max_n: int = 100,
    n_epochs: int = 1000,
    steps_evolve: int = 1,
    hidden_dim: int = 64,
    lr: float = 1e-4,
    seed: int = 0,
    non_negative: bool = False,
    sampler: Callable = None,
    output_path: Path = None,
):
    torch.manual_seed(seed)
    n_species = network.num_species
    adv = Adversary(network, n_species, max_n, population_size=n_adv_samples)
    if sampler is None:
        sampler = sample_normal

    lyap_model = SmoothLyapunov(
        n_species,
        ref_g,
        hidden_dim=hidden_dim,
        transition_center=max_n * 0.8,
        transition_width=max_n * 0.05,
        non_negative=non_negative,
    ).to(device)

    optimizer = optim.AdamW(lyap_model.parameters(), lr=lr)
    history_loss, history_dmax = [], []

    for epoch in (pbar := tqdm.tqdm(range(n_epochs))):
        try:
            adv.evolve(lyap_model, steps=steps_evolve)
            optimizer.zero_grad()

            x_adv = adv.population.detach()
            x_rand = sampler(max_n, n_rand_samples, n_species)
            x_combined = torch.cat([x_adv, x_rand], dim=0)
            drift_combined = get_drift(lyap_model, network, x_combined)
            with torch.no_grad():
                adv.update_population(x_rand, drift_combined[n_adv_samples:].detach())
            loss = loss_fn(drift_combined, x_combined)

            loss.backward()
            optimizer.step()

            dmax = loss_fn.max_drift
            pbar.set_description(f"Loss: {loss.item():.4e} | max D: {dmax:.4e}")
            history_loss.append(loss.item())
            history_dmax.append(dmax)

        except KeyboardInterrupt:
            print("Training interrupted by user.")
            break

    if output_path is not None:
        output_path.mkdir(exist_ok=True, parents=True)
        torch.save(lyap_model.state_dict(), output_path / "model.pt")
        with open(output_path / "history_loss.npy", "wb") as f:
            np.save(f, np.array(history_loss))
        with open(output_path / "history_dmax.npy", "wb") as f:
            np.save(f, np.array(history_loss))

        settings = dict(
            n_adv_samples=n_adv_samples,
            n_rand_samples=n_rand_samples,
            max_n=max_n,
            n_epochs=n_epochs,
            steps_evolve=steps_evolve,
            hidden_dim=hidden_dim,
            lr=lr,
            seed=seed,
            non_negative=non_negative,
            loss_fn=str(loss_fn),
        )
        with open(output_path / "settings.json", "w") as f:
            json.dump(settings, f, skipkeys=True)

    return lyap_model, adv, history_loss, history_dmax
