from collections.abc import Callable

import torch
import torch.optim as optim
import tqdm.auto as tqdm

from .smooth_lyapunov import SmoothLyapunov
from .crn import get_drift, ReactionNetwork

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class Adversary:
    def __init__(
        self,
        network,
        n_species,
        n_max,
        population_size=128,
        lr_ascent=5.0,
        mutation_var=0.2,
    ):
        self.network = network
        self.n_species = n_species
        self.pop_size = population_size
        self.n_max = n_max
        self.lr_ascent = lr_ascent
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

            mutation = torch.randn_like(parents_expanded) * (
                self.n_max * self.mutation_var
            )
            offspring = torch.clamp(parents_expanded + mutation, 0, self.n_max)

            self.population = torch.round(torch.cat([parents, offspring], dim=0))

        with torch.no_grad():
            self.last_drift = get_drift(model, self.network, self.population).view(-1)
        return self.population.detach()

    def update_population(self, xs, ds):
        with torch.no_grad():
            combined_x = torch.cat([self.population, xs.detach()], dim=0)
            combined_d = torch.cat([self.last_drift, ds.view(-1).detach()], dim=0)

            _, top_idx = torch.topk(combined_d, self.pop_size)
            self.population = combined_x[top_idx]
            self.last_drift = combined_d[top_idx]


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
):
    n_species = network.num_species
    adv = Adversary(network, n_species, max_n, population_size=n_adv_samples)

    lyap_model = SmoothLyapunov(
        n_species,
        ref_g,
        hidden_dim=hidden_dim,
        transition_center=max_n * 0.8,
        transition_width=max_n * 0.05,
        non_negative=False,
    ).to(device)

    optimizer = optim.AdamW(lyap_model.parameters(), lr=lr)
    history_loss, history_dmax = [], []

    pbar = tqdm.tqdm(range(n_epochs))
    for epoch in pbar:
        try:
            adv.evolve(lyap_model, steps=steps_evolve)
            optimizer.zero_grad()

            x_adv = adv.population.detach()
            x_rand = torch.randint(
                0, int(max_n) + 1, (n_rand_samples, n_species), device=device
            ).float()
            x_combined = torch.cat([x_adv, x_rand], dim=0)
            drift_combined = get_drift(lyap_model, network, x_combined)
            with torch.no_grad():
                adv.update_population(x_rand, drift_combined[n_adv_samples:].detach())
            loss = loss_fn(drift_combined)

            loss.backward()
            optimizer.step()

            dmax = loss_fn.max_drift
            pbar.set_description(f"Loss: {loss.item():.4e} | max D: {dmax:.4e}")
            history_loss.append(loss.item())
            history_dmax.append(dmax)

        except KeyboardInterrupt:
            print("Training interrupted by user.")
            break

    return lyap_model, adv, history_loss, history_dmax
