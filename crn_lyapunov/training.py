import torch
import torch.optim as optim
import torch.nn.functional as F
import tqdm.auto as tqdm

from .smooth_lyapunov import SmoothLyapunov
from .crn import get_drift

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class Adversary:
    def __init__(self, network, n_species, n_max, population_size=128, lr_ascent=5.0):
        self.network = network
        self.n_species = n_species
        self.pop_size = population_size
        self.n_max = n_max
        self.lr_ascent = lr_ascent

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

            mutation = torch.randn_like(parents_expanded) * (self.n_max * 0.1)
            offspring = torch.clamp(parents_expanded + mutation, 0, self.n_max)

            self.population = torch.round(torch.cat([parents, offspring], dim=0))

        with torch.no_grad():
            self.last_drift = get_drift(model, self.network, self.population).view(-1)
        return self.population.detach()

    def update_population(self, x_candidates, drift_candidates):
        with torch.no_grad():
            combined_x = torch.cat([self.population, x_candidates.detach()], dim=0)
            combined_d = torch.cat(
                [self.last_drift, drift_candidates.view(-1).detach()], dim=0
            )

            _, top_idx = torch.topk(combined_d, self.pop_size)
            self.population = combined_x[top_idx]
            self.last_drift = combined_d[top_idx]


def max_drift(drift):
    d_max_smooth = torch.logsumexp(drift, dim=0)
    return F.softplus(d_max_smooth) + 1e-4


def train_tight_sets(
    network,
    ref_g,
    n_species,
    probability=None,
    n_adv_samples=64,
    n_rand_samples=64,
    max_n=100,
    n_epochs=1000,
    steps_evolve=5,
    hidden_dim=32,
    lr=1e-3,
    gamma=0.9,
    reference_alignment=None,
    d_max_weight=0,
):
    adv = Adversary(network, n_species, max_n, population_size=n_adv_samples)

    lyap_model = SmoothLyapunov(
        n_species,
        ref_g,
        hidden_dim=hidden_dim,
        transition_center=max_n * 0.8,
        transition_width=10.0,
    ).to(device)

    optimizer = optim.AdamW(lyap_model.parameters(), lr=lr)
    history_loss, history_dmax = [], []

    d_max = None

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

            d_max_stable = max_drift(drift_combined)
            if d_max is None:
                d_max = d_max_stable

            d_max = max(d_max_stable, d_max_stable * (1 - gamma) + d_max * gamma)
            d_norm = drift_combined / d_max_stable

            if probability is None:
                loss = torch.mean(torch.exp(2.0 * d_norm))
            else:
                loss = torch.sum(
                    F.sigmoid((d_norm * probability + (probability - 1) + 1) * 3)
                )

            loss = loss + d_max_weight * d_max_stable

            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                pbar.set_description(
                    f"Loss: {loss.item():.4e} | d_max: {d_max_stable.item():.2e}"
                )
                history_loss.append(loss.item())
                history_dmax.append(d_max_stable.item())

        except KeyboardInterrupt:
            print("Training interrupted by user.")
            break

    return lyap_model, adv, history_loss, history_dmax
