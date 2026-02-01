import abc

import torch
from torch import nn
import torch.nn.functional as F

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class DriftLoss(nn.Module, abc.ABC):
    def __init__(self, gamma: float):
        super().__init__()
        self.register_buffer("_d_max", torch.tensor(-1.0, device=device))
        self.gamma = gamma

    @staticmethod
    def _max_drift_util(drift: torch.Tensor) -> torch.Tensor:
        d_max_smooth = torch.logsumexp(drift, dim=0)
        return F.softplus(d_max_smooth) + 1e-4

    @property
    def max_drift(self) -> float:
        return self._d_max.detach().item()

    def _update_d_max(self, d_max_stable: torch.Tensor):
        if self._d_max.item() < 0:
            self._d_max.data = d_max_stable.detach()
        else:
            self._d_max.data = (
                self.gamma * self._d_max.data + (1 - self.gamma) * d_max_stable.detach()
            )

    @abc.abstractmethod
    def forward(self, drift_combined: torch.Tensor) -> torch.Tensor:
        pass


class MaxDrift(DriftLoss):
    def __init__(self):
        super().__init__(gamma=0.0)

    def forward(self, drift_combined: torch.Tensor) -> torch.Tensor:
        return self._max_drift_util(drift_combined)


class TightLoss(DriftLoss):
    def __init__(
        self,
        gamma: float = 0.0,
        n_adv: int = None,
        adv_weight: float = 1.0,
        k: float = 2.0,
    ):
        super().__init__(gamma)
        self.n_adv = n_adv
        self.adv_weight = adv_weight
        self._mask = None
        self.k = k

    def forward(self, drift_combined: torch.Tensor) -> torch.Tensor:
        d_max_stable = self._max_drift_util(drift_combined)
        with torch.no_grad():
            self._update_d_max(d_max_stable)

        d_norm = drift_combined / self._d_max
        if self.n_adv is not None:
            if self._mask is None:
                self._mask = torch.ones(len(drift_combined))
                self._mask[: self.n_adv] = self.adv_weight
            d_norm = d_norm * self._mask
        loss = torch.mean(torch.exp(self.k * d_norm))
        return loss


class CombinedLoss(DriftLoss):
    def __init__(
        self,
        gamma: float = 0.0,
        dmax_weight: float = 1.0,
        n_ignore: int = None,
        k: float = 2.0,
    ):
        super().__init__(gamma)
        self.dmax_weight = dmax_weight
        self.n_ignore = n_ignore
        self.k = k

    def forward(self, drift_combined: torch.Tensor) -> torch.Tensor:
        d_max_stable = self._max_drift_util(drift_combined)
        with torch.no_grad():
            self._update_d_max(d_max_stable)

        d_norm = drift_combined / self._d_max
        if self.n_ignore is not None:
            d_norm = d_norm[: self.n_ignore]
        loss = torch.mean(torch.exp(self.k * d_norm)) + self.dmax_weight * d_max_stable
        return loss


class ProbabilityLoss(DriftLoss):
    def __init__(
        self,
        gamma: float,
        probability: float,
        steepness: float = 3,
        d_max_weight: float = 0.0,
    ):
        super().__init__(gamma, d_max_weight)
        self.register_buffer(
            "probability", torch.tensor(float(probability), device=device)
        )
        self.register_buffer("steepness", torch.tensor(float(steepness), device=device))

    def forward(self, drift_combined: torch.Tensor) -> torch.Tensor:
        d_max_stable = self._max_drift_util(drift_combined)
        with torch.no_grad():
            self._update_d_max(d_max_stable)

        d_norm = drift_combined / self._d_max
        loss = torch.sum(
            F.sigmoid(
                (d_norm * self.probability + (self.probability - 1) + 1) * steepness
            )
        )
        return loss + self.d_max_weight * d_max_stable
