from collections.abc import Callable

import torch


class ReactionNetwork:
    def __init__(
        self,
        stoichiometry: torch.Tensor,
        propensities: Callable[[torch.Tensor], torch.Tensor],
        device: str = "cpu",
    ):
        self.S = stoichiometry.to(device)
        self.propensities = propensities

    @property
    def num_reactions(self):
        return self.S.shape[0]

    @property
    def num_species(self):
        return self.S.shape[1]

    def to(device: str):
        self.S = self.S.to(device)


def get_drift(model, network, x):
    g_x = model(x)

    drift = g_x * 0.0

    rates = network.propensities(x)

    for j in range(network.num_reactions):
        v_j = network.S[j]
        x_next = torch.clamp(x + v_j, min=0)
        g_next = model(x_next)

        drift = drift + rates[:, j : j + 1] * (g_next - g_x)

    return drift


class BirthDeath(ReactionNetwork):
    def __init__(self, mu: float, gamma: float):
        S_matrix = torch.tensor([[1.0], [-1.0]])

        def bd_propensities(x):
            return torch.cat([torch.full((x.shape[0], 1), mu), gamma * x], dim=1)

        super().__init__(S_matrix, bd_propensities)


class Schloegl(ReactionNetwork):
    def __init__(self, c1: float, c2: float, c3: float, c4: float):
        S_matrix = torch.tensor([[1.0], [-1.0], [1.0], [-1.0]])

        def bd_propensities(x):
            return torch.cat(
                [
                    torch.full((x.shape[0], 1), c3),  # 0 -> X
                    c4 * x,  # X -> 0
                    c1 * x * (x - 1) / 2,  # 2 X -> 3 X
                    c2 * x * (x - 1) * (x - 2) / 6,
                ],
                dim=1,
            )

        super().__init__(S_matrix, bd_propensities)


class ParBD(ReactionNetwork):
    def __init__(self):
        S = torch.tensor(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ]
        )
        super().__init__(S, self._propensities)

        self.alpha = 1.0
        self.beta = 0.01

    def _propensities(self, x):
        X, Y = x[:, 0:1], x[:, 1:2]

        return torch.cat(
            [
                torch.full_like(X, self.alpha),
                self.beta * X,
                torch.full_like(Y, self.alpha),
                self.beta * Y,
            ],
            dim=1,
        )


class Toggle(ReactionNetwork):
    def __init__(self, alpha=10, beta=0.1, k=1.5):
        S = torch.tensor(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ]
        )
        super().__init__(S, self._propensities)

        self.alpha = alpha
        self.k = k
        self.beta = beta

    def _propensities(self, x):
        X, Y = x[:, 0:1], x[:, 1:2]

        return torch.cat(
            [
                self.alpha / (1 + self.k * Y),
                self.beta * X,
                self.alpha / (1 + self.k * X),
                self.beta * Y,
            ],
            dim=1,
        )


class LotkaVolterra(ReactionNetwork):
    def __init__(self):
        # Species Order: [Prey, Predator]
        S = torch.tensor(
            [
                [1.0, 0.0],  # Prey birth: X -> 2X
                [-1.0, 1.0],  # X + Y -> 2Y
                [0.0, -1.0],  # Predator death: Y -> 0
                [1.0, 0.0],  # Prey materialization: 0 -> X
                [-1.0, 1.0],  # Prey conversion: X -> Y
                [-1.0, 0.0],  #
            ]
        )
        super().__init__(S, self._propensities)

        self.alpha = 1.0
        self.beta = 0.001
        self.gamma = 10.0
        self.eps = 0.01
        self.delta = 1e-4

    def _propensities(self, x):
        prey, predator = x[:, 0:1], x[:, 1:2]

        return torch.cat(
            [
                self.alpha * prey,  # Prey birth
                self.beta * prey * predator,  # Predation
                self.gamma * predator,  # Predator death
                torch.full_like(prey, self.eps),  # Prey materialization
                self.eps * prey,  # Prey-predator conversion
                self.delta * prey * (prey - 1),  # 2X -> X
            ],
            dim=1,
        )


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

        # a4 is the non-linear degradation
        a4 = self.k3 * Mdm2 * (p53 / (p53 + self.k7))

        return torch.cat(
            [
                torch.full_like(p53, self.k1),  # k1
                self.k2 * p53,  # k2
                self.k4 * p53,  # k4
                a4,  # alpha4
                self.k5 * pMdm2,  # k5
                self.k6 * Mdm2,  # k6
            ],
            dim=1,
        )
