from collections.abc import Callable

import torch

from .utils import device


class ReactionNetwork:
    def __init__(
        self,
        stoichiometry: torch.Tensor,
        propensities: Callable[[torch.Tensor], torch.Tensor],
    ):
        self.S = stoichiometry.to(device)
        self.propensities = propensities

    @property
    def num_reactions(self):
        return self.S.shape[0]

    @property
    def num_species(self):
        return self.S.shape[1]

    def to(self, device: str):
        self.S = self.S.to(device)
        return self


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
    def __init__(self, alpha=1, beta=0.01):
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
        self.beta = beta

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


class GeneSwitchNetwork(ReactionNetwork):
    def __init__(self):
        S = torch.tensor(
            [
                [-1, 1, 0, 0, 0],  # G -> G1
                [1, -1, 0, 0, 0],  # G1 -> G
                [0, -1, 1, 0, 0],  # G1 -> G2
                [0, 1, -1, 0, 0],  # G2 -> G1
                [0, 0, 0, 1, 0],  # Production (from G1)
                [0, 0, 0, -1, 0],  # Degradation
                [0, 0, 0, 0, 1],  # Production (from G1)
                [0, 0, 0, 0, -1],  # Degradation
            ]
        ).float()
        super().__init__(S, self._propensities)

    # parameters
    #     p1 = 6
    #     p2 = 4
    #
    #     d1 = 0.05
    #     d2 = 0.05
    #
    #     b1 = 0.006
    #     b2 = 0.010
    #
    #     u1 = 0.1
    #     u2 = 0.1
    #

    def _propensities(self, x):
        G, G1, G2, P1, P2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]

        rates = torch.zeros((x.shape[0], self.num_reactions), device=x.device)
        rates[:, 0] = 0.1 * G  # Activation
        rates[:, 1] = 0.2 * G1  # Deactivation
        rates[:, 2] = 0.05 * G1  # Transition to G2
        rates[:, 3] = 0.1 * G2  # Transition back to G1
        rates[:, 4] = 10.0 * G1  # Translation (only when in G1 state)
        rates[:, 5] = 0.1 * P1  # Decay
        rates[:, 4] = 8.0 * G2  # Translation (only when in G1 state)
        rates[:, 5] = 0.1 * P2  # Decay
        return rates


class Competition(ReactionNetwork):
    """
    parameters
        k1 = 1000.0
        k2 = 1.0
        k3 = 0.00001

    species X Y

    reactions
        0 -> X @ k1;
        X -> 0 @ k2 * X;
        0 -> Y @ k1;
        Y -> 0 @ k2 * Y;
        2 X + Y -> 2 X @ k3 * X * (X - 1) * Y;
        X + 2 Y -> 2 Y @ k3 * X * Y * (Y - 1);

    init
        X = 0
        Y = 0
    """

    def __init__(self, k1=1000.0, k2=1.0, k3=0.00001):
        S = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, -1.0], [-1.0, 0.0]]
        )
        super().__init__(S, self._propensities)

        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

    def _propensities(self, x):
        X, Y = x[:, 0:1], x[:, 1:2]

        return torch.cat(
            [
                torch.full_like(X, self.k1),
                self.k2 * X,
                torch.full_like(Y, self.k1),
                self.k2 * Y,
                self.k3 * X * (X - 1) * Y,
                self.k3 * Y * (Y - 1) * X,
            ],
            dim=1,
        )
