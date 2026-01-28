import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothLyapunov(nn.Module):
    def __init__(
        self,
        input_dim: int,
        reference_g,
        hidden_dim: int = 64,
        transition_center: float = 100.0,
        transition_width: float = 20.0,
        non_negative: bool = False,
    ):
        super().__init__()
        self.scale = transition_center
        self.reference_g = reference_g
        self.register_buffer(
            "transition_center", torch.tensor(float(transition_center))
        )
        self.register_buffer("transition_scale", torch.tensor(float(transition_width)))

        stack = [
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        ]
        if non_negative:
            stack.append(nn.Softplus)
        self.net = nn.Sequential(*stack)

    def _get_mixing_weight(self, ref_val):
        magnitude = torch.sqrt(torch.clamp(ref_val, min=1e-6))
        alpha = torch.sigmoid(
            (magnitude - self.transition_center) / self.transition_scale
        )
        return alpha

    def forward(self, x):
        ref_val = self.reference_g(x)
        nn_val = self.net(x / self.scale) * self.scale
        alpha = self._get_mixing_weight(ref_val)

        return alpha * ref_val + (1 - alpha) * nn_val
