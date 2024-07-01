from math import pi

import torch
from torch import nn, Tensor
from torch.nn import Module, functional as F

from rlsandbox.lunar_lander_mbrl.nn import linear_layers
from rlsandbox.lunar_lander_mbrl.utils import assert_shape


class LunarLanderWorldModel(Module):
    def __init__(
            self,
            state_size: int = 8,
            action_size: int = 4,
    ) -> None:
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size

        activation_builders = [
            # nn.LeakyReLU,
            nn.ReLU,
            # lambda: nn.Dropout(p=0.5),
        ]

        self.state_layers = nn.Sequential(
            *linear_layers(
                state_size + action_size,
                256,
                # 256,
                state_size,
                activation_builders=activation_builders,
            )
        )

        self.done_layers = nn.Sequential(
            *linear_layers(
                state_size,
                256,
                # 256,
                1,
                activation_builders=activation_builders,
            )
        )

        self.reward_layers = nn.Sequential(
            *linear_layers(
                state_size + action_size + state_size + 1,
                256,
                # 256,
                1,
                activation_builders=activation_builders,
            )
        )

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            state:
                State batch of shape (N, S), dtype float.
                S = (x, y, dx, dy, theta, dtheta, leg_l, leg_r)
            action:
                Action batch of shape (N,), dtype int.

        Returns:
            A three-tuple of:
            0. Predicted next state tensor of shape (N, S);
            1. The predicted step reward tensor of shape (N, 1); and
            2. The predicted probability of the game ending
               as a tensor of shape (N, 1).
        """

        batch_size, _ = state.shape
        assert_shape(action, (batch_size,))

        mod_state = state.clone()
        # mod_state[:, 2:4] = F.tanh(state[:, 2:4])
        # mod_state[:, 4] /= pi
        # mod_state[:, 5] = F.tanh(state[:, 5])
        mod_state[:, 4:6] /= pi

        action = F.one_hot(action, num_classes=self.action_size)

        state_and_action = torch.cat([mod_state, action], dim=1)

        next_state_with_logits = self.state_layers(state_and_action)
        next_state_with_logits[:, :6] += state[:, :6]

        next_state = next_state_with_logits.clone()
        next_state[:, 6:8] = F.sigmoid(next_state[:, 6:8])

        done_logit = self.done_layers(next_state)

        reward = self.reward_layers(
            torch.cat([state_and_action, next_state, F.sigmoid(done_logit)], dim=1)
        )

        assert_shape(next_state, (batch_size, self.state_size))
        assert_shape(done_logit, (batch_size, 1))
        assert_shape(reward, (batch_size, 1))

        return next_state_with_logits, reward, done_logit

    def print_params_stats(self) -> None:
        for name, param in self.named_parameters():
            self.print_param_stats(name, param)

    def print_param_stats(self, name, param) -> None:
        param_mean = param.mean().item()
        param_std = param.std().item()
        if param.grad is not None:
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            print(f'{name}\t: {param_mean:.3f} ± {param_std:.3f}; {grad_mean:.3f} ± {grad_std:.3f}')
        else:
            print(f'{name}\t: {param_mean:.3f} ± {param_std:.3f}')


class SkipConnection(nn.Module):
    def __init__(self, *inners: nn.Module) -> None:
        super().__init__()

        self.inner = nn.Sequential(*inners)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.inner(x)


class LunarLanderValueModel(Module):
    def __init__(
            self,
            state_size: int = 8,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            *linear_layers(
                state_size,
                256,
                256,
                1,
                activation_builders=[
                    # nn.LeakyReLU,
                    nn.ReLU,
                    # lambda: nn.Dropout(p=0.5),
                ],
            )
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.layers(state).squeeze(1)
