from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn, Tensor
from torch.nn import Module

from rlsandbox.base.utils import assert_is_binary
from rlsandbox.lop.algos.cbp_linear import CBPLinear
from rlsandbox.walker.nn import ConcatTransformer
from rlsandbox.walker.utils import assert_shape

WalkerWorldModel = nn.Module


def get_world_model(
        cont_state_size: int = 22,
        disc_state_size: int = 2,
        action_size: int = 4,
) -> WalkerWorldModel:
    return MonoModel(cont_state_size, disc_state_size, action_size)


@dataclass
class ValueModelInput:
    cont_state: Tensor
    """
    Continuous state batch of shape (N, S_c), dtype float.
    
    S_c = (
        0: hull_angle,
        1: hull_angular_velocity,
        2: x_velocity,
        3: y_velocity,
        4: hip_1_joint_angle,
        5: hip_1_joint_angular_velocity,
        6: knee_1_joint_angle,
        7: knee_1_joint_angular_velocity,
        8: hip_2_joint_angle,
        9: hip_2_joint_angular_velocity,
        10: knee_2_joint_angle,
        11: knee_2_joint_angular_velocity,
        12-22: 10 lidar readings,
    )
    """

    disc_state: Tensor
    """
    Discrete state batch of shape (N, S_d), dtype float.
    
    S_d = (
        0: leg_1_ground_contact,
        1: leg_2_ground_contact,
    )
    """

    @classmethod
    def from_raw_state(cls, state: Tensor) -> 'ValueModelInput':
        """
        See raw state order here:
        https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/bipedal_walker.py#L566
        """

        input = cls(
            cont_state=torch.cat([
                state[:, 0:8],
                state[:, 9:13],
                state[:, 14:],
            ], dim=1),
            disc_state=torch.stack([
                state[:, 8],
                state[:, 13],
            ], dim=1),
        )

        assert_is_binary(input.disc_state)

        return input


@dataclass
class WorldModelInput(ValueModelInput):
    action: Tensor
    """Action batch of shape (N, 4), dtype int."""

    @classmethod
    def from_raw_state_and_action(cls, state: Tensor, action: Tensor) -> 'WorldModelInput':
        value_input = ValueModelInput.from_raw_state(state)
        return cls.from_value_model_input(value_input, action)

    @classmethod
    def from_value_model_input(cls, input: ValueModelInput, action: Tensor) -> 'WorldModelInput':
        return cls(
            cont_state=input.cont_state,
            disc_state=input.disc_state,
            action=action,
        )


@dataclass
class WorldModelOutput:
    next_cont_state: Tensor
    """Predicted next continuous state tensor of shape (N, S_c)."""

    next_disc_state: Tensor
    """Predicted next discrete state tensor (as logits) of shape (N, S_d)."""

    reward: Tensor
    """The predicted step reward tensor of shape (N, 1)."""

    done_logit: Tensor
    """The predicted probability logit of the game ending as a tensor of shape (N, 1)."""

    next_disc_state_prob: Tensor = None
    """The predicted next discrete state tensor as probabilities of shape (N, S_d)."""

    next_disc_state_thresholded: Tensor = None
    """Thresholded next_disc_state_prob as a tensor of shape (N, S_d)."""

    done_prob: Tensor = None
    """The predicted probability of the game ending as a tensor of shape (N, 1)."""

    done_thresholded: Tensor = None
    """Thresholded done_prob as a tensor of shape (N, 1)."""


class MonoModel(WalkerWorldModel):
    def __init__(
            self,
            cont_state_size: int,
            disc_state_size: int,
            action_size: int,
    ) -> None:
        super().__init__()

        self.cont_state_size = cont_state_size
        self.disc_state_size = disc_state_size
        self.total_state_size = cont_state_size + disc_state_size
        self.action_size = action_size

        self.thresholds = {'disc': (.5, .5), 'done': .5}

        self.state_end_index = self.reward_index = self.total_state_size
        self.done_index = self.reward_index + 1

        self.feature_transformer = LunarLanderFeatureTransformer()

        self.context_size = self.total_state_size + self.feature_transformer.addl_features + action_size

        activation_builders = [
            # nn.ReLU,
            # nn.LeakyReLU,
            nn.GELU,
            # lambda: nn.Dropout(p=0.5),
        ]

        # self.layers = nn.Sequential(
        #     *linear_layers(
        #         self.context_size,
        #         512,
        #         512,
        #         # (state, reward, done)
        #         self.total_state_size + 1 + 1,
        #         activation_builders=activation_builders,
        #     )
        # )

        self.state_norm = nn.BatchNorm1d(self.total_state_size + self.feature_transformer.addl_features)

        self.layers = nn.ModuleList([
            nn.Linear(self.context_size, 512),
            nn.Linear(512 + self.context_size, 512),
            nn.Linear(512 + self.context_size, self.total_state_size + 1 + 1),
        ])

        act_type = 'relu'
        self.activation = nn.ReLU()

        self.cbp = nn.ModuleList([
            CBPLinear(in_layer=self.layers[0], out_layer=self.layers[1], act_type=act_type),
            CBPLinear(in_layer=self.layers[1], out_layer=self.layers[2], act_type=act_type),
        ])

    def predict(self, input: WorldModelInput) -> WorldModelOutput:
        pred: WorldModelOutput = self(input)
        device = pred.next_cont_state.device

        pred.next_disc_state_prob = torch.sigmoid(pred.next_disc_state)
        thresholds = torch.tensor(self.thresholds['disc'], device=device)
        pred.next_disc_state_thresholded = (pred.next_disc_state_prob >= thresholds).float()

        pred.done_prob = torch.sigmoid(pred.done_logit)
        pred.done_thresholded = (pred.done_prob >= self.thresholds['done']).float()

        return pred

    def forward(self, input: WorldModelInput) -> WorldModelOutput:
        batch_size = input.cont_state.size(0)
        assert_shape(input.cont_state, (batch_size, self.cont_state_size))
        assert_shape(input.disc_state, (batch_size, self.disc_state_size))
        assert_shape(input.action, (batch_size, self.action_size))

        state = torch.cat([input.cont_state, input.disc_state], dim=1)

        state = self.feature_transformer(state)

        state = self.state_norm(state)

        mod_state = state.clone()
        # mod_state[:, 4:6] /= pi

        state_and_action = torch.cat([mod_state, input.action], dim=1)

        # preds = self.layers(state_and_action)
        preds = self.layers[0](state_and_action)
        preds = self.activation(preds)
        preds = torch.cat([preds, state_and_action], dim=1)
        preds = self.cbp[0](preds)
        preds = self.layers[1](preds)
        preds = self.activation(preds)
        preds = torch.cat([preds, state_and_action], dim=1)
        preds = self.cbp[1](preds)
        preds = self.layers[2](preds)

        next_cont_state = preds[:, :self.cont_state_size]
        next_cont_state += input.cont_state

        next_disc_state = preds[:, self.cont_state_size:self.total_state_size]

        reward = preds[:, self.reward_index].unsqueeze(1)

        done_logit = preds[:, self.done_index].unsqueeze(1)

        assert_shape(next_cont_state, (batch_size, self.cont_state_size))
        assert_shape(next_disc_state, (batch_size, self.disc_state_size))
        assert_shape(reward, (batch_size, 1))
        assert_shape(done_logit, (batch_size, 1))

        return WorldModelOutput(next_cont_state, next_disc_state, reward, done_logit)

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


class WalkerValueModel(Module):
    def __init__(
            self,
            cont_state_size: int = 22,
            disc_state_size: int = 2,
    ) -> None:
        super().__init__()

        # self.layers = nn.Sequential(
        #     *linear_layers(
        #         cont_state_size + disc_state_size,
        #         256,
        #         256,
        #         1,
        #         activation_builders=[
        #             # nn.ReLU,
        #             # nn.LeakyReLU,
        #             nn.GELU,
        #             # lambda: nn.Dropout(p=0.5),
        #         ],
        #     )
        # )

        self.feature_transformer = LunarLanderFeatureTransformer()

        context_size = cont_state_size + disc_state_size + self.feature_transformer.addl_features

        self.layers = nn.ModuleList([
            nn.Linear(context_size, 256),
            nn.Linear(256 + context_size, 256),
            nn.Linear(256 + context_size, 1),
        ])

        act_type = 'relu'
        self.activation = nn.ReLU()

        self.cbp = nn.ModuleList([
            CBPLinear(in_layer=self.layers[0], out_layer=self.layers[1], act_type=act_type),
            CBPLinear(in_layer=self.layers[1], out_layer=self.layers[2], act_type=act_type),
        ])

    def forward(self, input: ValueModelInput) -> Tensor:
        state = torch.cat([input.cont_state, input.disc_state], dim=1)

        # return self.layers(state).squeeze(1)

        state = self.feature_transformer(state)

        pred = self.layers[0](state)
        pred = self.activation(pred)
        pred = torch.cat([pred, state], dim=1)
        pred = self.cbp[0](pred)
        pred = self.layers[1](pred)
        pred = self.activation(pred)
        pred = torch.cat([pred, state], dim=1)
        pred = self.cbp[0](pred)
        pred = self.layers[2](pred)

        return pred.squeeze(1)


class LunarLanderFeatureTransformer(ConcatTransformer):
    def __init__(self):
        super().__init__()

        self.addl_features = 5 * 2

    def get_additional_features(self, batch: Tensor) -> Iterable[Tensor]:
        angle_indexes = [0, 4, 6, 8, 10]
        angles = [batch[:, i] for i in angle_indexes]

        features = [
            *(torch.sin(angle) for angle in angles),
            *(torch.cos(angle) for angle in angles),
        ]

        assert len(features) == self.addl_features, (len(features), self.addl_features)

        return features
