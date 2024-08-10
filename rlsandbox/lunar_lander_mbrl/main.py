import random
from collections import Counter
from typing import Optional

import gymnasium as gym
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from rlsandbox.lunar_lander_mbrl.agent import LunarLanderAgent
from rlsandbox.lunar_lander_mbrl.config import Config
from rlsandbox.lunar_lander_mbrl.dataset import RandomDropDataset, ExtendedStateChange, OnlyStateAndRewardSumDataset, \
    StateChangeDataset
from rlsandbox.lunar_lander_mbrl.env import WithActionRepeats, TransformedLunarLanderEnv
from rlsandbox.lunar_lander_mbrl.logging import log_code, log_metric, log_metrics
from rlsandbox.lunar_lander_mbrl.loss import NormalizedIfwBceWithLogitsLoss
from rlsandbox.lunar_lander_mbrl.metrics import compute_metrics, compute_metrics_at_threshold
from rlsandbox.lunar_lander_mbrl.model import get_world_model, LunarLanderValueModel
from rlsandbox.lunar_lander_mbrl.optim import OptimizerWrapper
from rlsandbox.lunar_lander_mbrl.utils import assert_shape, every


def main() -> None:
    run_name = input('Run name: ').strip()

    if run_name:
        mlflow.set_tracking_uri('http://potato:5000')
        mlflow.set_experiment('learn-env-model')
        with mlflow.start_run(
                run_name=run_name,
                log_system_metrics=True,
        ):
            _main()
    else:
        _main()


def _main(
        hp=Config(
            epochs=4000,

            # Env
            env_max_steps_per_game=1000,
            env_gravity=-10,
            env_enable_wind=False,
            env_action_repeats=4,
            env_max_y=1.75,
            env_reward_limit=100.,
            env_neg_reward_gain=1.0,

            # Training
            game_sars_history=1024 * 20,
            training_lr=0.001,
            training_lr_decay=.5,
            training_lr_decay_epochs=500,
            training_weight_decay=0.01,  # Default: 0.01
            max_batch_size=1024,
            valid_frac=0.1,
        ),
        device: torch.device = torch.device('cuda'),
) -> None:
    hp.max_valid_size = round(hp.game_sars_history * hp.valid_frac)

    log_code()
    mlflow.log_params(dict(hp))

    env_name = 'LunarLander-v2'
    env_args = dict(
        max_episode_steps=hp.env_max_steps_per_game,
        gravity=hp.env_gravity,
        enable_wind=hp.env_enable_wind,
    )
    train_env = WithActionRepeats(gym.make(env_name, **env_args), repeats=hp.env_action_repeats)
    watch_env = WithActionRepeats(gym.make(env_name, **env_args, render_mode='human'), repeats=hp.env_action_repeats)
    watch_env.metadata['render_fps'] = 999
    watch_env = TransformedLunarLanderEnv(
        watch_env,
        max_y=hp.env_max_y,
        reward_limit=hp.env_reward_limit,
        neg_reward_gain=hp.env_neg_reward_gain,
    )
    train_env = watch_env

    models = [
        get_world_model().to(device)
        for _ in range(2)
    ]

    value_model = LunarLanderValueModel().to(device)

    agent = LunarLanderAgent(models, value_model)

    regression_loss_class = nn.L1Loss
    state_loss_func_cont = regression_loss_class(reduction='none')
    state_loss_func_disc = NormalizedIfwBceWithLogitsLoss(reduction='none')
    reward_loss_func = regression_loss_class(reduction='none')
    done_loss_func = NormalizedIfwBceWithLogitsLoss(reduction='none')
    # done_loss_func = BCEWithLogitsFocalLoss(gamma=2, reduction='none')
    value_loss_func = nn.L1Loss()

    optimizers = [
        OptimizerWrapper(AdamW(
            model.parameters(),
            lr=hp.training_lr,
            weight_decay=hp.training_weight_decay,
        )) for model in models
    ]

    value_optimizer = OptimizerWrapper(AdamW(
        value_model.parameters(),
        lr=hp.training_lr,
        weight_decay=hp.training_weight_decay,
    ))

    dataset_class = RandomDropDataset
    all_train_game_sars = [
        dataset_class(max_len=hp.game_sars_history, device=device),
        dataset_class(max_len=hp.game_sars_history, device=device),
    ]
    valid_game_sars = dataset_class(max_len=hp.max_valid_size, device=device)
    all_total_rewards = []
    all_solved = []

    for epoch_i in range(hp.epochs):
        print()

        # model_i = epoch_i % len(models)
        model_i = random.randint(0, 1)
        game_sars = all_train_game_sars[model_i]

        # should_watch = every(20, epoch_i)
        should_watch = epoch_i >= 0  # 2000
        env = watch_env if should_watch else train_env

        state = env.reset()

        raw_rewards = []
        tmp_game_sars = []
        agent.eval()
        step_i = 0
        for step_i in range(hp.env_max_steps_per_game):
            # use_model = should_watch or step_i < (max_steps_per_game * epoch_i / epochs)
            # use_model = should_watch or random.random() < random_thres
            use_model = should_watch
            if use_model:
                with torch.no_grad():
                    action = agent.get_action(
                        state=torch.tensor(
                            state, dtype=torch.float32, device=device
                        ),
                    )
            else:
                action = env.action_space.sample()

            if every(100, step_i):
                print()

            state_change = env.step(action)
            reward = state_change.reward
            next_state = state_change.next_state
            done = state_change.done

            raw_rewards.append(reward)

            tmp_game_sars.append(ExtendedStateChange(
                state,
                action,
                reward,
                next_state,
                1. if done else 0.,
            ))

            if done:
                break

            state = next_state

        log_metric('episode_steps', step_i, step=epoch_i)
        agent.stats.emit(step=epoch_i)

        action_summary = Counter(it.action for it in tmp_game_sars)
        action_summary = np.array([action_summary[i] for i in range(4)])
        action_summary = action_summary / action_summary.sum()
        print()
        print(f'Actions: {action_summary}')

        total_reward = np.sum(raw_rewards)
        solved = total_reward >= 2
        solved_msg = '✔' if solved else '✘'
        print(f'Solved: {solved_msg}\t total reward: {total_reward}')
        print()
        all_total_rewards.append(total_reward)
        all_solved.append(1. if solved else 0.)
        mlflow.log_metrics(
            dict(
                total_reward=total_reward,
                total_reward_mean=np.mean(all_total_rewards),
                total_reward_mean_last_100=np.mean(all_total_rewards[-100:]),
                solved=1 if solved else 0,
                solved_mean=np.mean(all_solved),
                solved_mean_last_100=np.mean(all_solved[-100:]),
            ),
            step=epoch_i,
        )

        # Add values to game_sars
        values = [it.reward for it in reversed(tmp_game_sars)]
        values = np.cumsum(values)
        values = values[::-1]
        for sars, value in zip(tmp_game_sars, values):
            sars.reward_sum = value

        # Add to validation set
        if hp.valid_frac is not None and hp.valid_frac > 0:
            valid_sars_indices = set(random.sample(
                range(len(tmp_game_sars)),
                round(len(tmp_game_sars) * hp.valid_frac),
            ))

            tmp_valid_sars = [tmp_game_sars[i] for i in valid_sars_indices]

            valid_game_sars.extend(tmp_valid_sars)
            log_metric('valid_game_sars_len', len(valid_game_sars), step=epoch_i)

            tmp_game_sars = [
                sars for i, sars in enumerate(tmp_game_sars)
                if i not in valid_sars_indices
            ]

        game_sars.extend(tmp_game_sars)
        log_metric('game_sars_len', len(game_sars), step=epoch_i)

        epoch_losses = []
        agent.train()
        for model_i in (range(len(models)) if all(all_train_game_sars) else []):
            model = models[model_i]
            optimizer = optimizers[model_i]
            game_sars = all_train_game_sars[model_i]

            if every(hp.training_lr_decay_epochs, epoch_i):
                optimizer.lr = optimizer.lr * hp.training_lr_decay
            log_metric(f'optim_lr_model{model_i}', optimizer.lr, step=epoch_i)

            model.train()
            loss, losses = get_model_loss(
                hp.max_batch_size,
                epoch_i,
                model,
                model_i,
                game_sars,
                state_loss_func_cont,
                state_loss_func_disc,
                reward_loss_func,
                done_loss_func,
                dataset_type='train',
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            # Validate
            if every(10, epoch_i):
                model.eval()
                get_model_loss(
                    None,
                    epoch_i,
                    model,
                    model_i,
                    valid_game_sars,
                    state_loss_func_cont,
                    state_loss_func_disc,
                    reward_loss_func,
                    done_loss_func,
                    dataset_type='valid',
                )

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f'Epoch {epoch_i}: Avg loss = {avg_loss}')

            # TODO Remove this
            model.print_params_stats()

        # Train value model
        if 1:
            value_train_dataset = StateChangeDataset(device)
            value_train_dataset.data = (
                    all_train_game_sars[0].data +
                    all_train_game_sars[1].data
            )

            value_loss = get_value_loss(
                hp.max_batch_size,
                value_model,
                value_train_dataset,
                value_loss_func,
                epoch_i,
                'train',
            )

            if every(hp.training_lr_decay_epochs, epoch_i):
                value_optimizer.lr = value_optimizer.lr * hp.training_lr_decay
            log_metric(f'optim_lr_value', value_optimizer.lr, step=epoch_i)

            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

            # Validate
            if every(10, epoch_i):
                get_value_loss(
                    None,
                    value_model,
                    valid_game_sars,
                    value_loss_func,
                    epoch_i,
                    'valid',
                )

        # Save models to MLflow
        if every(100, epoch_i):
            for i, model in enumerate(models):
                mlflow.pytorch.log_model(model, f'model_{i}')

            mlflow.pytorch.log_model(value_model, 'value_model')


def get_model_loss(
        max_batch_size: Optional[int],
        epoch_i,
        model,
        model_i,
        game_sars: StateChangeDataset,
        state_loss_func_cont,
        state_loss_func_disc,
        reward_loss_func,
        done_loss_func,
        dataset_type: str,
):
    if max_batch_size is None:
        batch_size = len(game_sars)
    else:
        batch_size = min(max_batch_size, len(game_sars))

    data_loader = DataLoader(game_sars, batch_size=batch_size, shuffle=True)
    state, action, reward, next_state, done = next(iter(data_loader))

    pred_state, pred_reward, pred_done_logit = model(state, action)

    state_loss_cont = state_loss_func_cont(pred_state[:, :6], next_state[:, :6])
    state_loss_disc = state_loss_func_disc(pred_state[:, 6:], next_state[:, 6:])
    state_loss = torch.cat([state_loss_cont, state_loss_disc], dim=1)
    assert_shape(state_loss, (batch_size, 8))

    reward_loss = reward_loss_func(pred_reward, reward.unsqueeze(1))
    assert_shape(reward_loss, (batch_size, 1))

    done_loss = done_loss_func(pred_done_logit, done.unsqueeze(1))
    assert_shape(done_loss, (batch_size, 1))

    losses = torch.cat([state_loss, reward_loss, done_loss], dim=1)
    assert_shape(losses, (batch_size, 10))
    losses = losses.mean(dim=0)

    log_prefix = f'{dataset_type}_loss'
    loss_labels = ['x', 'y', 'dx', 'dy', 'theta', 'dtheta', 'leg_l', 'leg_r', 'reward', 'done']
    losses_loggable = losses.detach().cpu().numpy()
    for loss_label, l in zip(loss_labels, losses_loggable):
        key = f'{log_prefix}_model{model_i}_{loss_label}'
        log_metric(key, l, step=epoch_i)

    loss = losses.mean()

    log_metric(f'{log_prefix}_model{model_i}', loss.item(), step=epoch_i)

    if model_i == 0:
        try:
            log_model_bin_class_metrics(
                dataset_type,
                model_i,
                'done',
                epoch_i,
                y_true=done.cpu().numpy(),
                y_pred=F.sigmoid(pred_done_logit.detach().squeeze(1)).cpu().numpy(),
            )
        except ValueError as e:
            print(repr(e))

        try:
            true_leg_l = next_state[:, 6]
            pred_leg_l_logit = pred_state[:, 6]

            log_model_bin_class_metrics(
                dataset_type,
                model_i,
                'leg_l',
                epoch_i,
                y_true=true_leg_l.cpu().numpy(),
                y_pred=F.sigmoid(pred_leg_l_logit.detach()).cpu().numpy(),
            )
        except ValueError as e:
            print(repr(e))

    return loss, losses


def log_model_bin_class_metrics(
        dataset_type: str,
        model_i: int,
        feature: str,
        epoch_i: int,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        # thresholds=(.1, .3, .5, .7),
        thresholds=None,
) -> None:
    all_metrics = {}

    metrics = compute_metrics(y_true, y_pred)
    for key, value in metrics.items():
        all_metrics[f'{dataset_type}_model{model_i}_{feature}_{key}'] = value

    for threshold in (thresholds or []):
        metrics = compute_metrics_at_threshold(y_true, y_pred, threshold=threshold)
        for key, value in metrics.items():
            all_metrics[f'{dataset_type}_model{model_i}_{feature}_{key}_threshold_{threshold}'] = value

    log_metrics(all_metrics, step=epoch_i)


def get_value_loss(
        max_batch_size: Optional[int],
        value_model,
        game_sars: StateChangeDataset,
        value_loss_func,
        epoch_i,
        dataset_type: str,
):
    game_sars = OnlyStateAndRewardSumDataset(game_sars)

    if max_batch_size is None:
        batch_size = len(game_sars)
    else:
        batch_size = min(max_batch_size, len(game_sars))

    data_loader = DataLoader(game_sars, batch_size=batch_size, shuffle=True)

    state, value = next(iter(data_loader))

    value_model.train()
    value_pred = value_model(state)
    value_loss = value_loss_func(value_pred, value)

    log_metric(f'{dataset_type}_loss_value', value_loss.item(), step=epoch_i)

    return value_loss


def get_game_sars_column(game_sars, column: int, device, dtype=torch.float32) -> Tensor:
    column = [row[column] for row in game_sars]
    return torch.tensor(column, dtype=dtype, device=device)
