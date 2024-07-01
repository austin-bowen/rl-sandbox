import random
from collections import Counter

import gymnasium as gym
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torch.optim import AdamW

from rlsandbox.lunar_lander_mbrl.agent import LunarLanderAgent
from rlsandbox.lunar_lander_mbrl.config import Config
from rlsandbox.lunar_lander_mbrl.env import WithActionRepeats
from rlsandbox.lunar_lander_mbrl.loss import NormalizedIfwBceWithLogitsLoss
from rlsandbox.lunar_lander_mbrl.metrics import compute_metrics
from rlsandbox.lunar_lander_mbrl.model import LunarLanderWorldModel, LunarLanderValueModel
from rlsandbox.lunar_lander_mbrl.utils import assert_shape, every


class Wrapper:
    def __init__(self, obj):
        self.obj = obj

    def __getattr__(self, name):
        return getattr(self.obj, name)


class OptimizerWrapper(Wrapper):
    @property
    def lr(self) -> float:
        return self.param_groups[0]['lr']

    @lr.setter
    def lr(self, value: float) -> None:
        for param_group in self.param_groups:
            param_group['lr'] = value


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
            neg_reward_gain=1.0,

            # Training
            game_sars_history=1024 * 10,
            training_lr=0.001,
            training_weight_decay=0.01,  # Default: 0.01
            max_batch_size=1024,
            value_model_max_items=1024 * 10,
            valid_frac=0.1,
            max_valid_size=1024 * 10,
        ),
        device: torch.device = torch.device('cuda'),
) -> None:
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
    # watch_env = train_env

    models = [
        LunarLanderWorldModel().to(device)
        for _ in range(2)
    ]

    value_model = LunarLanderValueModel().to(device)
    # value_model = SklearnKnnRegressor(
    #     max_items=hp.value_model_max_items,
    #     n_neighbors=20,  # Default: 5
    #     weights='distance',  # Default: 'uniform'
    #     algorithm='auto',  # Default: 'auto'
    #     p=2,  # Default: 2
    #     n_jobs=-1,  # Default: None
    # )

    agent = LunarLanderAgent(models, value_model)

    regression_loss_class = nn.L1Loss
    state_loss_func_cont = regression_loss_class(reduction='none')
    state_loss_func_disc = NormalizedIfwBceWithLogitsLoss(reduction='none')
    reward_loss_func = regression_loss_class(reduction='none')
    done_loss_func = NormalizedIfwBceWithLogitsLoss(reduction='none')
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

    all_game_sars = [[], []]
    valid_game_sars = []
    all_total_rewards = []
    all_solved = []

    for epoch_i in range(hp.epochs):
        print()

        # model_i = epoch_i % len(models)
        model_i = random.randint(0, 1)
        game_sars = all_game_sars[model_i]

        # should_watch = every(20, epoch_i)
        should_watch = epoch_i >= 0  # 2000
        env = watch_env if should_watch else train_env

        state, info = env.reset()

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

            next_state, reward, done, truncated, info = env.step(action)

            # Kill if too high
            if 1:
                y = next_state[1].item()
                if y >= 1.75:
                    reward = -100.
                    done = True

            raw_rewards.append(reward)

            reward_limit = 100.
            reward = min(max(-reward_limit, reward), reward_limit)
            reward /= reward_limit
            if reward < 0:
                reward *= hp.neg_reward_gain

            tmp_game_sars.append([
                state,
                action,
                reward,
                next_state,
                1. if done or truncated else 0.,
            ])

            if done or truncated:
                break

            state = next_state

        log_metric('episode_steps', step_i, step=epoch_i)

        action_summary = Counter(it[1] for it in tmp_game_sars)
        action_summary = np.array([action_summary[i] for i in range(4)])
        action_summary = action_summary / action_summary.sum()
        print()
        print(f'Actions: {action_summary}')

        total_reward = np.sum(raw_rewards)
        solved = total_reward >= 200
        solved_msg = '✔' if total_reward >= 200 else '✘'
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

        if 1:
            # inputs = np.vstack([it[0] for it in tmp_game_sars])

            values = [it[2] for it in reversed(tmp_game_sars)]
            values = np.cumsum(values)
            values = values[::-1]
            for sars, value in zip(tmp_game_sars, values):
                sars.append(value)

            # value_model.add(inputs, values)

        # Add to validation set
        if 1:
            valid_sars_indices = set(random.sample(
                range(len(tmp_game_sars)),
                int(len(tmp_game_sars) * hp.valid_frac),
            ))
            tmp_valid_sars = [tmp_game_sars[i] for i in valid_sars_indices]
            valid_game_sars.extend(tmp_valid_sars)
            valid_game_sars = valid_game_sars[-hp.max_valid_size:]
            log_metric('valid_game_sars_len', len(valid_game_sars), step=epoch_i)

            tmp_game_sars = [sars for i, sars in enumerate(tmp_game_sars) if i not in valid_sars_indices]

        game_sars.extend(tmp_game_sars)

        epoch_losses = []
        agent.train()
        for model_i in (range(len(models)) if all(all_game_sars) else []):
            # if 1 or len(game_sars) >= hp.game_sars_history:
            model = models[model_i]
            optimizer = optimizers[model_i]
            game_sars = all_game_sars[model_i]

            if every(500, epoch_i):
                optimizer.lr = optimizer.lr * .5
            log_metric(f'optim_lr_model{model_i}', optimizer.lr, step=epoch_i)

            model.train()
            loss, losses = get_model_loss(
                hp,
                device,
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
                    hp,
                    device,
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

            keep_worst = 3
            game_sars = all_game_sars[model_i]
            if keep_worst and len(game_sars) > hp.game_sars_history:
                # Randomly choose based on loss
                if keep_worst == 1:
                    # Keep max_steps_per_game sars samples with the highest losses
                    keep_probs = F.softmax(losses.detach().mean(dim=1), dim=0).cpu().numpy()
                    assert_shape(keep_probs, (len(game_sars),))

                    indexes_to_keep = set(np.random.choice(
                        range(len(game_sars)),
                        size=hp.game_sars_history,
                        replace=False,
                        p=keep_probs,
                    ))

                # Keep the highest loss examples
                elif keep_worst == 2:
                    # Keep max_steps_per_game sars samples with the highest losses
                    keep_probs = F.softmax(losses.detach().mean(dim=1), dim=0).cpu().numpy()
                    assert_shape(keep_probs, (len(game_sars),))

                    indexes_to_keep = enumerate(keep_probs)
                    indexes_to_keep = sorted(indexes_to_keep, key=lambda _: _[1], reverse=True)
                    indexes_to_keep = indexes_to_keep[:hp.game_sars_history]
                    indexes_to_keep = {_[0] for _ in indexes_to_keep}

                # Keep uniformly random examples
                elif keep_worst == 3:
                    indexes_to_keep = set(np.random.choice(
                        range(len(game_sars)),
                        size=hp.game_sars_history,
                        replace=False,
                    ))

                else:
                    raise ValueError(f'Invalid keep_worst={keep_worst}')

                game_sars = [
                    sars for i, sars in enumerate(game_sars)
                    if i in indexes_to_keep
                ]
                assert len(game_sars) == hp.game_sars_history, len(game_sars)

                p_pos = sum(it[2] > 0. for it in game_sars) / len(game_sars)
                p_neg = sum(it[2] < 0. for it in game_sars) / len(game_sars)
                print('p_pos:', p_pos)
                print('p_neg:', p_neg)

            game_sars = game_sars[-hp.game_sars_history:]
            all_game_sars[model_i] = game_sars
            log_metric('game_sars_len', len(game_sars), step=epoch_i)

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f'Epoch {epoch_i}: Avg loss = {avg_loss}')

            # TODO Remove this
            model.print_params_stats()

        # Train value model
        if 1:
            game_sars = all_game_sars[0] + all_game_sars[1]

            if 1:
                game_sars = random.sample(game_sars, min(len(game_sars), hp.max_batch_size))

            value_loss = get_value_loss(
                value_model,
                game_sars,
                device,
                value_loss_func,
                epoch_i,
                'train',
            )

            if every(500, epoch_i):
                value_optimizer.lr = value_optimizer.lr * .5
            log_metric(f'optim_lr_value', value_optimizer.lr, step=epoch_i)

            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

            # Validate
            if every(10, epoch_i):
                get_value_loss(
                    value_model,
                    valid_game_sars,
                    device,
                    value_loss_func,
                    epoch_i,
                    'valid',
                )

        # Save models to MLflow
        if every(100, epoch_i):
            for model_i, model in enumerate(models):
                mlflow.pytorch.log_model(model, f'model_{model_i}')

            mlflow.pytorch.log_model(value_model, 'value_model')


def get_model_loss(
        hp,
        device,
        epoch_i,
        model,
        model_i,
        game_sars,
        state_loss_func_cont,
        state_loss_func_disc,
        reward_loss_func,
        done_loss_func,
        dataset_type: str,
):
    if hp.max_batch_size is not None:
        game_sars = random.sample(game_sars, min(len(game_sars), hp.max_batch_size))

    state = get_game_sars_column(game_sars, 0, device)
    action = get_game_sars_column(game_sars, 1, device, dtype=torch.int64)
    reward = get_game_sars_column(game_sars, 2, device)
    next_state = get_game_sars_column(game_sars, 3, device)
    done = get_game_sars_column(game_sars, 4, device)

    pred_state, pred_reward, pred_done_logit = model(state, action)

    state_loss_cont = state_loss_func_cont(pred_state[:, :6], next_state[:, :6])
    state_loss_disc = state_loss_func_disc(pred_state[:, 6:], next_state[:, 6:])
    state_loss = torch.cat([state_loss_cont, state_loss_disc], dim=1)
    batch_size = len(game_sars)
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

    for threshold in [0.05, 0.1, 0.5, 0.9]:
        metrics = compute_metrics(
            y_true=done.cpu().numpy(),
            y_pred=F.sigmoid(pred_done_logit.detach().squeeze(1)).cpu().numpy(),
            threshold=threshold,
        )
        for key, value in metrics.items():
            log_metric(f'{dataset_type}_model{model_i}_done_{key}_threshold_{threshold}', value, step=epoch_i)

    return loss, losses


def get_value_loss(
        value_model,
        game_sars,
        device,
        value_loss_func,
        epoch_i,
        dataset_type: str,
):
    state = get_game_sars_column(game_sars, 0, device)
    value = get_game_sars_column(game_sars, 5, device)

    value_model.train()
    value_pred = value_model(state)
    value_loss = value_loss_func(value_pred, value)

    log_metric(f'{dataset_type}_loss_value', value_loss.item(), step=epoch_i)

    return value_loss


def get_game_sars_column(game_sars, column: int, device, dtype=torch.float32) -> Tensor:
    column = [row[column] for row in game_sars]
    return torch.tensor(column, dtype=dtype, device=device)


def log_metric(key: str, value, step: int = None) -> None:
    if step is None:
        print(f'{key}={value}')
    else:
        print(f'{key}={value} (step={step})')

    mlflow.log_metric(key, value, step=step)


def log_metrics(metrics: dict[str, ...], step: int = None) -> None:
    for key, value in metrics.items():
        log_metric(key, value, step=step)