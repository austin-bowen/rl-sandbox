from typing import Any
from time import sleep

import gymnasium as gym


class Agent:
    def reset(self):
        pass

    def get_action(self, state: Any) -> Any:
        raise NotImplementedError()

    def process_transition(
            self,
            prev_state: Any,
            action: Any,
            next_state: Any,
            done: bool,
            reward: float
    ) -> Any:
        pass


class RandomAgent(Agent):
    def __init__(self, env):
        super().__init__()

        self.env = env

    def get_action(self, state: Any) -> Any:
        return self.env.action_space.sample()


class EnvRunner:
    def __init__(
            self,
            env, agent: Agent,
            render: bool = True,
            fps: float = None
    ):
        self.env = env
        self.agent = agent
        self.is_render = render
        self.fps = fps

    def run_forever(self) -> None:
        try:
            while True:
                self.run_one_episode()
        except KeyError:
            pass

    def run_one_episode(self) -> None:
        state = self.reset()
        self._render()

        step = 0
        done = False
        while not done:
            action = self.agent.get_action(state)

            if self.fps is not None:
                sleep(1 / self.fps)

            next_state, reward, done, *_ = self.env.step(action)
            self._render()

            # TODO: REMOVE THIS
            print(f'Step {step}:\t action={action};\t reward={reward}')

            self.agent.process_transition(
                state, action, next_state, done, reward
            )

            step += 1
            state = next_state

    def reset(self) -> Any:
        self.agent.reset()
        return self.env.reset()

    def _render(self):
        if self.is_render:
            self.env.render()


def run_env(env, agent: Agent):
    runner = EnvRunner(env, agent, fps=60)

    try:
        runner.run_forever()
    finally:
        env.close()


def do_bipedal_walker():
    env = gym.make("BipedalWalker-v3", render_mode='human')
    agent = RandomAgent(env)
    run_env(env, agent)


def do_car_racing():
    env = gym.make("CarRacing-v2", render_mode='human')
    agent = RandomAgent(env)
    run_env(env, agent)


def do_lunar_lander():
    env = gym.make("LunarLander-v2", render_mode='human')
    agent = RandomAgent(env)
    run_env(env, agent)


def do_soccer():
    env = ...
    agent = RandomAgent(env)
    run_env(env, agent)


def main():
    do_bipedal_walker()
    # do_car_racing()
    # do_lunar_lander()
    # do_soccer()


if __name__ == '__main__':
    main()
