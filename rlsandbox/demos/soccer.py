import random

from rlsandbox.envs.renderers.soccer_renderer import SoccerEnvRenderer
from rlsandbox.envs.soccer import SoccerEnv, SoccerAction
from rlsandbox.types import Size2D


def main():
    env = SoccerEnv(field_size=Size2D(40, 20), max_steps=80)
    state = env.reset()

    env_renderer = SoccerEnvRenderer(env, fps=30, scale=30)
    env_renderer.render(state)

    while True:
        action = SoccerAction(
            move_dist=random.random(),
            turn_angle=0.,
            # turn_angle=(random.random() * 2 - 1) * 0.1,
            kick_strength=1.,
            # kick_strength=random.random(),
        )

        state_change = env.step(action)
        print(state_change)

        env_renderer.render(state_change.next_state)

        if state_change.done:
            state = env.reset()
            env_renderer.render(state)


if __name__ == '__main__':
    main()
