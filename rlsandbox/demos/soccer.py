from rlsandbox.agents.soccer import SoccerAgent
from rlsandbox.envs.renderers.soccer_renderer import SoccerEnvRenderer
from rlsandbox.envs.soccer import SoccerEnv
from rlsandbox.types import Size2D


def main():
    env = SoccerEnv(field_size=Size2D(40, 20), max_steps=100)
    state = env.reset()

    env_renderer = SoccerEnvRenderer(env, fps=30, scale=30)
    env_renderer.render(state)

    agent = SoccerAgent()
    agent.reset()

    while True:
        action = agent.get_action(state)

        state_change = env.step(action)

        env_renderer.render(state_change.next_state)

        if state_change.done:
            state = env.reset()
            env_renderer.render(state)

            agent.reset()


if __name__ == '__main__':
    main()
