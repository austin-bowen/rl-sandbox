class WithActionRepeats:
    def __init__(self, env, repeats: int = 1):
        self.env = env
        self.repeats = repeats

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def metadata(self):
        return self.env.metadata

    def reset(self):
        return self.env.reset()

    def step(self, action):
        total_reward = 0.

        for _ in range(self.repeats):
            next_state, reward, done, truncated, info = self.env.step(action)
            total_reward += reward

            if done or truncated:
                break

        return next_state, total_reward, done, truncated, info

    def _get_repeats(self, action: int) -> int:
        if isinstance(self.repeats, int):
            return self.repeats
        else:
            return self.repeats[action]
