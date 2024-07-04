from rlsandbox.base import EnvTransformer, GymWrapper, State, StateChange


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


class TransformedLunarLanderEnv(EnvTransformer):
    def __init__(
            self,
            wrapped_env,
            max_y: float = None,
            reward_limit: float = 100.,
            neg_reward_gain: float = 1.,
    ):
        wrapped_env = GymWrapper(wrapped_env)
        super().__init__(wrapped_env)

        self.max_y = max_y
        self.reward_limit = reward_limit
        self.neg_reward_gain = neg_reward_gain

    def transform_state(self, state: State) -> State:
        return super().transform_state(state)

    def transform_state_change(self, state_change: StateChange) -> StateChange:
        reward = state_change.reward

        y = state_change.next_state[1]
        if self.max_y is not None and y >= self.max_y:
            reward = -100.
            state_change.done = True

        reward_limit = self.reward_limit
        reward = min(max(-reward_limit, reward), reward_limit)
        reward /= reward_limit

        if reward < 0:
            reward *= self.neg_reward_gain

        state_change.reward = reward

        return state_change
