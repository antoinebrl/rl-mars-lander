from .environment import MarsLanderEnv


from gym.envs.registration import register


register(
    id="MarsLander-v1",
    entry_point=MarsLanderEnv,
)