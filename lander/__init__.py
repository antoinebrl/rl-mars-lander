from gym.envs.registration import register

from .environment import MarsLanderEnv

register(
    id="MarsLander-v1",
    entry_point=MarsLanderEnv,
)
