import os
import sys

from stable_baselines3 import PPO

from lander import MarsLanderEnv

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python enjoy.py log/2022-01-01/")
        sys.exit(1)

    env = MarsLanderEnv()
    model = PPO.load(os.path.join(sys.argv[1], "best_model.zip"), env)
    obs = env.reset()

    while True:
        action, state = model.predict(observation=obs, deterministic=True)
        obs, reward, done, infos = env.step(action)
        env.render(scale=10)
        print(reward)
        if done:
            break
        input()

    env.close()
