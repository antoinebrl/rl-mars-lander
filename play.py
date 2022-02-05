from lander import MarsLanderEnv

if __name__ == "__main__":
    env = MarsLanderEnv()
    env.reset()

    while True:
        env.render(scale=10)

        action = input().split()
        while len(action) != 2:
            action = input().split()

        action[0] = int(action[0]) / 15
        if abs(action[0]) > 1:
            print("Invalid angle. Value should be between -15 and 15")

        action[1] = int(action[1])
        if abs(action[1]) > 1:
            print("Invalid thrust. Value should be between -1, 0 or 1")

        state, reward, done, info = env.step(action)
        if done:
            break

    env.close()
