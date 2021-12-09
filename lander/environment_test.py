import unittest

import numpy as np

from lander.environment import MarsLanderEnv


class MarsLanderEnvTest(unittest.TestCase):
    def test_detection_landing_area(self):
        # https://www.codingame.com/ide/puzzle/mars-lander  -  Initial speed, correct side
        ground = np.array(
            [
                [0, 100],
                [1000, 500],
                [1500, 100],
                [3000, 100],
                [3500, 100],
                [3700, 200],
                [5000, 1500],
                [5800, 100],
                [6000, 120],
                [6999, 2000],
            ]
        )

        # Test detection of landing area
        env = MarsLanderEnv(ground=ground)
        assert env.landing_area == [6, 11]

    def test_correct_physic(self):
        # https://www.codingame.com/ide/puzzle/mars-lander  -  Initial speed, correct side
        init_state = tuple(dict(x=6500, y=2800, vx=-100, vy=0, fuel=600, angle=90, power=0).values())
        env = MarsLanderEnv(init_state)

        for i in range(3):
            state = env.update_state(-15, 1)
        for i in range(4):
            state = env.update_state(-15, 0)
        state = env.update_state(-5, 0)
        for i in range(60):
            state = env.update_state(0, 0)

        np.testing.assert_array_equal(np.rint(state), [1191, 405, -44, -66, 399, -20, 3])

    def test_collision(self):
        # https://www.codingame.com/ide/puzzle/mars-lander  -  Initial speed, correct side
        initial_state = tuple(dict(x=6500, y=2800, vx=-100, vy=0, fuel=600, angle=90, power=0).values())
        ground = np.array(
            [
                [0, 100],
                [1000, 500],
                [1500, 100],
                [3000, 100],
                [3500, 100],
                [3700, 200],
                [5000, 1500],
                [5800, 100],
                [6000, 120],
                [6999, 2000],
            ]
        )

        env = MarsLanderEnv(initial_state, ground)
        for i in range(70):
            state, reward, done, info = env.step([-1, 1])
            if done:
                break

        self.assertEqual(i, 39)
        self.assertLess(reward, 0)

    def test_rover_exits_field_of_view(self):
        # https://www.codingame.com/ide/puzzle/mars-lander  -  Initial speed, correct side
        initial_state = tuple(dict(x=6500, y=2800, vx=-100, vy=0, fuel=600, angle=90, power=0).values())
        ground = np.array(
            [
                [0, 100],
                [1000, 500],
                [1500, 100],
                [3000, 100],
                [3500, 100],
                [3700, 200],
                [5000, 1500],
                [5800, 100],
                [6000, 120],
                [6999, 2000],
            ]
        )
        env = MarsLanderEnv(initial_state, ground)

        for i in range(6):
            state, reward, done, info = env.step([-1, 1])
        for i in range(64):
            state, reward, done, info = env.step([0, 0])
            if done:
                break
        self.assertEqual(i, 54)
        self.assertLess(reward, 0)

    def test_successful_landing(self):
        # https://www.codingame.com/ide/puzzle/mars-lander-episode-1  -  Straight landing
        initial_state = tuple(dict(x=2500, y=2500, vx=0, vy=0, fuel=500, angle=0, power=0).values())
        ground = np.array(
            [
                [0, 100],
                [1000, 500],
                [1500, 100],
                [3000, 100],
                [5000, 1500],
                [6999, 1000],
            ]
        )

        env = MarsLanderEnv(rover=initial_state, ground=ground)
        # Boost speed to max
        env.step([0, 1])
        env.step([0, 1])
        env.step([0, 1])
        env.step([0, 1])
        for i in range(70):
            state, reward, done, info = env.step([0, -1])
            if done:
                break
            state, reward, done, info = env.step([0, 1])

        self.assertEqual(i, 64)
        self.assertGreater(reward, 0)


if __name__ == "__main__":
    unittest.main()
