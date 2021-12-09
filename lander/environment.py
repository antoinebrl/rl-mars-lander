import math

import gym
from gym import spaces
import numpy as np
import pyglet
from pyglet.gl import *

from lander.geometry import convert_to_fixed_length_polygon, find_flat_segment, is_inside_polygon
from lander.rover_state import RoverState

TEST_CASES_CG = [
    # Episode II
    {
        # Easy on the right
        "ground": [[0, 100], [1000, 500], [1500, 1500], [3000, 1000], [4000, 150], [5500, 150], [6999, 800]],
        "rover": [2500, 2700, 0, 0, 550, 0, 0],
    },
    {
        # Initial speed, correct side
        "ground": [[0, 100], [1000, 500], [1500, 100], [3000, 100], [3500, 500], [3700, 200], [5000, 1500], [5800, 300],
                   [6000, 1000], [6999, 2000]],
        "rover": [6500, 2800, -100, 0, 600, 90, 0],
    },
    {
        # Initial speed, wrong side
        "ground": [[0, 100], [1000, 500], [1500, 1500], [3000, 1000], [4000, 150], [5500, 150], [6999, 800]],
        "rover": [6500, 2800, -90, 0, 750, 90, 0],
    },
    {
        # Deep canyon
        "ground": [[0, 1000], [300, 1500], [350, 1400], [500, 2000], [800, 1800], [1000, 2500], [1200, 2100],
                   [1500, 2400], [2000, 1000], [2200, 500], [2500, 100], [2900, 800], [3000, 500], [3200, 1000],
                   [3500, 2000], [3800, 800], [4000, 200], [5000, 200], [5500, 1500], [6999, 2800]],
        "rover": [500, 2700, 100, 0, 800, -90, 0],
    },
    {
        # Deep canyon
        "ground": [[0, 1000], [300, 1500], [350, 1400], [500, 2100], [1500, 2100], [2000, 200], [2500, 500],
                   [2900, 300], [3000, 200], [3200, 1000], [3500, 500], [3800, 800], [4000, 200], [4200, 800],
                   [4800, 600], [5000, 1200], [5500, 900], [6000, 500], [6500, 300], [6999, 500]],
        "rover": [6500, 2700, -50, 0, 1000, 90, 0],
    },
]


class MarsLanderEnv(gym.Env):
    """
    Custom Environment that follows Open AI gym interface.
    The physic corresponds to the Mars Lander game from CodinGame:
    https://www.codingame.com/multiplayer/optimization/mars-lander
    The goal is to minimize the remaining fuel used for landing the rover.
    """

    metadata = {"render.modes": ["human"], "video.frames_per_second": 30}

    # Constants
    G = 3.711  # meters/sec^2
    SCENE_WIDTH = 7000  # meters
    SCENE_HEIGHT = 3000  # meters
    THRUST_MIN = 0
    THRUST_MAX = 4
    ROT_MIN = -90  # degrees
    ROT_MAX = 90  # degrees
    ROT_MAX_STEP = 15  # degrees

    # Actions: continuous values for angle and thrust
    action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=int)

    # Observations:
    # - ground: surface of Mars as a broken line. 30 pairs of 2d points (x,y).
    #       The scene is considered square
    # - rover: Rover state
    #       position: 0 <= x <= SCENE_WIDTH and 0 <= y <= SCENE_HEIGHT
    #       velocity: -500 <= vx, vy <= 500
    #       content of the tank: 0 <= fuel <= 2000
    #       tilt angle to vertical (in degrees): -90 <= angle <= 90
    #       current thrust: 0 <= power <= 4
    observation_space = spaces.Dict({
        "ground": spaces.Box(
            low=0,
            high=max(SCENE_WIDTH, SCENE_HEIGHT),
            shape=(30, 2),
            dtype=np.float32
        ),
        "rover": spaces.Box(
            low=np.array([0, 0, -500, -500, 0, ROT_MIN, 0]),
            high=np.array([SCENE_WIDTH, SCENE_HEIGHT, 500, 500, 2000, ROT_MAX, 4]),
            dtype=np.float32
        )
    })

    def __init__(self, rover=None, ground=None):
        super().__init__()
        self.iter = -1
        self.viewer = None

        self.rover = None
        if not isinstance(rover, RoverState):
            self.rover = RoverState(rover)
        self.ground = ground

    def __repr__(self):
        return repr(self.rover)

    __str__ = __repr__

    @property
    def ground(self):
        return self._surface

    @ground.setter
    def ground(self, value):
        if value is None:
            self._surface = None
        else:
            ground = np.array(value, dtype=np.float32)
            ground = convert_to_fixed_length_polygon(ground, n=30)
            self.landing_area = find_flat_segment(ground)
            self._surface = np.array(ground, dtype=np.float32)

    def update_state(self, angle, trust):
        # Update rotation.  Value of the previous turn +/-15°.
        self.rover.angle += np.rint(angle)
        self.rover.angle = np.clip(self.rover.angle, -90, 90)


        # Adjust engine power. Value of the previous turn +/-1.
        self.rover.power += np.rint(trust)
        self.rover.power = np.clip(self.rover.power, 0, 4)

        # Tank content. For a thrust power of T, T liters of fuel are consumed.
        self.rover.fuel -= abs(self.rover.power)

        # Newton's second law projected along X and Y
        new_vy = self.rover.vy + math.cos(math.radians(self.rover.angle)) * self.rover.power - self.G
        self.rover.y += (new_vy + self.rover.vy) / 2
        self.rover.vy = new_vy

        new_vx = self.rover.vx - math.sin(math.radians(self.rover.angle)) * self.rover.power
        self.rover.x += (new_vx + self.rover.vx) / 2
        self.rover.vx = new_vx

        return self.rover.numpy()

    def is_landing_successful(self):
        flat_area = self.ground[self.landing_area[0]][0] <= self.rover.x <= self.ground[self.landing_area[1]][0]
        no_angle = self.rover.angle == 0
        low_speed = abs(self.rover.vy) < 40 and abs(self.rover.vx) < 20
        return flat_area and no_angle and low_speed

    def step(self, action):
        self.iter += 1
        done = False
        reward = 1
        info = {}

        angle, trust = action
        assert -1 <= angle <= 1
        assert -1 <= trust <= 1

        angle = np.rint(angle * 15)
        trust = -1 if trust < -1/3 else 0 if trust < 1/3 else 1
        self.update_state(angle, trust)

        if not self.rover.is_within_bounds(width=self.SCENE_WIDTH, height=self.SCENE_HEIGHT):
            done = True
            reward = -100
            info = {"Rover is running away!"}
        else:
            grounded = is_inside_polygon(self.ground, self.rover.x, self.rover.y, self.SCENE_HEIGHT)
            if grounded:
                done = True

                mission_completed = self.is_landing_successful()
                if not mission_completed:
                    reward = -100
                    info = {"Rover has been destroyed"}
                else:
                    reward = self.rover.fuel
                    info = {"Mission accomplished!"}

        observation = {"ground": self.ground, "rover": self.rover.numpy()}
        return observation, reward, done, info

    def render(self, mode="human", scale=4):
        if mode != "human":
            raise ValueError(f"{mode} is not a valid render mode. Only 'human' is supported.")
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(self.SCENE_WIDTH // scale, self.SCENE_HEIGHT // scale)
            self.viewer.set_bounds(0, self.SCENE_WIDTH, 0, self.SCENE_HEIGHT)

            self.rovertrans = rendering.Transform()

            self.label = pyglet.text.Label(
                "",
                x=50,
                y=self.SCENE_HEIGHT // scale - 50,
                width=4000,
                anchor_x='left',
                anchor_y='top',
                font_name="Monospace",
                font_size=64 // scale,
                color=(255, 255, 255, 255),
                multiline=True
            )

        # background
        self.viewer.draw_polygon([
            [0, 0],
            [self.SCENE_WIDTH, 0],
            [self.SCENE_WIDTH, self.SCENE_HEIGHT],
            [0, self.SCENE_HEIGHT]
        ],
            color=(0.2, 0.2, 0.2)
        )
        # Surface line
        self.viewer.draw_polyline(self.ground.astype(int), linewidth=2, color=(0.7, 0, 0))

        # Rover
        size = 100
        rover = self.viewer.draw_polygon(np.array([
            [- size, - size],
            [+ size, - size],
            [+ size // 2, + size],
            [- size // 2, + size]
        ], dtype=int), color=(0.9, 0, 0))

        rover.add_attr(self.rovertrans)

        self.rovertrans.set_translation(self.rover.x, self.rover.y)
        self.rovertrans.set_rotation(math.radians(self.rover.angle))

        # Text
        self.label.text = repr(self.rover)

        # Update render
        self.viewer.window.switch_to()
        self.viewer.window.dispatch_events()
        self.viewer.transform.enable()
        for geom in self.viewer.geoms:
            geom.render()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.transform.disable()
        self.label.draw()
        self.viewer.onetime_geoms = []
        self.viewer.window.flip()
        return

    def reset(self):
        landing_area = [0, 0]

        # Choose between augmented test cases and random generation
        coin = np.random.rand()

        if coin < 0.5:
            # Generate random map and state
            nb_point = np.random.randint(6, 20)
            ground = np.random.rand(nb_point, 2)
            ground[:, 0] *= self.SCENE_WIDTH
            ground[:, 1] *= self.SCENE_HEIGHT * 0.80
            ground = ground[ground[:, 0].argsort()]

            # smooth land
            for i in range(1, nb_point - 1):
                ground[i, 1] = ground[(i - 1):(i + 1), 1].sum() / 3

            landing_area[0] = np.random.randint(1, nb_point - 3)

            # Position rover in the air
            rover = RoverState(self.observation_space["rover"].sample())
            while is_inside_polygon(ground, rover.x, rover.y - 100, self.SCENE_HEIGHT):
                rover = RoverState(self.observation_space["rover"].sample())

            rover.vx = (np.random.rand() - 0.5) * 100 * 2
            rover.vy = (np.random.rand() - 0.75) * 100
            rover.power = np.clip(rover.power, 0, 1)

        else:
            # Augment a test case
            test_idx = np.random.randint(len(TEST_CASES_CG))
            ground = np.array(TEST_CASES_CG[test_idx]["ground"], dtype=np.float32)
            rover = RoverState(TEST_CASES_CG[test_idx]["rover"])
            landing_area = find_flat_segment(ground)

            # flip left-right
            if np.random.rand() < 0.5:
                ground[:, 0] = self.SCENE_WIDTH - ground[:, 0]
                ground = ground[::-1]
                landing_area = [len(ground) - landing_area[0], len(ground) - landing_area[1]]

                rover.x = self.SCENE_WIDTH - rover.x
                rover.vx *= -1
                rover.angle *= -1

            # random offset
            if np.random.rand() < 0.5:
                magnitude = np.random.rand() * 0.08
                scale = np.array([self.SCENE_WIDTH, self.SCENE_HEIGHT]) * magnitude
                ground += np.random.rand(len(ground), 2) * scale
                ground[landing_area[1], 1] = ground[landing_area[0], 1]

            # delete a point
            if np.random.rand() < 0.5:
                index = np.random.randint(len(ground))
                while landing_area[0] <= index <= landing_area[0]:
                    index = np.random.randint(len(ground))
                ground = np.delete(ground, index, 0)
                landing_area = find_flat_segment(ground)

            # Modify rover sensors
            rover.x += (np.random.rand() - 0.5) * 150 * 2
            rover.x = np.clip(rover.x, 0, self.SCENE_WIDTH)

            rover.y += (np.random.rand() - 0.5) * 150 * 2
            rover.y = np.clip(rover.y, 0, self.SCENE_HEIGHT)

            rover.vx += (np.random.rand() - 0.5) * 10
            rover.vy += (np.random.rand() - 0.75) * 5

            rover.angle += np.random.rand() - 0.5 * 15 * 2
            rover.angle = np.clip(rover.angle, -90, 90)

        # Ensure large landing area
        ground[landing_area[0] + 1, 1] = ground[landing_area[0], 1]
        i = 1
        while landing_area[0] + i < len(ground) - 1 \
                and ground[landing_area[0] + i, 0] - ground[landing_area[0], 0] <= 1000:
            ground[landing_area[0] + i, 1] = ground[landing_area[0], 1]
            i += 1
        ground[landing_area[0] + i, 1] = ground[landing_area[0], 1]

        ground[:, 0] = np.clip(ground[:, 0], 0, self.SCENE_WIDTH)
        ground[:, 1] = np.clip(ground[:, 1], 0, self.SCENE_HEIGHT)
        ground[0, 0] = 0
        ground[-1, 0] = self.SCENE_WIDTH - 1
        self.ground = ground
        self.rover = rover

        return {"ground": self.ground, "rover": self.rover.numpy()}

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    import time
    from gym.utils.env_checker import check_env

    env = MarsLanderEnv()
    env.reset()
    check_env(env, warn=True, skip_render_check=True)

    for i in range(100):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        env.render()
        if done:
            break
        time.sleep(0.050)

    env.close()
