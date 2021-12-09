import math

import gym
from gym import spaces
import numpy as np

from lander.geometry import convert_to_fixed_length_polygon, find_flat_segment
from lander.rover_state import RoverState


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
