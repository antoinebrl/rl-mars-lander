import numpy as np


class RoverState:
    """Encapsulate the Mars Lander state.

    Note:
        The state is represented as a contiguous array but
        getters and setters are provided to ease its manipulation.

    Arg:
        state (list or np.array): an array of 6 physical measurements represented as floats.

    Attributes:
        x (float): rover's position along x-axis
        y (float): rover's position along y-axis
        vx (float): rover's speed along x-axis
        vy (float): rover's speed along y-axis
        fuel (float): remaining quantity of fuel in liters
        angle (float): rotation of the rover in degrees
        power (float): thrust power of the rover
    """

    def __init__(self, state):
        if isinstance(state, (list, tuple)):
            state = np.array(state, dtype=np.float32)
        self.state = state

    # Getters and Setters for the manipulation of the state
    @property
    def x(self):
        return self.state[0]

    @property
    def y(self):
        return self.state[1]

    @property
    def vx(self):
        return self.state[2]

    @property
    def vy(self):
        return self.state[3]

    @property
    def fuel(self):
        return self.state[4]

    @property
    def angle(self):
        return self.state[5]

    @property
    def power(self):
        return self.state[6]

    @x.setter
    def x(self, value):
        self.state[0] = value

    @y.setter
    def y(self, value):
        self.state[1] = value

    @vx.setter
    def vx(self, value):
        self.state[2] = value

    @vy.setter
    def vy(self, value):
        self.state[3] = value

    @fuel.setter
    def fuel(self, value):
        self.state[4] = value

    @angle.setter
    def angle(self, value):
        self.state[5] = value

    @power.setter
    def power(self, value):
        self.state[6] = value

    def __repr__(self):
        return (
            f"state: x={self.x:7.02f},  y={self.y:7.02f},  vx={self.vx:7.02f},  vy={self.vy:7.02f}\n"
            + f"       fuel={self.fuel:4.0f},  angle={self.angle:3.0f},  power={self.power:1.0f}"
        )

    __str__ = __repr__

    def numpy(self):
        return self.state

    def is_within_bounds(self, width, height):
        return 0 <= self.x < width and 0 <= self.y < height
