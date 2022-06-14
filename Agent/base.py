import numpy as np
import math

"""
    agent base model : Entity
        Contains properties and methods common to some entities

        properties:
            ID
            position[longitude, latitude, height]
            speed
            speed_direction[horizontal, vertical]

        methods:
            distance
"""


class Entity(object):

    def __init__(self):
        # Entity's ID number
        self.ID = None
        # Entity's position: [longitude, latitude, height]
        self.position = np.zeros(3, dtype=float)
        # Entity's Linear Speed
        self.speed = 0
        # Entity's Speed direction [0,2π) in [horizontal, vertical]
        self.speed_dir = np.zeros(2, dtype=float)

        # state information : obs , act
        self.dim_obs = 6  # [ position[3], speed[3]]
        self.dim_act = 3  # [ speed[3] ]
        self.observation = None
        self.action = None

    def entity_move(self):

        self.position[2] += self.speed * math.sin(self.speed_dir[1])
        hor = self.speed * math.cos(self.speed_dir[1])
        self.position[1] += hor * math.sin(self.speed_dir[0])
        self.position[0] += hor * math.cos(self.speed_dir[0])

        self.position[2] = np.clip(self.position[2], 0, 500)
        self.position[0:2] = np.clip(self.position[0:2], -1000, 1000)
        # print(self.speed * math.sin(self.speed_dir[1]),
        #       hor * math.sin(self.speed_dir[0]),
        #       hor * math.cos(self.speed_dir[0]))
        return self.position

    def distance_with(self, B_entity):
        return euclidean_distance(self.position, B_entity.position, 3)


def distance_between(A_entity, B_entity):
    return euclidean_distance(A_entity.position, B_entity.position, 3)


def euclidean_distance(point_A, point_B, dim):
    sum = 0.0
    for i in range(dim):
        sum += (point_A[i] - point_B[i]) ** 2
    return sum ** 0.5


if __name__ == "__main__":
    a = Entity()
    b = Entity()
    print(distance_between(a, b))
