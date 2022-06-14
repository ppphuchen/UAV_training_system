# Energy function
import math

from Env.Agent import *


def upload_energy_cost(power):
    """
    :param power:
    :return:  上传的能量消耗
    """

    energy = power

    return energy/2


def download_energy_cost(power):
    """

    :param power:
    :return:  下载的能量消耗
    """

    energy = power

    # print("energy:",energy)
    return energy/2


def move_energy_cost(speed):
    # speed
    horizontal_speed = speed[0] * math.cos(speed[2])   # x轴速度
    vertical_speed = speed[0] * math.sin(speed[2])  # y轴速度

    x = horizontal_speed / 10
    y = 0.0167 * x * x * x - 0.0017 * x * x - 0.1333 * x + 1.5  # 原型函数 x = 0-5 y = 2-3

    horizontal_energy = y / 10

    vertical_energy = vertical_speed / 100  # 垂直能量
    # if vertical_energy < 0 :
    #     vertical_energy = 0
    # return 0
    return horizontal_energy
