import numpy as np
import math

def upload_probability(world, uav_id, bs_id):
    """

    :param world:
    :param uav_id:
    :param bs_id:
    :return: 上传概率
    """
    max_dis = UAV.BS_max_com_dis

    dis = distance_between(world.UAVs[uav_id], world.BSs[bs_id])
    pro = 1.2 - math.exp(1.8 * dis / max_dis) * 0.2
    # print(pro, dis)
    if pro >= 1:
        return 1
    elif pro < 0:
        return 0
    else:
        return pro


def download_probability(world, uav_id, user_id):

    # max_dis = UAV.User_max_com_dis * (world.UAVs[uav_id].download_power / UAV.max_download_power)
    max_dis = UAV.User_max_com_dis

    user_target = world.Users[user_id].position.copy()
    user_target[2] = 200

    dis = euclidean_distance(world.UAVs[uav_id].position,
                             user_target,
                             dim=3)

    pro = 1.2 - math.exp(1.8 * dis/max_dis) * 0.2
    # if pro > 0:
    #     print(dis,pro,world.UAVs[uav_id].position, user_target)
    if pro >= 1:
        return 1
    elif pro < 0:
        return 0
    else:
        return pro

def get_uav_state(matrix_distance, matrix_angle,matrix_agent,now_step):
    """
    更新UAV的状态

    :param matrix_distance:(array) matrix_distance[i]为第i个UAV的归一化距离
    :param matrix_angle:(array) matrix_angle[i]为第i个UAV的角度(弧度制)
    :param matrix_agent:(array) matrix_agent[i]为第i个UAV智能体实例
    :param now_step:(int) 当前的时隙数
    :return: new_state:(array) 更新后UAV智能体组的状态
    """

    new_state = [[], [], [], 0, ]
    new_state[:][0] = matrix_distance
    new_state[:][1] = matrix_angle
    new_state[:][2] = matrix_agent
    new_state[0][3] = now_step
    return  new_state