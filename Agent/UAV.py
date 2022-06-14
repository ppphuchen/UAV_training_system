from Env.Agent import *
from Env.Function.Energy import *
import numpy as np
import torch
"""
    User class:  (Entity)

        properties:
            user value
            

        methods:

"""


class UAV(Entity):
    type = "UAV"
    # some of constraint
    max_UAV_speed = 20
    max_UAV_energy = 100
    energy_charge_speed = 5 # 单独训练下层网络，不需要充电
    max_UAV_height = 1000
    max_connect_distance = 1000
    max_upload_power = 1
    max_download_power = 1

    # some of size （现在不用view的概念，在一个BS中所有信息都是通过广播透明的）
    BS_view_size = 10000
    User_view_size = 10000
    UAV_view_size = 10000
    BS_max_com_dis = 1600
    User_max_com_dis = 200

    # some of dim
                      #   0,1,2      3         4,5        6        7       8       9        10-12      13     14           15-17
    dim_obs = 18   # [ position[3], speed, speed_dir[2], energy, charge, trans, goal_id, goal_pos[3], life, goal_t_id, goal_t_pos[3]]
    dim_dqn_self = 5  # [ position[3], energy, life]
    dim_dqn_nei = 9  # [ position[3], energy, charge, goal_id, goal_pos[3] ]
                       #     0-2              3-5       6       7-9           10-12       13-15            16     17-18
    dim_low_obs = 16  # [ position[3],   goal_pos[3], life, goal_t_pos[3], goal_dis_dir, goal_t_dis_dir, speed, speed_dir[2]]
                    #  0        1,2          3      4      5         6       7-9         10       11-13
    dim_act = 14  # [ speed, speed_dir[2], trans, up_pow, down_pow, user_id, user_pos, next_id, next_pos]
    dim_low_act = 3  # [speed, speed_dir[2], trans, up_pow, down_pow]

    # transfer_rate
    data_collect = 1000
    # count id
    id_now = 0

    # lower action std         v     theta
    action_std = torch.Tensor([0.02, 0.01, 0.02, 0, 0.1, 0.1])

    def __init__(self, x, y, z):
        super(UAV, self).__init__()

        # basic info
        self.ID = UAV.id_now
        self.position = np.array([x, y, z], dtype=float)
        self.energy = UAV.max_UAV_energy
        self.energy_comm_step = 0
        self.energy_move_step = 0

        # 方向上有一个初始化的不同
        if self.ID % 2 == 1:
            self.speed_dir[0] = np.pi

        # some working state
        self.charge = -1  # -1: 不充电 0：开始充电  >0 :已经开始充电的时间
        self.trans = -1  # -1：不传输  0：开始传输  >0 :已经开始传输的时间
        self.mission_success = False  # 只有当communication后成功者被记为True，get_reward后置位False

        # next goal information
        self.goal_next_id = -1
        self.goal_next_pos = -1 * np.ones(3)

        # sense information
        self.goal_user_id = -1        # -1表示目的地是基站
        self.goal_user_pos = -1 * np.ones(3)
        self.goal_range = 0           # 目标范围
        self.goal_finished = False

        # transition information
        self.num_view_BS = None
        self.view_BSs_id = []   # BS的ID就是world.BSs[]的下标
        self.access_BS_id = -1  # -1表示未找到BS的状态
        self.access_BS_pos = np.empty(3)
        self.sub_channel = -1   # -1表示未建立连接，此处由BS分配信道后外部更改

        # sense information
        self.access_User_id = -1  # -1表示不准备连接

        self.success = False

        # energy control
        self.upload_power = 50        # 表示上行和下行传输的功率[0 , 100]
        self.download_power = 50

        # state information
        self.observation = np.empty(UAV.dim_obs)
        self.action = np.empty(UAV.dim_act)
        self.life = 0

        # add count id
        UAV.id_now += 1

    def energy_loss(self, derta):
        # return True
        self.energy -= derta
        if self.energy <= 0:
            self.energy = 0
            return False
        else:
            return True

    def move(self):

        # cut down the energy cost in moving
        self.energy_move_step = move_energy_cost(self.observation[3:6])
        if self.energy_loss(self.energy_move_step) is False:
            self.speed = 0
            self.energy_move_step = 0

        # 修改相应状态
        if euclidean_distance(self.access_BS_pos, self.position, 3) < BS.home_radius\
                and self.speed < 1:
            # 修改充电状态
            self.charge = True
            self.life = 0
            self.energy += UAV.energy_charge_speed
            if self.energy > UAV.max_UAV_energy:
                self.energy = UAV.max_UAV_energy
            else:
#                print("uav ", self.ID, " charging... [", self.energy, "/100")
                pass
        else:
            # 不在家或在基站上空飞行，就不是充电状态
            self.charge = False
            # 非充电情况下才工作，报告工作状态

            if self.access_BS_id == -1:
            #    print("\n\n-------------", "uav ", self.ID, " 出界-------------\n\n")
                pass
            if self.trans < 0:
            #    print("uav ", self.ID, " move")
                pass
            else:
            #    print("uav ", self.ID, " send to bs", self.access_BS_id, " from user ", self.goal_user_id)
                pass
        if self.energy > 0:
            self.entity_move()


        return None

    def get_observation(self, world):

        if self.energy == 0:
            self.die()
        # Todo：一块是BS的信息用于决策和哪一个基站进行通信，另一块是UAV自己的状态，不考虑邻居UAV观察
        # view the BSs
        self.view_BSs_id.clear()
        view_BSs_info = []
        for i in range(world.num_BSs):
            if distance_between(self, world.BSs[i]) <= UAV.BS_view_size:
                # BS in the view
                self.view_BSs_id.append(world.BSs[i].ID)
                # put BS into observation
                view_BSs_info.append([world.BSs[i].ID, world.BSs[i].position, world.BSs[i].num_acc_UAV])

        """  uav obs list
        [
            speed,
            speed_dir[2], 
            energy, 
            charge, 
            trans, 
            goal_id, 
            goal_pos[3]
        ]

        """

        # [position[3],
        # speed, speed_dir[2], energy, charge, trans, goal_id, goal_pos[3]]
        self.observation[0:3] = self.position
        self.observation[3] = self.speed
        self.observation[4:6] = self.speed_dir
        self.observation[6] = self.energy
        self.observation[7] = self.charge
        self.observation[8] = self.trans
        self.observation[9] = self.goal_user_id
        self.observation[10:13] = self.goal_user_pos
        self.observation[13] = self.life
        self.observation[14] = self.goal_next_id
        self.observation[15:18] = self.goal_next_pos

        return [self.observation, view_BSs_info]

    def decode_action(self, action):
        # 合理动作约束
        """
        [
            speed,
            speed_dir[2],
            trans,            开始传输决策 大于零True 小于等于零False
            up_pow,
            down_pow,
            user_id,
            user_pos[3]
        ]
        """
        if self.energy == 0:
            return None

        self.speed = np.clip(action[0], 0, UAV.max_UAV_speed)
        self.speed_dir = action[1:3]
        # print(action[1:3])
        # 只有再未开始传输的情况下，收到传输信号，uav会开始和目标通信
        if self.trans < 0 and action[3] >= 0:
            self.trans = 0
        self.upload_power = np.clip(action[4], 0.2, UAV.max_upload_power)
        self.download_power = np.clip(action[5], 0.2, UAV.max_download_power)
        self.goal_user_id = int(action[6])
        self.access_User_id = int(action[6])
        self.goal_user_pos = action[7:10]
        self.goal_next_id = action[10]
        self.goal_next_pos = action[11:14]
        # 后续还会有其他action再做补充

    def die(self):
        self.trans = -1
        self.access_User_id = -1
        self.access_BS_id = -1



