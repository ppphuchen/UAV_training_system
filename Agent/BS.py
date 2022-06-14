from Env.Agent import *
import numpy as np

"""
    Base-Station class:  (Entity)
        
        properties:
            number about UAVs
            number about neighbor BSs
            
            number about Sub-Channel and time-slot
            

        methods:
            distance    
"""


class BS(Entity):
    type = "BS"
    height = 200 # the height of BS is fixed
    home_radius = 30
    user_view = 3000

    Env_time = 1000

    # some of dim      0-2           3         4         5
    dim_obs = 6  # [ position[3], num_uav, num_user, time_step]
    dim_dqn = 4  # [ position[3], time_step]
    dim_act = 3  # [ speed, speed_dir[2] ]

    # id count
    id_now = 0

    def __init__(self, x, y, owned_uav):
        super(BS, self).__init__()

        # basic properties
        self.time_step = 0
        self.ID = BS.id_now    # BS的ID就是world.BSs[]的下标
        self.position = np.array([x, y, BS.height])
        self.num_neighbor_BS = 1
        self.num_owned_UAV = owned_uav
        self.reset()

        # communication properties
        # 本实验的暂不涉及信道分配问题
        self.num_subChannel = 3
        self.num_timeSlot = 0

        # transfer with uav
        self.num_acc_UAV = 0
        self.acc_UAVs_id = []
        # sense from user
        self.num_serve_user = 0

        # state information 先假设BS这里state和action全是零
        self.observation = np.zeros(BS.dim_obs)
        self.user_observation = []
        self.action = np.zeros(BS.dim_act)

        # add id count
        BS.id_now += 1

    def reset(self):

        self.time_step = 0
        self.num_acc_UAV = self.num_owned_UAV
        self.num_charging_UAV = self.num_owned_UAV

    def move(self):

        #print("BS ", self.ID, " acc:", self.acc_UAVs_id)
        self.time_step += 1

    def get_observation(self, world):

        # bs obs 暂时的用途是基站自己的信息：位置、接入uav数、服务user数
        self.acc_UAVs_id = []
        self.num_acc_UAV = 0
        for uav in world.UAVs:
            if uav.access_BS_id == self.ID:
                # 同一基站内的所有uav信息进行记录传递
                self.acc_UAVs_id.append(uav.ID)
                self.num_acc_UAV += 1

        self.num_serve_user = 0
        self.user_observation.clear()
        for user in world.Users:
            if user.distance_with(self) < BS.user_view:
                self.num_serve_user += 1
                self.user_observation.append(user.get_observation())

        self.observation[0:3] = self.position
        self.observation[3] = self.num_acc_UAV
        self.observation[4] = self.num_serve_user
        self.observation[5] = self.time_step / BS.Env_time

        # 返回BS自身状态 + User状态列表
        return [self.observation, self.user_observation]

    def decode_action(self, action):
        # 动作解码
        pass
