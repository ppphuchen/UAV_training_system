from Env.Agent.base import *
import random
"""
    User class:  (Entity)

        properties:
            user value
            user state
            Connection status
            
        methods:
              
"""


class User(Entity):
    type = "User"
    # id count
    id_now = 0
    # aoi
    aoi_max = 3
    aoi_over = 1000
    aoi_fact = 9.999e-4
    aoi_change_pro = 0.0002
    aoi_change_time = 120
    aoi_change_wait = 20
    aoi_add_factor = 1
    # reward
    reward_factor = 1
    # dim obs
    dim_obs = 10  # [ ID, position[3], aoi_weight, aoi_now, aoi_time, aoi_total, aoi_change_factor, aoi_change_flag]
    dim_dqn = 9  # [ position[3], aoi_weight, aoi_now, aoi_time, aoi_total , aoi_change_factor, aoi_change_flat]
    dim_dqn_self = 10  # [ position[3], aoi_weight, aoi_now, aoi_time, aoi_total, distance ]

    def __init__(self, x, y, z):
        super(User, self).__init__()
        self.ID = User.id_now
        self.position = np.array([x, y, z], dtype=float)
        self.ground = (z == 0)  # The altitude coordinate is always zero

        # AoI and data
        self.aoi_weight = User.aoi_max * User.aoi_fact   # aoi 权重
        self.aoi_last_update = 0  # 上次访问的时间戳
        self.aoi_time = 0     # 距上次访问时间
        self.aoi_now = 0      # 当前 aoi
        self.aoi_total = 0    # 当前 aoi 累计
        self.aoi_last_time = 0  # 上次拖延时间 时间戳
        self.aoi_overflow = 0  # aoi累计溢出值
        self.aoi_overflow_punish = 0  # aoi惩罚累计
        self.aoi_change_flag = 0  # aoi突变标志，从准备阶段就置为1
        self.aoi_change_add = 0  # aoi突变计时器
        self.aoi_change_factor = 0  # aoi突变的倍数

        self.aoi_has_changed = False

        # time step
        self.time_step = 0
        # User Connection status
        self.request = False  # True means User is waiting an UAV to serve
        self.access = False   # 这两项在world com到get reward中间起作用
        self.connect_uavID = None

        # User observation
        self.observation = np.zeros(User.dim_obs)

        self.style = "fixed"  # Enum:[fixed, move, random]

        # update id count
        User.id_now += 1

    def aoi_weight_change(self):

        if self.ID == 9:
            # 固定就一个
            if self.aoi_change_add == 0:
                # 正常情况
                if not self.aoi_has_changed and self.time_step == 50:

                    # 是否突变
                    self.aoi_change_flag = 1
                    self.aoi_change_add += 1
                    print("change :", self.time_step)
                    self.aoi_has_changed = True

            else:
                # 已突变，是否改回
                if self.aoi_change_add >= User.aoi_change_time:
                    self.aoi_change_factor = 0
                    self.aoi_weight = User.aoi_max * User.aoi_fact
                    self.aoi_change_add = 0

                elif self.aoi_change_add == User.aoi_change_wait:
                    self.aoi_change_factor = User.aoi_add_factor
                    self.aoi_weight *= User.aoi_add_factor
                    self.aoi_change_add += 1
                    self.aoi_change_flag = 0

                else:
                    self.aoi_change_add += 1

    def aoi_weight_change_(self):

        if self.aoi_change_add == 0:
            # 正常情况
            if random.random() < User.aoi_change_pro:
                # 是否突变
                self.aoi_change_flag = 1
                self.aoi_change_add += 1
                print("change :", self.position, self.time_step)

        else:
            # 已突变，是否改回
            if self.aoi_change_add >= User.aoi_change_time:
                self.aoi_change_factor = 0
                self.aoi_weight = User.aoi_max * User.aoi_fact
                self.aoi_change_add = 0
                # print("end :", self.position)

            elif self.aoi_change_add == User.aoi_change_wait:

                self.aoi_change_factor = User.aoi_add_factor
                self.aoi_weight *= User.aoi_add_factor
                self.aoi_change_add += 1
                self.aoi_change_flag = 0

            else:
                self.aoi_change_add += 1

    def move(self):

        self.aoi_weight_change()

        self.time_step += 1
        self.aoi_time += 1
        self.aoi_now += self.aoi_weight

        # 加完是否溢出
        # if self.aoi_now > User.aoi_max:
        #     self.aoi_now = User.aoi_max
        #     self.aoi_overflow += self.aoi_weight
        #     self.aoi_overflow_punish += self.aoi_overflow

        self.aoi_total += self.aoi_now

        self.request = (self.aoi_now > 0)

        return self.position

    def get_punish(self):
        # 领取惩罚
        punish = self.aoi_overflow_punish
        self.aoi_overflow_punish = 0
        return punish

    def clean_aoi(self, timestep):

        self.aoi_last_update = timestep
        classic_reward = self.aoi_now * 1000
        if self.aoi_now < User.aoi_max:
            reward = self.aoi_now * (User.aoi_max - self.aoi_now) / self.aoi_weight
            reward += 0.5 * self.aoi_now * self.aoi_now / self.aoi_weight
            # print("reward: ",
            #       self.aoi_now * (User.aoi_max - self.aoi_now) / self.aoi_weight,
            #       " + ",
            #       0.5 * self.aoi_now * self.aoi_now / self.aoi_weight,
            #       ' = ',
            #       reward)

        else:
            reward = 0.5 * User.aoi_max * User.aoi_max / self.aoi_weight
            # print("reward: ", 0.5 * User.aoi_max * User.aoi_max / self.aoi_weight)

        self.aoi_overflow = 0
        self.aoi_total = 0
        self.aoi_time = 0
        self.aoi_now = 0
        # print("(",User.aoi_max,"-",aoi_now,")/" ,self.aoi_weight,\
        #         " - 0.5 * ",aoi_now,"^2",'/',self.aoi_weight,' = ',reward)

        return classic_reward * User.reward_factor

    def get_observation(self):

        # [ ID, position[3], aoi_weight, aoi_last, aoi_now]
        self.observation[0] = self.ID
        self.observation[1:4] = self.position
        self.observation[4] = self.aoi_weight
        self.observation[5] = self.aoi_time
        self.observation[6] = self.aoi_now
        self.observation[7] = self.aoi_total
        self.observation[8] = self.aoi_change_factor
        self.observation[9] = self.aoi_change_flag

        return self.observation

class MoveUser(User):

    def __init__(self, traj, max_speed=1):
        super(MoveUser, self).__init__(traj[0, 0], traj[1, 0], traj[2, 0])
        self.style = "move"
        self.traj = np.array(traj, dtype=float)  # size of trajectory numpy: [ 3 * N ]
        self.step = 0
        self.num_point = self.traj.shape[1]
        self.max_speed = max_speed

    def move(self):

        target_distance = euclidean_distance(self.position, self.traj[:, self.step], 3)

        if target_distance <= self.max_speed:
            # move to next tarjectory point, go ahead!
            self.position = self.traj[:, self.step]
            self.step = (self.step + 1) % self.num_point
        else:
            # move a small step
            per = self.max_speed/target_distance
            self.position += (self.traj[:, self.step] - self.position) * per

        return self.position


class RandomUser(User):

    def __init__(self, x, y, z=0, ground=True, max_speed=5):
        super(RandomUser, self).__init__(x, y, z)
        self.style = "random"
        self.position = np.array([x, y, z])
        self.max_speed = max_speed
        self.ground = ground

    def move(self):

        self.speed = random.random() * self.max_speed
        self.speed_dir[0] = random.random() * 2 * np.pi
        if self.ground is False:
            self.speed_dir[1] = random.uniform(-0.5*np.pi, 0.5*np.pi)

        return self.entity_move()
