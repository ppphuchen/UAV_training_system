from Env.Function.Communication import *
from Env.Function.Energy import *
from Algorithm.Idiot import *

import copy
import random
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_Env_time():
    return World.Env_time


class World(object):
    Env_time = 1000

    def __init__(self,
                 alg,
                 world_size,
                 save_name,
                 show_3d,
                 discount=0.9):

        self.world_size = world_size
        self.discount = discount
        # model
        self.model = alg
        self.model.save_name += save_name

        # numbers
        self.num_UAVs = None
        self.num_BSs = None
        self.num_Users = None

        # list of players
        self.UAVs = []
        self.BSs = []
        self.Users = []

        # draw point numpy
        self.BS_pos_array = None
        self.UAV_pos_array = None
        self.User_pos_array = None

        # draw others
        self.circle_list = []
        self.line_list = []
        self.user_scatter_list = []
        self.uav_scatter_list = []

        # observation input
        self.UAVs_obs_np = []
        self.BSs_obs_np = []

        # step
        self.time_step = 0

        # draw the map
        self.show_3d = show_3d
        print("set show_3d")

        if self.show_3d is False:
            self.fig, self.ax2D = plt.subplots(figsize=(15, 15), frameon=True, facecolor='#666666')
        else:
            self.fig3D, self.ax3D = plt.subplots(figsize=(20, 15), frameon=True, facecolor='#666666')
            self.ax3D = Axes3D(self.fig3D)

    def build_UAV(self, x, y, z):
        self.UAVs.append(UAV(x, y, z))

    def build_BS(self, x, y, own):
        # add a BS
        self.BSs.append(BS(x, y, own))

        # add UAVs

        for i in range(own):
            # UAV start at in cycle with home radius
            self.build_UAV(x + random.randint(-1 * BS.home_radius - 1, BS.home_radius + 1),
                           y + random.randint(-1 * BS.home_radius - 1, BS.home_radius + 1),
                           BS.height)
            # self.build_UAV(x + random.randint(-999, 999),
            #                y + random.randint(-999, 999),
            #                BS.height*2)


    def step(self, rand):

        # init store tuple [obs_old, action, reward, obs_now]
        obs_old_uav = None  # shape: ( num_uav , UAV.dim_obs)
        obs_old_bs = None  # shape: ( num_bs , BS.dim_obs)

        action_uav = None  # shape: ( num_uav , UAV.dim_act)
        action_bs = None  # shape: ( num_bs , BS.dim_act)

        reward_uav = None  # shape: ( num_uav )
        reward_bs = None  # shape: ( num_bs )

        obs_now_uav = None  # shape: ( num_uav , UAV.dim_obs)
        obs_now_bs = None  # shape: ( num_bs , BS.dim_obs)

        # ---------------------------------------------------------
        #                   observation & get action
        # ----------------------------------------------------------
        # world observation is ready, save the old uav obs
        obs_old_uav = copy.deepcopy(self.UAVs_obs_np)
        # put UAVs obs into model ,and decode UAVs action
        action_uav = self.model.uav_act(self.UAVs_obs_np, self.time_step, rand=rand)
        self.decode_UAVs_action(action_uav)

        # 暂时用不到 BS 决策
        # supplement BS observation, save the old bs obs
        # self.get_BSs_observation()
        # obs_old_bs = self.BSs_obs_np.copy()

        # put BSs obs into model ,and decode BSs action
        # action_bs = bs_act(self.BSs_obs_np)
        # self.decode_BSs_action(action_bs)

        # ---------------------------------------------------------
        #                      move and communication
        # ----------------------------------------------------------

        self.world_move()  # the world move together
        self.world_communication()  # the world communicate together

        # 这里默认不图形化，只有test自己的step中会有选项
        # self.store_draw()  # store the point circle line

        # ---------------------------------------------------------
        #                      get reward
        # ----------------------------------------------------------

        reward_uav, reward_bs = self.get_world_reward()

        # ---------------------------------------------------------
        #                      update observation
        # ---------------------------------------------------------

        self.get_World_observation()
        # save obs now
        obs_now_bs = self.BSs_obs_np.copy()
        obs_now_uav = self.UAVs_obs_np.copy()

        # ---------------------------------------------------------
        #                      save experience
        # ---------------------------------------------------------

        self.model.add_experience([[obs_old_bs, action_bs, reward_bs, obs_now_bs],
                                   [obs_old_uav, action_uav, reward_uav, obs_now_uav]], self.time_step == (World.Env_time-1))

        # close the communication
        self.close_communication()

        self.time_step += 1

    def is_done(self):
        num_done = 0
        for uav in self.UAVs:
            if uav.energy <= 0:
                num_done += 1
        if num_done == self.num_UAVs:
            return True
        else:
            return False

    def get_world_reward(self):
        # 先临时凑一个奖励，假设无论如何奖励全为零
        reward_uav = np.zeros(self.num_UAVs)
        reward_bs = np.zeros(self.num_BSs)

        # 通信成功奖励
        for i in range(self.num_UAVs):
            if self.UAVs[i].success:
                user_id = self.UAVs[i].goal_user_id
                reward_uav[i] = self.Users[user_id].clean_aoi(self.time_step) + User.aoi_max / 100000
                # print(reward_uav[i])
                # reward_uav[i] *= self.discount ** self.UAV_period[i]
        # aoi爆炸集体惩罚 不写在这里，因为收集经验时集体算罚时

        return [reward_uav, reward_bs]

    def world_move(self):
        # all the entity move
        for i in range(self.num_Users):
            self.Users[i].move()
        for i in range(self.num_UAVs):
            self.UAVs[i].move()
        for i in range(self.num_BSs):
            self.BSs[i].move()

    def world_communication(self):

        # all line try to communication
        for uav in self.UAVs:
            if uav.access_User_id != -1 and uav.access_BS_id != -1 and uav.energy > 0:
                # 两边同时可连接
                if uav.trans >= 0:  # -1 表示不通信,0开始表示通信的第几个cycle

                    success = random.random() <= \
                              (upload_probability(self, uav.ID, uav.access_BS_id) *
                               download_probability(self, uav.ID, uav.access_User_id))

                    if success:
                        # 修改user
                        self.Users[uav.access_User_id].access = True
                        self.Users[uav.access_User_id].connect_uavID = uav.ID
                        uav.life += 1

                        # self.Users[uav.access_User_id].clean_aoi(self.time_step)

                    # 修改trans状态
                    uav.trans += 1

                    uav.success = success
                    #      print("------------- user ", uav.access_User_id, " to uav ", uav.ID, " trans ", uav.success,
                    #                          "------------")

                    # UAV energy loss
                    uav.energy_comm_step = upload_energy_cost(uav.upload_power) \
                                           + download_energy_cost(uav.download_power)

                    uav.energy_loss(uav.energy_comm_step)

    def close_communication(self):
        # 与world comm相对应，清楚通信成功标记位，在此之前getreward已经读取了这些
        for user in self.Users:
            user.success = False
            # user.connect_uavID = -1

        for uav in self.UAVs:
            # uav.access_BS_id = -1 BSid在观察的过程中由uav直接选择
            # uav.sub_channel = -1 暂时不考虑子信道
            # uav.access_User_id = -1
            if uav.success:
                uav.trans = -1
                uav.success = False

    def store_draw(self):
        # init clear
        self.circle_list.clear()
        self.user_scatter_list.clear()
        self.uav_scatter_list.clear()

        # 连接线打印延迟
        if random.random() < 0.9:
            self.line_list.clear()
        # write pos to numpy
        self.write_pos_array()

        # store user
        for user in self.Users:
            # color setting
            if user.aoi_weight < User.aoi_max * User.aoi_fact:
                color = 'lightskyblue'
            elif user.aoi_weight <= User.aoi_max * User.aoi_fact * 2:
                color = 'blue'
            elif user.aoi_weight <= User.aoi_max * User.aoi_fact * 3:
                color = 'mediumblue'
            elif user.aoi_weight <= User.aoi_max * User.aoi_fact * 5:
                color = 'darkblue'
            else:
                color = 'black'
            # size setting
            size = user.aoi_now * 2000
            if size > 600:
                size = 600
            self.user_scatter_list.append(plt.scatter(
                user.position[0],
                user.position[1],
                marker='+',
                color=color,
                s=size,
                label='User'
            ))

        # store uav
        for uav in self.UAVs:
            alpha = (uav.energy / UAV.max_UAV_energy) * 0.7 + 0.3
            if uav.energy == 0:
                color = 'black'
            else:
                color = 'red'
            self.uav_scatter_list.append(plt.scatter(
                uav.position[0],
                uav.position[1],
                marker='.',
                color=color,
                s=200,
                label='UAV',
                alpha=alpha
            ))

        # store bs
        for bs in self.BSs:
            self.circle_list.append(plt.Circle((bs.position[0],
                                                bs.position[1]),
                                               BS.home_radius,
                                               color='r',
                                               fill=True,
                                               alpha=0.1,
                                               zorder=10))
        # store line
        for uav in self.UAVs:
            if uav.trans >= 0:
                # print("uav.success ", uav.success)
                if uav.success:
                    # 成功画实线
                    # print("add success line")
                    self.line_list.append(plt.Line2D([uav.position[0], self.BSs[uav.access_BS_id].position[0]],
                                                     [uav.position[1], self.BSs[uav.access_BS_id].position[1]],
                                                     linewidth=2,
                                                     color='b'
                                                     ))
                    self.line_list.append(plt.Line2D([uav.position[0], self.Users[uav.access_User_id].position[0]],
                                                     [uav.position[1], self.Users[uav.access_User_id].position[1]],
                                                     linewidth=2,
                                                     color='b'
                                                     ))
                else:
                    # 失败画虚线
                    # print("add failure line")
                    self.line_list.append(plt.Line2D([uav.position[0], self.BSs[uav.access_BS_id].position[0]],
                                                     [uav.position[1], self.BSs[uav.access_BS_id].position[1]],
                                                     linestyle='--',
                                                     linewidth=2,
                                                     color='r'
                                                     ))
                    self.line_list.append(plt.Line2D([uav.position[0], self.Users[uav.access_User_id].position[0]],
                                                     [uav.position[1], self.Users[uav.access_User_id].position[1]],
                                                     linestyle='--',
                                                     linewidth=2,
                                                     color='r'
                                                     ))
            # else:
            #     # 飞行中画圆
            #     self.circle_list.append(plt.Circle((uav.position[0],
            #                                         uav.position[1]),
            #                                        UAV.User_view_size,
            #                                        color='black',
            #                                        fill=False,
            #                                        alpha=0.2,
            #                                        linestyle='--'))

    def get_World_observation(self):

        # 先BS，固定的5个信息，无关紧要
        self.get_BSs_observation()
        # 再用UAV，根据连接的bs拿到自己的所有信息
        self.get_UAVs_observation()

    def get_UAVs_observation(self):
        # get UAV observation 同时UAV做出绑定BS的决策
        self.UAVs_obs_np.clear()
        for uav in self.UAVs:
            uav_state, view_bs = uav.get_observation(self)
            uav.access_BS_id = uav_choice_bs(view_bs)  # 这里调用idiot算法自动帮uav绑定了bs（直接绑定能连接到的第一个）
            uav.access_BS_pos = self.BSs[uav.access_BS_id].position

        for uav in self.UAVs:
            # 制作uav邻居表
            neighbor_list = [[nei.observation, distance_between(uav, nei)] for nei in self.UAVs]

            neighbor_list = sorted(neighbor_list, key=lambda neighbor: neighbor[1])

            neighbor = []
            write = 0
            for i in range(self.num_UAVs):
                if neighbor_list[i][1] < UAV.UAV_view_size and neighbor_list[i][0][6] > 0 and i > 0:
                    neighbor.append(neighbor_list[i][0])
                    write += 1
            for i in range(self.num_UAVs - 1 - write):
                neighbor.append(np.zeros(UAV.dim_obs))
            # print(neighbor)
            # 制作user发现列表
            view_user_list = self.BSs_obs_np[uav.access_BS_id][1]
            for i in range(len(view_user_list)):
                # 视野外的设为-1
                if euclidean_distance(view_user_list[i][1:4], uav.position, 3) > UAV.User_view_size:
                    view_user_list[i] = np.ones(User.dim_obs) * -1

            # 在有无BS的基础上综合

            if uav.access_BS_id != -1:  # uav有连接bs的情况下才能拿到BS信息，否则为-1
                bs_obs = self.BSs_obs_np[uav.access_BS_id][0]
            else:
                bs_obs = np.ones(BS.dim_obs) * -1

            self.UAVs_obs_np.append([uav.observation,
                                     bs_obs,
                                     view_user_list,
                                     neighbor])

    def get_BSs_observation(self):
        # get BS observation
        self.BSs_obs_np.clear()
        for bs in self.BSs:
            self.BSs_obs_np.append(bs.get_observation(self))

    def decode_UAVs_action(self, action):
        for i in range(self.num_UAVs):
            self.UAVs[i].decode_action(action[i])

    def decode_BSs_action(self, action):
        for i in range(self.num_BSs):
            self.BSs[i].decode_action(action[i])

    def write_pos_array(self):

        # write the _pos_array[] numpy ,from each entity's position[]
        for i in range(self.num_BSs):
            self.BS_pos_array[i] = self.BSs[i].position
        for i in range(self.num_UAVs):
            self.UAV_pos_array[i] = self.UAVs[i].position
        for i in range(self.num_Users):
            self.User_pos_array[i] = self.Users[i].position

    def draw(self):

        if self.show_3d is True:
            self.draw_3d()
        else:
            self.draw_2d()

    def draw_2d(self):
        # self.ax2D.axis([0, 1000, 0, 1000])
        self.ax2D.clear()
        plt.title("2D Cellular", fontsize=24)
        self.ax2D.set_xlim((-self.world_size, self.world_size))
        self.ax2D.set_ylim((-self.world_size, self.world_size))
        # print BS
        self.ax2D.scatter(self.BS_pos_array[:, 0],
                          self.BS_pos_array[:, 1],
                          marker='^',
                          color='green',
                          s=300,
                          label='BS')
        for circle in self.circle_list:
            self.ax2D.add_artist(circle)
        for line in self.line_list:
            self.ax2D.add_line(line)

        for user in self.user_scatter_list:
            self.ax2D.add_artist(user)
        for uav in self.uav_scatter_list:
            self.ax2D.add_artist(uav)

        # # print UAV
        # self.ax2D.scatter(self.UAV_pos_array[:, 0],
        #                   self.UAV_pos_array[:, 1],
        #                   marker='.',
        #                   color='red',
        #                   s=200,
        #                   label='UAV')

        # # print User
        # self.ax2D.scatter(self.User_pos_array[:, 0],
        #                   self.User_pos_array[:, 1],
        #                   marker='+',
        #                   color='blue',
        #                   s=200,
        #                   label='User')

        self.ax2D.legend(loc='best', fontsize=20)
        plt.pause(0.1)

    def draw_3d(self):

        self.ax3D.clear()

        plt.title("3D Cellular", fontsize=24)
        self.ax3D.set_xlim((-self.world_size, self.world_size))
        self.ax3D.set_ylim((-self.world_size, self.world_size))
        self.ax3D.set_zlim((0, self.world_size / 2))
        self.ax3D.view_init(elev=15, azim=40)
        self.ax3D.scatter(self.BS_pos_array[:, 0],
                          self.BS_pos_array[:, 1],
                          self.BS_pos_array[:, 2],
                          marker='^',
                          color='green',
                          s=500,
                          label='BS')

        # print UAV
        self.ax3D.scatter(self.UAV_pos_array[:, 0],
                          self.UAV_pos_array[:, 1],
                          self.UAV_pos_array[:, 2],
                          marker='.',
                          color='red',
                          s=200,
                          label='UAV')
        # print User
        self.ax3D.scatter(self.User_pos_array[:, 0],
                          self.User_pos_array[:, 1],
                          marker='+',
                          color='blue',
                          s=200,
                          label='User')

        self.ax3D.legend(loc='best', fontsize=20)
        plt.pause(0.01)
