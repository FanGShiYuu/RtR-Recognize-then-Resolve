import copy
import Environment
from params import *

class Bayesian_Agent:
    """solve 1v1 nash game for now"""
    def __init__(self, hdv_info, cav_info, action_type):
        self.ego_info = hdv_info
        self.ego_state = [hdv_info['x'], hdv_info['y'], hdv_info['v'], hdv_info['heading'], hdv_info['dis2des'], hdv_info['entrance'], hdv_info['exit']]
        self.ego_central_vertices = Environment.ALL_REF_LINE[self.ego_state[5]][self.ego_state[6]]
        self.ego_aggressiveness = hdv_info['type']

        self.inter_info = cav_info
        self.inter_state = [cav_info['x'], cav_info['y'], cav_info['v'], cav_info['heading'], cav_info['dis2des'], cav_info['entrance'], cav_info['exit']]
        self.inter_central_vertices = Environment.ALL_REF_LINE[self.inter_state[5]][self.inter_state[6]]
        self.action_type = action_type
        # self.recede = self.if_recede()
        self.aggressiveness_distribution = [0.14, 0.41, 0.45]  # ['agg', 'nor', 'con']
        self.max_speed = self.get_max_speed()

    def get_max_speed(self):
        if self.ego_aggressiveness == 'agg':
            return Target_speed[0]
        elif self.ego_aggressiveness == 'nor':
            return Target_speed[1]
        elif self.ego_aggressiveness == 'con':
            return Target_speed[2]
        else:
            self.ego_aggressiveness = 'nor'
            print('input ego_info as cav, should not!')
            return Target_speed[0]

    def kinematic_model(self, state, action, temp):
        # 我们在两个时刻需要用动力模型
        # 1.计算收益时(此时如果限制ego的最高速度会导致计算的自身收益与他人对我收益的预估不符 无法形成共识的收益矩阵) 所以不限最高速
        # 2.更新状态(此时需要限速)
        state_now = copy.deepcopy(state)
        state_now[2] += Action_space[action, 0] * Dt  # v'=v+a*Dt
        if state[0] == self.ego_state[0] and not temp:  # ego自己知道自己的目标速度 这里看现在算的state 是不是ego
            if state_now[2] > self.max_speed:
                state_now[2] = self.max_speed
            if state_now[2] < 0:
                state_now[2] = 0
        state_now[4] -= state_now[2] * Dt
        state_now[0], state_now[1] = Environment.update_pos_from_dis2des_to_Cartesian({'entrance': state_now[5], 'exit': state_now[6], 'dis2des': state_now[4]})
        state_now[3] = Environment.calculate_heading(state[0], state[1], state_now[0], state_now[1])
        return state_now

    def reward_weight(self, aggressiveness):
        if aggressiveness == 'agg':
            reward_weight = Weight_hv[0]
        elif aggressiveness == 'nor':
            reward_weight = Weight_hv[1]
        else:
            reward_weight = Weight_hv[2]
        return reward_weight

    def get_ttc_thr(self, aggressiveness):
        if aggressiveness == 'agg':
            ttc_thr = 2
        elif aggressiveness == 'nor':
            ttc_thr = 6
        else:
            ttc_thr = 7
        return ttc_thr

    def update_state(self):  # donnot forget for discrete action u need agrmax action; but for continuous, its minimize so reward should add '-' to find agrmax
        if not Environment.if_passed_conflict_point(self.ego_info, self.inter_info):
            r = np.zeros(Action_length)
            for inter, inter_aggressiveness in enumerate(['agg', 'nor', 'con']):
                nash_equilibrium_solution = self.nash_equilibrium(inter_aggressiveness)
                if nash_equilibrium_solution == []:
                    inter_pure_strategy = 0
                else:
                    inter_pure_strategy = nash_equilibrium_solution[0][1]

                for ego_action in range(Action_length):
                    r[ego_action] += self.aggressiveness_distribution[inter] * \
                                     self.reward(ego_action, inter_pure_strategy, inter_vehicle_aggressiveness=inter_aggressiveness)[0]  # 0表示自身收益
            bayesian_pure_strategy = np.argmax(r)
            # print('bayesian game下ego均衡解为', bayesian_pure_strategy)
            acc = Action_space[bayesian_pure_strategy, 0]
        else:
            acc = max(Acceleration_list)
        return acc

    # below is for discrete action
    def nash_equilibrium(self, inter_vehicle_aggressiveness):
        nash_matrix = np.zeros((Action_length, Action_length))  # 长度等于Action_space
        ego_best_response, inter_best_response = self.get_best_response(inter_vehicle_aggressiveness=inter_vehicle_aggressiveness)
        for act in range(Action_length):
            nash_matrix[act, inter_best_response[act]] += 1  # 给矩阵时 每一行给列相应的best response
            nash_matrix[ego_best_response[act], act] += 1
        # print(nash_matrix)
        _ = [i.tolist() for i in np.where(nash_matrix == 2)]
        nash = list(zip(*_))  # 纳什均衡的坐标集合
        return nash

    def get_best_response(self, inter_vehicle_aggressiveness):
        """输出收益最大的动作索引"""
        # 分别得到ego inter在所有动作下的收益阵
        ego_reward_matrix = np.zeros((Action_length, Action_length))  # 行是ego动作 列是inter动作
        inter_reward_matrix = np.zeros((Action_length, Action_length))
        for act1 in range(Action_length):  # ego动作
            for act2 in range(Action_length):  # inter动作
                ego_reward, inter_reward = self.reward(act1=act1, act2=act2, inter_vehicle_aggressiveness=inter_vehicle_aggressiveness)
                ego_reward_matrix[act1, act2] = ego_reward
                inter_reward_matrix[act1, act2] = inter_reward

        # 计算best response
        inter_best_response = []
        ego_best_response = []
        for act in range(Action_length):  # 找ego动作下 inter的best response
            inter_best_response.append(np.argmax(inter_reward_matrix[act, :]))  # 对于每一行找最大收益的索引
            ego_best_response.append(np.argmax(ego_reward_matrix[:, act]))  # 对于每一列找最大值索引
        return ego_best_response, inter_best_response

    def reward(self, act1, act2, inter_vehicle_aggressiveness):  # act1 本车动作 act2 他车动作 算得双方的收益
        # 位置更新
        ego_state = self.kinematic_model(state=self.ego_state, action=act1, temp=True)
        inter_state = self.kinematic_model(state=self.inter_state, action=act2, temp=True)

        # 1st 偏航收益（-）
        ego_dis2cv = np.amin(np.linalg.norm(self.ego_central_vertices - ego_state[0:2], axis=1))
        inter_dis2cv = np.amin(np.linalg.norm(self.inter_central_vertices - inter_state[0:2], axis=1))
        if Environment.if_right_turning(self.ego_state[5], self.ego_state[6]) == 'rt':
            ego_reward1 = - max(0, ego_dis2cv) * 20
        else:
            ego_reward1 = - max(0.1, ego_dis2cv) * 10
        if Environment.if_right_turning(self.inter_state[5], self.inter_state[6]) == 'rt':
            inter_reward1 = - max(0, inter_dis2cv) * 20
        else:
            inter_reward1 = - max(0.1, inter_dis2cv) * 10

        # 2st 速度收益（+）
        ego_reward2 = ego_state[2]
        inter_reward2 = inter_state[2]

        ego_destination = self.ego_central_vertices[-1]
        inter_destination = self.inter_central_vertices[-1]
        # 3rd 目的地驱动收益（-）           ###########################################note that mixed bayesian game have not change reward3 by -——>+ abs——>**2 **0.5s
        ego_reward3 = - ((ego_state[0] - ego_destination[0])**2 + (ego_state[1] - ego_destination[1])**2)**0.5
        inter_reward3 = - ((inter_state[0] - inter_destination[0])**2 + (inter_state[1] - inter_destination[1])**2)**0.5

        # 4th 碰撞惩罚(-)：提前减速 这里两个车是相同的
        dis = ((ego_state[0] - inter_state[0])**2 + (ego_state[1] - inter_state[1])**2)**0.5
        # dis = abs(ego_state[4] -  + inter_state[4])
        ego_ttc_thr = self.get_ttc_thr(self.ego_aggressiveness)
        inter_ttc_thr = self.get_ttc_thr(inter_vehicle_aggressiveness)
        ego_ttc = (dis / ego_state[2]) / ego_ttc_thr
        inter_ttc = (dis / inter_state[2]) / inter_ttc_thr
        ego_reward4 = (- 1/ego_ttc)
        inter_reward4 = (- 1/inter_ttc)

        # add up reward
        ego_reward = np.array([ego_reward1, ego_reward2, ego_reward3, ego_reward4])
        inter_reward = np.array([inter_reward1, inter_reward2, inter_reward3, inter_reward4])
        ego_reward_weight = self.reward_weight(self.ego_aggressiveness)
        inter_reward_weight = self.reward_weight(inter_vehicle_aggressiveness)
        return np.dot(ego_reward, ego_reward_weight), np.dot(inter_reward, inter_reward_weight)  # 返回ego inter的收益

    def state_without_inter_vehicle(self):  # 在没有交互对象时用的动作 以及相应的state
        reward_without_iv = []
        for act in range(Action_length):
            self_state = self.kinematic_model(state=self.ego_state, action=act, temp=True)
            central_vertices = self.ego_central_vertices  # 考虑lt在act1下 gs的各动作收益 所以计算的是gs的收益
            dis2cv = np.amin(np.linalg.norm(central_vertices - self_state[0:2], axis=1))
            if Environment.if_right_turning(self.ego_state[5], self.ego_state[6]) == 'rt':
                reward1 = - max(0.1, dis2cv) * 20
            else:
                reward1 = - max(0.1, dis2cv) * 10
            reward2 = self_state[2]
            destination = self.ego_central_vertices[-1]
            reward3 = - abs(self_state[0] - destination[0]) - abs(self_state[1] - destination[1]) * 2
            reward4 = 0  # 因为已经没有交互对象了
            reward = np.array([reward1, reward2, reward3, reward4])
            reward_weight = self.reward_weight(self.ego_aggressiveness)
            reward_without_iv.append(np.dot(reward, reward_weight) + reward2)
            # print(act, 'reward', reward, np.dot(reward, reward_weight))
        max_reward_action = np.argmax(reward_without_iv)
        self_state = self.kinematic_model(state=self.ego_state, action=max_reward_action, temp=False)
        return self_state  # 与update不同 只返回自身状态位置

