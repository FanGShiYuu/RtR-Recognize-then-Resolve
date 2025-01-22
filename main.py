import copy
import os
from matplotlib import animation
import MCTS
from matplotlib.animation import FuncAnimation
import Environment
import random
from time import gmtime, strftime
import time
import matplotlib.pyplot as plt
from params import *
from bayesian_game_agent import Bayesian_Agent
from idm_controller import IDM
import comparison_method as cm


Sim_times = 300
suffix = ''
M = ['MCTS-all', 'MCTS', 'single', 'FCFS', 'GAME', 'iDFST']
# M = ['MCTS']
plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\房世玉\AppData\Local\Programs\Python\Python310\Lib\site-packages\ffmpeg_3.4.2\bin\x64\ffmpeg.exe'

class Simulator:
    def __init__(self, case_id, method):
        random.seed(case_id)  # 为本次 Simulator 设置随机种子
        np.random.seed(case_id)  # 如果使用 numpy 的随机性
        self.state_list = Environment.initialize_vehicle(6)
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.case_id = case_id
        self.triggered = False
        self.final_order = None
        self.intention_remains = False
        self.acc_list = [0 for vehicle in self.state_list]
        self.svm_direction_list = {}
        self.not_passed_vehicle_list = []
        self.method = method
        self.file_name, self.workbook = Environment.open_excel(case_id, suffix, self.method)
        self.cal_time = 0
        self.tri_time = 0


    def run(self):
        " ---- option 1: show animation ---- "
        # ani = FuncAnimation(self.fig, self.update, interval=10, frames=Sim_times, blit=False, repeat=False, save_count=Sim_times)
        # plt.show()

        " ---- option 2: save as gif ---- "
        # ani = FuncAnimation(self.fig, self.update, interval=10, frames=Sim_times, blit=False, repeat=False, save_count=Sim_times)
        # video_dir = f'./video/' + strftime("%Y-%m-%d", gmtime()) + '/'
        # if not os.path.exists(video_dir):
        #     os.makedirs(video_dir)
        # ani.save(video_dir + str(self.case_id) + '.gif', dpi=300)

        " ---- option 3: save as mp4 video ---- "
        ani = FuncAnimation(self.fig, self.update, interval=10, frames=Sim_times, blit=False, repeat=False, save_count=Sim_times)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, codec="h264", bitrate=-1, metadata=dict(dpi=600, artist='Me'))
        video_dir = './video/' + strftime("%Y-%m-%d", gmtime()) + suffix + '-' + self.method + '/'
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        ani.save(video_dir + str(self.case_id) + '.mp4', writer=writer)
        CAL_LIST.append(self.cal_time)
        TRI_LIST.append(self.tri_time)
        plt.close()

    def update(self, frame):
        self.state_list = self.vehicle_info_update()
        self.plot_figs()
        time2 = time.time()
        self.triggered = self.trigger_cooperation()
        tri_t = (time.time() - time2)
        self.tri_time += tri_t
        # print(frame, self.triggered, self.cal_time, self.tri_time)
        workbook = Environment.write_data(self.workbook, self.state_list, frame)
        workbook.save(self.file_name)

    def trigger_cooperation(self):
        if self.method == 'single':
            return False
        elif self.method == 'MCTS-all':
            return True
        else:
            trigger = False
            svm_direction_list = {}
            for index, vehicle in enumerate(self.state_list):
                # if vehicle['type'] != 'cav':
                opponent_vehicle = Environment.find_opponent(vehicle, self.state_list)
                if opponent_vehicle is not None:
                    opponent_index = next(i for i, v in enumerate(self.state_list) if v == opponent_vehicle)
                    current_ttcp_ego = Environment.cal_instance_ttcp(vehicle, self.acc_list[index])
                    current_ttcp_opp = Environment.cal_instance_ttcp(opponent_vehicle, self.acc_list[opponent_index])
                    next_vehicle, next_opponent_vehicle = Environment.kinematic_model(vehicle, self.acc_list[index]), Environment.kinematic_model(opponent_vehicle, self.acc_list[opponent_index])
                    next_ttcp_ego  = Environment.cal_instance_ttcp(next_vehicle, self.acc_list[index])
                    next_ttcp_opp = Environment.cal_instance_ttcp(next_opponent_vehicle, self.acc_list[opponent_index])
                    if vehicle['id'] < opponent_vehicle['id']:
                        svm_direction = Environment.calculate_heading(current_ttcp_ego , current_ttcp_opp, next_ttcp_ego, next_ttcp_opp)
                        svm_direction_list[str(vehicle['id']) + str(opponent_vehicle['id'])] = svm_direction
                    else:
                        svm_direction = Environment.calculate_heading(current_ttcp_opp, current_ttcp_ego, next_ttcp_opp, next_ttcp_ego)
                        svm_direction_list[str(opponent_vehicle['id']) + str(vehicle['id'])] = svm_direction
                    # print(svm_direction, current_ttcp_ego, current_ttcp_opp, next_ttcp_ego, next_ttcp_opp, vehicle['dis2des'], vehicle['v'], opponent_vehicle['dis2des'], opponent_vehicle['v'])
                    if 90 >= svm_direction >= 0:  # in first quadrant, both vehicle decelerate, cause inefficiency, trigger coop
                        trigger = True
                    elif 270 > svm_direction > 180 and min(next_ttcp_opp, next_ttcp_ego) < 3:  # in third quadrant, both vehicle accelerate and ttcp is small, might cause collision, trigger coop
                        trigger = True

            for dir in svm_direction_list.keys():
                if dir in self.svm_direction_list:
                    current_dir = self.svm_direction_list[dir]
                    next_dir = svm_direction_list[dir]
                    if abs(current_dir - next_dir) >= 90:
                        trigger = True
            self.svm_direction_list = svm_direction_list
            # print(trigger, self.svm_direction_list)
            return trigger
        # if trigger:
        #     return True
        # else:
        #     return self.triggered

    def MCTS_coop(self, not_passed_state_list):
        batch_vehicles, batch_relation = MCTS.generate_batch_vehicles(not_passed_state_list)
        root = MCTS.MCTSNode(batch_vehicles)
        # 执行 MCTS
        best_order, best_score = MCTS.mcts(root, iterations=1000)
        # 输出最佳通行顺序和总分
        full_best_order = MCTS.return_batch2full(not_passed_state_list, best_order, batch_relation)
        final_order = MCTS.organize_order(full_best_order)[1]
        # print('MCTS通行顺序', final_order)
        return final_order

    def vehicle_info_update(self):
        time1 = time.time()
        acc_list = []  # consider all vehicle as CAV, then adjust it if is HDV
        not_passed_state_list = []
        for vehicle in self.state_list:
            if Environment.find_opponent(vehicle, self.state_list) is not None:
                not_passed_state_list.append(vehicle)
        if self.triggered:
            if self.method == 'MCTS':
                final_order = self.MCTS_coop(not_passed_state_list)
            elif self.method == 'MCTS-all':
                final_order = self.MCTS_coop(not_passed_state_list)
            elif self.method == 'FCFS':
                final_order = cm.FCFS(not_passed_state_list)
            elif self.method == 'iDFST':
                final_order = cm.iDFST(not_passed_state_list)
            elif self.method == 'GAME':
                final_order = cm.CoopGame(not_passed_state_list)
            else:
                final_order = None
                raise ValueError('no such coop method')
            self.final_order = final_order
            for vehicle in self.state_list:
                if vehicle['id'] in final_order[0]:
                    acc_list.append(MAX_ACCELERATION)
                else:
                    if vehicle['id'] not in [id for order in final_order for id in order]:
                        acc_list.append(MAX_ACCELERATION)

                    else:
                        acc_list.append(MIN_ACCELERATION)
        else:
            # print('++++', len(not_passed_state_list) == len(self.not_passed_vehicle_list), self.final_order is not None, self.intention_remains)
            if len(not_passed_state_list) == len(self.not_passed_vehicle_list) and self.final_order is not None and self.intention_remains:
                acc_list = self.acc_list
            else:
                for vehicle in self.state_list:
                    opponent_vehicle = Environment.find_opponent(vehicle, self.state_list, front=True)
                    if opponent_vehicle is not None:
                        acc_list.append(IDM(vehicle, opponent_vehicle).cal_acceleration())
                    else:
                        acc_list.append(MAX_ACCELERATION)

        cal_t = (time.time() - time1)
        self.cal_time += cal_t

        next_state_list = copy.deepcopy(self.state_list)
        intention_remains = True
        for index, vehicle in enumerate(self.state_list):
            if vehicle['type'] != 'cav':
                opponent_vehicle = Environment.find_opponent(vehicle, self.state_list)
                if opponent_vehicle is not None:
                    bayesian_agent = Bayesian_Agent(vehicle, opponent_vehicle, action_type='discrete')
                    acc = bayesian_agent.update_state()
                else:
                    acc = MAX_ACCELERATION
                vehicle, intention_remains = Environment.adjust_intention(vehicle, acc_list[index], acc, intention_remains)
                acc_list[index] = acc  # update true HDV action
            else:
                acc = acc_list[index]
            if Environment.conflict_point_occupied(vehicle, self.state_list)[0]:  # 此时将intention改为激进 优先级高于通过速度修改intention
                acc = MIN_ACCELERATION
            acc = Environment.adjust_acc(vehicle, acc)  # adjust vehicle acc based on if they will stop before stop line and if they have no conflict
            next_state_list[index] = Environment.kinematic_model(vehicle, acc)
            self.acc_list = acc_list
        self.not_passed_vehicle_list = not_passed_state_list
        self.intention_remains = intention_remains
        # print(acc_list)
        return next_state_list

    def plot_figs(self):
        self.ax.cla()
        self.ax.axis('scaled')
        Environment.scenario_outfit(self.ax)
        for vehicle in self.state_list:
            vehicle_color = 'purple'
            if vehicle['type'] == 'agg':
                vehicle_color = 'orange'
            elif vehicle['type'] == 'con':
                vehicle_color = 'green'
            elif vehicle['type'] == 'nor':
                vehicle_color = 'blue'
            self.ax.scatter(vehicle['x'], vehicle['y'], color=vehicle_color)
            plt.plot(Environment.ALL_REF_LINE[vehicle['entrance']][vehicle['exit']][:, 0], Environment.ALL_REF_LINE[vehicle['entrance']][vehicle['exit']][:, 1], color='grey')
            plt.text(vehicle['x'] + 2, vehicle['y'] + 2, round(vehicle['v'], 2))
            plt.text(vehicle['x'], vehicle['y'], vehicle['id'])
            if vehicle['type'] != 'cav':
                intention = vehicle['aggressive_intention']
                plt.text(vehicle['x'] + 14, vehicle['y'] + 2, f'[{intention}]')
        plt.text(50, 50, str(self.triggered))
        plt.text(40, 60, str(self.final_order))


for method in M:
    CAL_LIST = []
    TRI_LIST = []
    for id in range(100):
        Simulator(id, method).run()
    print(f'+++{method}: sum cal{sum(CAL_LIST)}, sum tri{sum(TRI_LIST)}')
    print(f'+++{method}: mean cal{sum(CAL_LIST)/len(CAL_LIST)}, mean tri{sum(TRI_LIST)/len(TRI_LIST)}')

