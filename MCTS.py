import copy
import random
import time
import matplotlib.pyplot as plt
from Environment import *

def generate_batch_vehicles(vehicles):
    batch_vehicles = copy.deepcopy(vehicles)
    batch_relation = {}
    for vehicle in vehicles:
        batch_relation[vehicle['id']] = []
    for index_i, vehicle_i in enumerate(vehicles):
        front_vehicle = vehicle_i
        for vehicle_j in vehicles:
            if vehicle_i['entrance'] == vehicle_j['entrance']:
                if vehicle_i['exit'] == vehicle_j['exit'] and (vehicle_i['dis2des'] - vehicle_j['dis2des']) > (vehicle_i['dis2des'] - front_vehicle['dis2des']):
                    front_vehicle = vehicle_j
                elif vehicle_i['exit'] != vehicle_j['exit'] and (vehicle_i['dis2des'] - vehicle_j['dis2des']) > (vehicle_i['dis2des'] - front_vehicle['dis2des']) and vehicle_j['dis2des'] > 100:
                    front_vehicle = vehicle_j
        if front_vehicle['id'] != vehicle_i['id'] and front_vehicle['exit'] == vehicle_i['exit']:
            batch_relation[front_vehicle['id']].append(vehicle_i['id'])
            batch_vehicles = [vehicle for vehicle in batch_vehicles if vehicle['id'] != vehicle_i['id']]  # 删除vehicle_i
    for vehicle in batch_vehicles:
        if len(batch_relation[vehicle['id']]) != 0:
            vehicle['multiple_vehicle'] += len(batch_relation[vehicle['id']])
    # print('batch融合结果为：', batch_relation)
    return batch_vehicles, batch_relation

def return_batch2full(vehicles, passed_order, batch_relation):
    full_passed_order = copy.deepcopy(passed_order)
    for vehicle in passed_order:
        if len(batch_relation[vehicle['id']]) != 0:
            index = full_passed_order.index(vehicle)
            rear_vehicle = [vehicle]
            for id in batch_relation[vehicle['id']]:
                r_v = [veh for veh in vehicles if veh['id'] == id][0]
                rear_vehicle.append(r_v)
            full_passed_order[index:index+1] = rear_vehicle
    return full_passed_order

def organize_order(passed_vehicles):
    passed_order = [[]]
    for p_v in passed_vehicles:
        for order in range(len(passed_order)-1, -1, -1):  # 从大通行权到小通行权
            have_conflict = False
            for order_p_v in passed_order[order]:
                if p_v['entrance'] + p_v['exit'] in CONFLICT_RELATION[order_p_v['entrance']][order_p_v['exit']]:
                    have_conflict = True
            if not have_conflict:  # 不存在冲突
                if order == 0:  # 如果已经搜索到最先通行顺序 直接添加到最优层
                    passed_order[order].append(p_v)
                else:
                    continue  # 继续向前搜索
            else:  # 存在冲突
                if order == len(passed_order) - 1:  # 如果order已经是最后一个了，另起一个新的通行次序
                    passed_order.append([p_v])
                    break
                else:  # 否则 加到后一个order
                    passed_order[order+1].append(p_v)
                    break
    passed_order_id = []
    for order in passed_order:
        order_id = []
        for p_v in order:
            order_id.append(p_v['id'])
        passed_order_id.append(order_id)
    # print('已经通过的车辆编号:', passed_order_id)
    return passed_order, passed_order_id

def accumulate_delay(passed_order):
    delay = []
    pass_time = []
    former_vehicle = []
    for order in range(len(passed_order)):
        order_delay_time = []
        order_pass_time = []
        order_former_vehicle = []
        for order_p_v in passed_order[order]:
            max_former_pass_time = 0
            max_former_vehicle = None
            for veh, p_v in enumerate([j for i in passed_order[:order] for j in i]):
                if p_v['entrance'] + p_v['exit'] in CONFLICT_RELATION[order_p_v['entrance']][order_p_v['exit']]:
                    if max_former_pass_time < [j for i in pass_time[:order] for j in i][veh]:
                        max_former_pass_time = [j for i in pass_time[:order] for j in i][veh]
                        max_former_vehicle = [j for i in passed_order[:order] for j in i][veh]['id']
            order_pass_time.append((order_p_v['dis2des']-80)/SPEED_LIMIT + max_former_pass_time)
            order_delay_time.append(max_former_pass_time)
            order_former_vehicle.append(max_former_vehicle)
        pass_time.append(order_pass_time)
        delay.append(order_delay_time)
        former_vehicle.append(order_former_vehicle)
    # print('车辆离开交叉口的时间', pass_time)
    return delay, former_vehicle

def heuristic_rule(passed_vehicles, score):
    passed_order, passed_order_id = organize_order(passed_vehicles)
    for index_i, vehicle_i in enumerate(passed_vehicles):
        for index_j, vehicle_j in enumerate(passed_vehicles):
            # 1. reachable cost 可达惩罚：避免处于同一进口道的车辆后车通行顺序先于前车
            if vehicle_i['entrance'] == vehicle_j['entrance'] and vehicle_i['dis2des'] > 100 and vehicle_j['dis2des'] > 100:
                if (vehicle_i['dis2des'] - vehicle_j['dis2des']) * (index_i - index_j) < 0:
                    score -= 100
        # 2. interaction cost 交互惩罚：当HV为抢行意图时 需要将其通行权放在第一顺位
        if vehicle_i['aggressive_intention'] == 1 and vehicle_i not in passed_order[0]:
            score -= 10000
        if vehicle_i['aggressive_intention'] == -1 and vehicle_i in passed_order[0]:
            score -= 100
    return score

class MCTSNode:
    def __init__(self, vehicles, passed_vehicles=None, parent=None):
        self.vehicles = vehicles  # 剩余车辆
        self.passed_vehicles = passed_vehicles if passed_vehicles is not None else []  # 当前通行顺序
        self.total_score = 0  # 当前总分
        self.visits = 0  # 访问次数
        self.children = []  # 子节点
        self.parent = parent  # 父节点

    def is_terminal(self):
        return len(self.vehicles) == 0  # 没有车辆待通行

    def is_fully_expanded(self):
        return len(self.children) == len(self.vehicles)

    def expand(self):
        # 扩展节点，生成子节点
        for vehicle in self.vehicles:
            remaining_vehicles = self.vehicles.copy()
            remaining_vehicles.remove(vehicle)
            new_order = self.passed_vehicles + [vehicle]
            child_node = MCTSNode(remaining_vehicles, new_order, parent=self)
            self.children.append(child_node)

    def get_best_child(self, exploration_weight=1.41):
        # 使用 UCB1 算法选择最佳子节点
        return max(self.children,
                   key=lambda child: (child.total_score / (child.visits + 1)) + exploration_weight * math.sqrt(
                       math.log(self.visits + 1) / (child.visits + 1)))


def mcts(root, iterations):
    for _ in range(iterations):
        node = root
        # 1&2. Search and Expand
        while not node.is_terminal():
            if not node.is_fully_expanded():
                node.expand()
            else:
                node = node.get_best_child()
        # 3. Simulation: 在确定完整的通行顺序后再计算分数
        passed_vehicles = node.passed_vehicles + node.vehicles  # 将剩余的车辆加入当前通行顺序
        final_order, final_order_id = organize_order(passed_vehicles)  # 确定最终通行顺序
        final_delay = accumulate_delay(final_order)[0]  # 计算最终延误
        score = -sum(sum(d) for d in final_delay)  # 累计延误总和作为分数
        score = heuristic_rule(passed_vehicles, score)
        # print('第', _, '次迭代, order和延误为：', final_order_id, score)
        # 4. Backpropagation: 回溯更新节点信息
        while node is not None:
            node.visits += 1
            node.total_score += score
            node = node.parent  # 回溯到父节点
    # 从根节点开始遍历，构建完整的通行顺序
    node = root
    best_order = []
    while not node.is_terminal():
        if not node.is_fully_expanded():
            node.expand()
        else:
            node = node.get_best_child(exploration_weight=0)  # 不使用探索权重选择最终解
        # print(len(node.children), [veh['id'] for veh in node.vehicles], node.is_terminal())
        best_order.extend(node.passed_vehicles[len(best_order):])  # 获取新车辆加入通行顺序
    return best_order, node.total_score

if __name__ == "__main__":
    vehicles = initialize_vehicle(12)
    'a demo to check if results stable and optimal'
    # vehicles = [{'v': 7.3283142908774535, 'heading': 4.71238898038469, 'dis2des': 163.412486837088, 'entrance': 'n3', 'exit': 'e1', 'x': -2.5, 'y': 64.97999999999894, 'id': 0}, {'v': 6.897856516653328, 'heading': 0.0, 'dis2des': 160.9899999999398, 'entrance': 'w1', 'exit': 'e1', 'x': -60.98999999998004, 'y': -2.5, 'id': 1}, {'v': 6.134892047908905, 'heading': 3.141592653589793, 'dis2des': 162.18169159380608, 'entrance': 'e2', 'exit': 'n2', 'x': 79.97999999999861, 'y': 7.5, 'id': 2}, {'v': 6.959807386720023, 'heading': 3.141592653589793, 'dis2des': 160.98999999990008, 'entrance': 'e3', 'exit': 'w3', 'x': 60.97999999999902, 'y': 2.5, 'id': 3}, {'v': 8.899019559013528, 'heading': 0.0, 'dis2des': 166.19849001185344, 'entrance': 'w2', 'exit': 's2', 'x': -83.98999999999181, 'y': -7.5, 'id': 4}, {'v': 6.1543485249670855, 'heading': 1.5707963267948966, 'dis2des': 188.9899999999398, 'entrance': 's1', 'exit': 'n1', 'x': 2.5, 'y': -88.98999999999437, 'id': 5}]
    'a demo to check batch strategy and heuristic_rule'
    vehicles = [{'v': 8.096,  'dis2des': 162.394, 'entrance': 'e3', 'exit': 's3', 'x': 63.979, 'y': 2.5, 'id': 0, 'aggressive_intention': 1, 'multiple_vehicle': 1},
                {'v': 6.423,  'dis2des': 162.989, 'entrance': 's1', 'exit': 'n1', 'x': 2.5, 'y': -62.989, 'id': 1, 'aggressive_intention': 0, 'multiple_vehicle': 1},
                {'v': 8.713,  'dis2des': 145.190, 'entrance': 'n2', 'exit': 'w2', 'x': -7.5, 'y': 62.979, 'id': 2, 'aggressive_intention': 0, 'multiple_vehicle': 1},
                {'v': 7.650,  'dis2des': 190.989, 'entrance': 's2', 'exit': 'n2', 'x': 7.5, 'y': -90.989, 'id': 3, 'aggressive_intention': 0, 'multiple_vehicle': 1},
                {'v': 8.341,  'dis2des': 170.181, 'entrance': 'e2', 'exit': 'n2', 'x': 87.979, 'y': 7.5, 'id': 4, 'aggressive_intention': 0, 'multiple_vehicle': 1},
                {'v': 6.630,  'dis2des': 168.989, 'entrance': 'n3', 'exit': 's3', 'x': -2.5, 'y': 68.979, 'id': 5, 'aggressive_intention': 0, 'multiple_vehicle': 1},
                {'v': 7.765,  'dis2des': 178.393, 'entrance': 'w1', 'exit': 'n1', 'x': -79.989, 'y': -2.5, 'id': 6, 'aggressive_intention': 0, 'multiple_vehicle': 1},
                {'v': 6.874,  'dis2des': 199.989, 'entrance': 'w2', 'exit': 'e2', 'x': -100.0, 'y': -7.5, 'id': 7, 'aggressive_intention': 0, 'multiple_vehicle': 1},
                {'v': 6.345,  'dis2des': 152.192, 'entrance': 's2', 'exit': 'e2', 'x': 7.5, 'y': -69.989, 'id': 8, 'aggressive_intention': 0, 'multiple_vehicle': 1},
                {'v': 6.097,  'dis2des': 166.989, 'entrance': 'w1', 'exit': 'e1', 'x': -66.989, 'y': -2.5, 'id': 9, 'aggressive_intention': 0, 'multiple_vehicle': 1},
                {'v': 7.951,  'dis2des': 161.181, 'entrance': 'e2', 'exit': 'n2', 'x': 78.979, 'y': 7.5, 'id': 10, 'aggressive_intention': 0, 'multiple_vehicle': 1},
                {'v': 6.010,  'dis2des': 190.989, 'entrance': 'w2', 'exit': 'e2', 'x': -90.989, 'y': -7.5, 'id': 11, 'aggressive_intention': 0, 'multiple_vehicle': 1},
                {'v': 7.951,  'dis2des': 180.181, 'entrance': 'e2', 'exit': 'n2', 'x': 97.979, 'y': 7.5, 'id': 12, 'aggressive_intention': 0, 'multiple_vehicle': 1}]
    print(vehicles)

    for i in range(10):
        print('-------------------')
        time1 = time.time()
        # 初始化根节点
        batch_vehicles, batch_relation = generate_batch_vehicles(vehicles)
        root = MCTSNode(batch_vehicles)
        # 执行 MCTS
        best_order, best_score = mcts(root, iterations=1000)
        # 输出最佳通行顺序和总分
        print("批处理最佳通行顺序:", [v["id"] for v in best_order], organize_order(best_order)[1])
        full_best_order = return_batch2full(vehicles, best_order, batch_relation)
        print("解压后完整最佳通行顺序:", [v["id"] for v in full_best_order], organize_order(full_best_order)[1])
        best_order_delay = accumulate_delay(organize_order(best_order)[0])[0]
        print("最佳score(越小越好):", sum(sum(d) for d in best_order_delay))
        print('搜索通行顺序计算时间:', time.time() - time1)

    plt.figure()
    for veh in vehicles:
        cv = ALL_REF_LINE[veh['entrance']][veh['exit']]
        plt.scatter(veh['x'], veh['y'])
        plt.plot(cv[:,0], cv[:,1], label=veh['id'])
        plt.text(veh['x']+1, veh['y']+1, veh['id'])
    plt.legend()
    plt.show()
