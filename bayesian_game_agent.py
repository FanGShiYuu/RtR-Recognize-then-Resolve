import copy
import Environment
from params import *

class Bayesian_Agent:
    """Solve 1v1 Nash game for now"""
    
    def __init__(self, hdv_info, cav_info, action_type):
        # Initialize ego (human-driven) and inter (autonomous) vehicle info
        self.ego_info = hdv_info
        self.ego_state = [hdv_info['x'], hdv_info['y'], hdv_info['v'], hdv_info['heading'], hdv_info['dis2des'], hdv_info['entrance'], hdv_info['exit']]
        self.ego_central_vertices = Environment.ALL_REF_LINE[self.ego_state[5]][self.ego_state[6]]
        self.ego_aggressiveness = hdv_info['type']

        self.inter_info = cav_info
        self.inter_state = [cav_info['x'], cav_info['y'], cav_info['v'], cav_info['heading'], cav_info['dis2des'], cav_info['entrance'], cav_info['exit']]
        self.inter_central_vertices = Environment.ALL_REF_LINE[self.inter_state[5]][self.inter_state[6]]
        self.action_type = action_type
        
        # Aggressiveness distribution: ['agg', 'nor', 'con']
        self.aggressiveness_distribution = [0.14, 0.41, 0.45] 
        self.max_speed = self.get_max_speed()

    def get_max_speed(self):
        """Get the max speed for ego vehicle based on aggressiveness"""
        if self.ego_aggressiveness == 'agg':
            return Target_speed[0]
        elif self.ego_aggressiveness == 'nor':
            return Target_speed[1]
        elif self.ego_aggressiveness == 'con':
            return Target_speed[2]
        else:
            self.ego_aggressiveness = 'nor'  # Default to 'nor'
            print('Input ego_info as cav, should not!')
            return Target_speed[1]

    def kinematic_model(self, state, action, temp):
        """
        Update vehicle state based on kinematic model.
        State update: v' = v + a*Dt, position update, and heading update.
        """
        state_now = copy.deepcopy(state)
        state_now[2] += Action_space[action, 0] * Dt  # Update velocity
        if state[0] == self.ego_state[0] and not temp:  # If ego vehicle
            if state_now[2] > self.max_speed:
                state_now[2] = self.max_speed
            if state_now[2] < 0:
                state_now[2] = 0
        state_now[4] -= state_now[2] * Dt  # Update distance to destination
        state_now[0], state_now[1] = Environment.update_pos_from_dis2des_to_Cartesian({'entrance': state_now[5], 'exit': state_now[6], 'dis2des': state_now[4]})
        state_now[3] = Environment.calculate_heading(state[0], state[1], state_now[0], state_now[1])
        return state_now

    def reward_weight(self, aggressiveness):
        """Get reward weight based on vehicle aggressiveness"""
        if aggressiveness == 'agg':
            return Weight_hv[0]
        elif aggressiveness == 'nor':
            return Weight_hv[1]
        else:
            return Weight_hv[2]

    def get_ttc_thr(self, aggressiveness):
        """Get TTC threshold based on aggressiveness"""
        if aggressiveness == 'agg':
            return 2
        elif aggressiveness == 'nor':
            return 6
        else:
            return 7

    def update_state(self):
        """Update ego vehicle state considering interaction with another vehicle"""
        if not Environment.if_passed_conflict_point(self.ego_info, self.inter_info):
            r = np.zeros(Action_length)
            for inter, inter_aggressiveness in enumerate(['agg', 'nor', 'con']):
                # Find Nash equilibrium solution
                nash_equilibrium_solution = self.nash_equilibrium(inter_aggressiveness)
                inter_pure_strategy = nash_equilibrium_solution[0][1] if nash_equilibrium_solution else 0

                # Calculate reward for each possible action combination
                for ego_action in range(Action_length):
                    r[ego_action] += self.aggressiveness_distribution[inter] * self.reward(ego_action, inter_pure_strategy, inter_vehicle_aggressiveness=inter_aggressiveness)[0]
            
            # Select action based on Bayesian equilibrium
            bayesian_pure_strategy = np.argmax(r)
            acc = Action_space[bayesian_pure_strategy, 0]
        else:
            acc = max(Acceleration_list)
        return acc

    def nash_equilibrium(self, inter_vehicle_aggressiveness):
        """Compute Nash equilibrium for interaction"""
        nash_matrix = np.zeros((Action_length, Action_length))
        ego_best_response, inter_best_response = self.get_best_response(inter_vehicle_aggressiveness)
        
        for act in range(Action_length):
            nash_matrix[act, inter_best_response[act]] += 1
            nash_matrix[ego_best_response[act], act] += 1
        
        nash = list(zip(*np.where(nash_matrix == 2)))  # Find Nash equilibrium coordinates
        return nash

    def get_best_response(self, inter_vehicle_aggressiveness):
        """Compute the best response for both vehicles"""
        ego_reward_matrix = np.zeros((Action_length, Action_length))
        inter_reward_matrix = np.zeros((Action_length, Action_length))
        
        # Calculate reward matrices for all action pairs
        for act1 in range(Action_length):  
            for act2 in range(Action_length):  
                ego_reward, inter_reward = self.reward(act1=act1, act2=act2, inter_vehicle_aggressiveness=inter_vehicle_aggressiveness)
                ego_reward_matrix[act1, act2] = ego_reward
                inter_reward_matrix[act1, act2] = inter_reward

        # Find best responses for both vehicles
        inter_best_response = [np.argmax(inter_reward_matrix[act, :]) for act in range(Action_length)]
        ego_best_response = [np.argmax(ego_reward_matrix[:, act]) for act in range(Action_length)]
        
        return ego_best_response, inter_best_response

    def reward(self, act1, act2, inter_vehicle_aggressiveness):
        """
        Calculate reward for both ego and inter vehicles based on their actions.
        Reward components include distance to central vertex, speed, destination, and collision penalty.
        """
        ego_state = self.kinematic_model(state=self.ego_state, action=act1, temp=True)
        inter_state = self.kinematic_model(state=self.inter_state, action=act2, temp=True)

        # 1st: Yaw penalty
        ego_dis2cv = np.amin(np.linalg.norm(self.ego_central_vertices - ego_state[0:2], axis=1))
        inter_dis2cv = np.amin(np.linalg.norm(self.inter_central_vertices - inter_state[0:2], axis=1))
        ego_reward1 = - max(0.1, ego_dis2cv) * 10 if Environment.if_right_turning(self.ego_state[5], self.ego_state[6]) != 'rt' else - max(0, ego_dis2cv) * 20
        inter_reward1 = - max(0.1, inter_dis2cv) * 10 if Environment.if_right_turning(self.inter_state[5], self.inter_state[6]) != 'rt' else - max(0, inter_dis2cv) * 20

        # 2nd: Speed reward
        ego_reward2 = ego_state[2]
        inter_reward2 = inter_state[2]

        # 3rd: Destination-driven reward
        ego_destination = self.ego_central_vertices[-1]
        inter_destination = self.inter_central_vertices[-1]
        ego_reward3 = - ((ego_state[0] - ego_destination[0])**2 + (ego_state[1] - ego_destination[1])**2)**0.5
        inter_reward3 = - ((inter_state[0] - inter_destination[0])**2 + (inter_state[1] - inter_destination[1])**2)**0.5

        # 4th: Collision penalty
        dis = ((ego_state[0] - inter_state[0])**2 + (ego_state[1] - inter_state[1])**2)**0.5
        ego_ttc = dis / (ego_state[2] * self.get_ttc_thr(self.ego_aggressiveness))
        inter_ttc = dis / (inter_state[2] * self.get_ttc_thr(inter_vehicle_aggressiveness))
        ego_reward4 = (- 1/ego_ttc)
        inter_reward4 = (- 1/inter_ttc)

        # Aggregate rewards
        ego_reward = np.array([ego_reward1, ego_reward2, ego_reward3, ego_reward4])
        inter_reward = np.array([inter_reward1, inter_reward2, inter_reward3, inter_reward4])

        ego_reward_weight = self.reward_weight(self.ego_aggressiveness)
        inter_reward_weight = self.reward_weight(inter_vehicle_aggressiveness)
        
        return np.dot(ego_reward, ego_reward_weight), np.dot(inter_reward, inter_reward_weight)  # Return both ego and inter rewards

    def state_without_inter_vehicle(self):
        """Calculate the optimal state for the ego vehicle without interaction"""
        reward_without_iv = []
        for act in range(Action_length):
            self_state = self.kinematic_model(state=self.ego_state, action=act, temp=True)
            dis2cv = np.amin(np.linalg.norm(self.ego_central_vertices - self_state[0:2], axis=1))
            reward1 = - max(0.1, dis2cv) * 20 if Environment.if_right_turning(self.ego_state[5], self.ego_state[6]) == 'rt' else - max(0.1, dis2cv) * 10
            reward2 = self_state[2]  # Speed reward
            destination = self.ego_central_vertices[-1]
            reward3 = - abs(self_state[0] - destination[0]) - abs(self_state[1] - destination[1]) * 2
            reward4 = 0  # No interaction, no collision penalty
            
            reward = np.array([reward1, reward2, reward3, reward4])
            reward_weight = self.reward_weight(self.ego_aggressiveness)
            reward_without_iv.append(np.dot(reward, reward_weight) + reward2)  # Add speed reward for optimality

        max_reward_action = np.argmax(reward_without_iv)
        self_state = self.kinematic_model(state=self.ego_state, action=max_reward_action, temp=False)
        return self_state  # Return state for optimal action without interaction
