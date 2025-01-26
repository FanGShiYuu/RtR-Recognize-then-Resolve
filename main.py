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


Sim_times = 300  # Simulation runs for 300 frames
suffix = ''
# Simulation methods
M = ['MCTS-all', 'MCTS', 'single']
# M = ['MCTS-all', 'MCTS', 'single', 'FCFS', 'GAME', 'iDFST']

class Simulator:
    def __init__(self, case_id, method):
        random.seed(case_id)  # Set seed for reproducibility
        np.random.seed(case_id)  # Ensure reproducibility with numpy
        self.state_list = Environment.initialize_vehicle(6)  # Initialize vehicles in the environment
        self.fig, self.ax = plt.subplots(figsize=(8, 8))  # Create a figure for plotting
        self.case_id = case_id
        self.triggered = False  # Flag to track if cooperation is triggered
        self.final_order = None  # Final vehicle order for passage
        self.intention_remains = False  # Track if vehicle intentions remain unchanged
        self.acc_list = [0 for vehicle in self.state_list]  # List of accelerations
        self.svm_direction_list = {}  # Direction list for decision-making
        self.not_passed_vehicle_list = []  # Vehicles that haven't passed
        self.method = method  # Cooperation method ('MCTS', 'single', etc.)
        self.file_name, self.workbook = Environment.open_excel(case_id, suffix, self.method)  # Open Excel for recording data

    def run(self):
        """ Run the simulation and save it as a GIF """
        ani = FuncAnimation(self.fig, self.update, interval=10, frames=Sim_times, blit=False, repeat=False, save_count=Sim_times)
        video_dir = f'./video/' + strftime("%Y-%m-%d", gmtime()) + '/'  # Directory for saving video
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)  # Create the directory if not exists
        ani.save(video_dir + str(self.case_id) + '.gif', dpi=300)  # Save animation as GIF

    def update(self, frame):
        """ Update state of simulation for each frame """
        self.state_list = self.vehicle_info_update()  # Update vehicle states
        self.plot_figs()  # Plot the updated vehicle positions
        self.triggered = self.trigger_cooperation()  # Check if cooperation should be triggered
        workbook = Environment.write_data(self.workbook, self.state_list, frame)  # Write data to Excel
        workbook.save(self.file_name)  # Save workbook with new data

    def trigger_cooperation(self):
        """ Check if cooperation is needed based on method """
        if self.method == 'single':
            return False  # No cooperation for 'single' method
        elif self.method == 'MCTS-all':
            return True  # Always trigger cooperation for 'MCTS-all'
        else:
            trigger = False
            svm_direction_list = {}  # Track direction for potential cooperation
            for index, vehicle in enumerate(self.state_list):
                opponent_vehicle = Environment.find_opponent(vehicle, self.state_list)  # Find opponent vehicle
                if opponent_vehicle is not None:
                    opponent_index = next(i for i, v in enumerate(self.state_list) if v == opponent_vehicle)
                    # Calculate TTCP (Time-to-Collision Prediction) for both vehicles
                    current_ttcp_ego = Environment.cal_instance_ttcp(vehicle, self.acc_list[index])
                    current_ttcp_opp = Environment.cal_instance_ttcp(opponent_vehicle, self.acc_list[opponent_index])
                    next_vehicle, next_opponent_vehicle = Environment.kinematic_model(vehicle, self.acc_list[index]), Environment.kinematic_model(opponent_vehicle, self.acc_list[opponent_index])
                    next_ttcp_ego = Environment.cal_instance_ttcp(next_vehicle, self.acc_list[index])
                    next_ttcp_opp = Environment.cal_instance_ttcp(next_opponent_vehicle, self.acc_list[opponent_index])

                    # Calculate vehicle interaction directions
                    if vehicle['id'] < opponent_vehicle['id']:
                        svm_direction = Environment.calculate_heading(current_ttcp_ego, current_ttcp_opp, next_ttcp_ego, next_ttcp_opp)
                        svm_direction_list[str(vehicle['id']) + str(opponent_vehicle['id'])] = svm_direction
                    else:
                        svm_direction = Environment.calculate_heading(current_ttcp_opp, current_ttcp_ego, next_ttcp_opp, next_ttcp_ego)
                        svm_direction_list[str(opponent_vehicle['id']) + str(vehicle['id'])] = svm_direction
                    
                    # Trigger cooperation based on relative direction and predicted collision risks
                    if 90 >= svm_direction >= 0:  # First quadrant (both decelerating, inefficient)
                        trigger = True
                    elif 270 > svm_direction > 180 and min(next_ttcp_opp, next_ttcp_ego) < 3:  # Third quadrant (accelerating, potential collision)
                        trigger = True

            # Check if vehicle directions have changed significantly, triggering cooperation if so
            for dir in svm_direction_list.keys():
                if dir in self.svm_direction_list:
                    current_dir = self.svm_direction_list[dir]
                    next_dir = svm_direction_list[dir]
                    if abs(current_dir - next_dir) >= 90:  # Significant direction change
                        trigger = True
            self.svm_direction_list = svm_direction_list
            return trigger  # Return True if cooperation is triggered, False otherwise

    def MCTS_coop(self, not_passed_state_list):
        """ Perform MCTS to decide vehicle passage order """
        batch_vehicles, batch_relation = MCTS.generate_batch_vehicles(not_passed_state_list)
        root = MCTS.MCTSNode(batch_vehicles)
        best_order, best_score = MCTS.mcts(root, iterations=1000)  # Run MCTS
        full_best_order = MCTS.return_batch2full(not_passed_state_list, best_order, batch_relation)
        final_order = MCTS.organize_order(full_best_order)[1]  # Organize the final order
        return final_order

    def vehicle_info_update(self):
        """ Update vehicle states based on decision-making """
        acc_list = []  # List to store new acceleration values
        not_passed_state_list = []  # Vehicles that haven't passed
        for vehicle in self.state_list:
            if Environment.find_opponent(vehicle, self.state_list) is not None:
                not_passed_state_list.append(vehicle)
        
        if self.triggered:
            if self.method == 'MCTS':
                final_order = self.MCTS_coop(not_passed_state_list)  # Get final order from MCTS
            elif self.method == 'MCTS-all':
                final_order = self.MCTS_coop(not_passed_state_list)
            elif self.method == 'FCFS':
                final_order = cm.FCFS(not_passed_state_list)  # First-Come, First-Served method
            elif self.method == 'iDFST':
                final_order = cm.iDFST(not_passed_state_list)  # iDFST method
            elif self.method == 'GAME':
                final_order = cm.CoopGame(not_passed_state_list)  # Game-theory based cooperation
            else:
                final_order = None
                raise ValueError('no such coop method')  # Raise error if method is not valid
            
            self.final_order = final_order
            # Update accelerations based on final order
            for vehicle in self.state_list:
                if vehicle['id'] in final_order[0]:
                    acc_list.append(MAX_ACCELERATION)
                else:
                    acc_list.append(MIN_ACCELERATION)
        else:
            # If no cooperation, update accelerations based on IDM (Intelligent Driver Model)
            for vehicle in self.state_list:
                opponent_vehicle = Environment.find_opponent(vehicle, self.state_list, front=True)
                if opponent_vehicle is not None:
                    acc_list.append(IDM(vehicle, opponent_vehicle).cal_acceleration())
                else:
                    acc_list.append(MAX_ACCELERATION)

        next_state_list = copy.deepcopy(self.state_list)
        intention_remains = True
        for index, vehicle in enumerate(self.state_list):
            if vehicle['type'] != 'cav':
                opponent_vehicle = Environment.find_opponent(vehicle, self.state_list)
                if opponent_vehicle is not None:
                    bayesian_agent = Bayesian_Agent(vehicle, opponent_vehicle, action_type='discrete')
                    acc = bayesian_agent.update_state()  # Update acceleration based on Bayesian agent
                else:
                    acc = MAX_ACCELERATION
                vehicle, intention_remains = Environment.adjust_intention(vehicle, acc_list[index], acc, intention_remains)
                acc_list[index] = acc  # Update true HDV action
            else:
                acc = acc_list[index]
            
            if Environment.conflict_point_occupied(vehicle, self.state_list)[0]:
                acc = MIN_ACCELERATION  # Adjust acceleration if conflict point is occupied
            acc = Environment.adjust_acc(vehicle, acc)  # Adjust acceleration for stopping at the line
            next_state_list[index] = Environment.kinematic_model(vehicle, acc)  # Update vehicle state based on new acceleration
            self.acc_list = acc_list

        self.not_passed_vehicle_list = not_passed_state_list  # Update list of not-passed vehicles
        return next_state_list  # Return updated vehicle states

    def plot_figs(self):
        """ Plot the positions and status of all vehicles """
        self.ax.cla()  # Clear previous plot
        self.ax.axis('scaled')  # Set axis to be scaled
        Environment.scenario_outfit(self.ax)  # Plot environment scenario
        for vehicle in self.state_list:
            # Set color based on vehicle type
            vehicle_color = 'purple'
            if vehicle['type'] == 'agg':
                vehicle_color = 'orange'
            elif vehicle['type'] == 'con':
                vehicle_color = 'green'
            elif vehicle['type'] == 'nor':
                vehicle_color = 'blue'
            
            # Plot vehicle position and add ID, velocity, and intention labels
            self.ax.scatter(vehicle['x'], vehicle['y'], color=vehicle_color)
            plt.plot(Environment.ALL_REF_LINE[vehicle['entrance']][vehicle['exit']][:, 0], 
                     Environment.ALL_REF_LINE[vehicle['entrance']][vehicle['exit']][:, 1], color='grey')
            plt.text(vehicle['x'] + 2, vehicle['y'] + 2, round(vehicle['v'], 2))
            plt.text(vehicle['x'], vehicle['y'], vehicle['id'])
            if vehicle['type'] != 'cav':
                intention = vehicle['aggressive_intention']
                plt.text(vehicle['x'] + 14, vehicle['y'] + 2, f'[{intention}]')
        plt.text(50, 50, str(self.triggered))  # Show if cooperation was triggered
        plt.text(40, 60, str(self.final_order))  # Show final order of vehicles


# Run the simulation for each method
for method in M:
    for id in range(100):
        Simulator(id, method).run()  # Run the simulation for each case ID
