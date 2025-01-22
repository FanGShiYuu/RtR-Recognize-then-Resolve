import math
import numpy as np

POSSIBLE_ENTRANCE = ['n2', 'n3', 'e2', 'e3', 's2', 's1', 'w2', 'w1']  # possible entrance
ENTRANCE_EXIT_RELATION = {'n2': ['s2', 'w2'], 'n3': ['s3', 'e1'],
                          'e2': ['w2', 'n2'], 'e3': ['w3', 's3'],
                          'w2': ['s2', 'e2'], 'w1': ['n1', 'e1'],
                          's2': ['n2', 'e2'], 's1': ['n1', 'w3'],}
VEH_L = 4
MAX_ACCELERATION = 2
MIN_ACCELERATION = -4
SPEED_LIMIT = 10
Dt = 0.1
Action_space = np.array([[0, 0], [2, 0], [-2, 0], [-4, 0]])
Action_length = len(Action_space)
Target_speed = [10, 10, 8]
Weight_hv = [[1.56, 0, 8.33, 3.69], [1.72, 0, 8.2, 5.7], [2.1, 0, 7.79, 8.44]]  # agg nor con
Acceleration_list = [0, 2, -2, -4, 0, 0]
Acceleration_bds = [min(Acceleration_list), max(Acceleration_list)]
Pi = math.pi


