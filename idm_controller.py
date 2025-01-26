from params import *
import Environment

VEH_L, VEH_W = 4, 2
MIN_ACCELERATION = -4
MIN_STOP_GAP = 2

class IDM:
    def __init__(self, ego_info, other_info):
        self.ego_info = ego_info
        self.idm_parameter = [
            2,  # minimum spacing at a standstill (m)
            2,  # + np.random.uniform(-0.5, 0.5),  # desired time headway (s)
            2,  # max acceleration (m/s^2)
            2,  # + np.random.uniform(-1, 1),  # comfortable acceleration (m/s^2)
            10,  # (120 + np.random.uniform(-10, 10)) / 3.6  # desired speed (m/s)
            -4,  # max deceleration (m/s^2)
        ]
        self.other_info = other_info
        # calculate all variables for IDM
        self.distance = self.update_d()
        self.delta_v = self.update_v()

    def update_d(self):
        if str(self.other_info['entrance']) + str(self.other_info['exit']) in Environment.CONFLICT_RELATION[self.ego_info['entrance']][self.ego_info['exit']]:
            ego_dis2cp = self.ego_info['dis2des'] - Environment.CONFLICT_RELATION[self.ego_info['entrance']][self.ego_info['exit']][str(self.other_info['entrance']) + str(self.other_info['exit'])]
            other_dis2cp = self.other_info['dis2des'] - Environment.CONFLICT_RELATION[self.other_info['entrance']][self.other_info['exit']][str(self.ego_info['entrance']) + str(self.ego_info['exit'])]
            if ego_dis2cp > 0 and other_dis2cp > 0:
                distance = ego_dis2cp - other_dis2cp - 3 * VEH_L - MIN_STOP_GAP  # not car-following, give more space to yield (whether yield depends on llm_action not distance anymore)
            else:
                distance = None
        else:
            print('ego has no conflict with inter vehicle')
            distance = None
        return distance

    def update_v(self):
        delta_v = self.ego_info['v'] - self.other_info['v']
        return delta_v

    def cal_acceleration(self):
        if self.other_info is not None:
            akgs = self.ego_info['v'] * self.idm_parameter[1] + self.ego_info['v'] * self.delta_v / 2 / np.sqrt(self.idm_parameter[2] * self.idm_parameter[3])
            if akgs < 0:
                sss = self.idm_parameter[0]
            else:
                sss = self.idm_parameter[0] + akgs

            if self.distance > 0:
                acc = self.idm_parameter[2] * (1 - np.power((self.ego_info['v'] / self.idm_parameter[4]), 5) - np.power((sss / self.distance), 2))
                # if self.state[7] == 9:
                #     print('acc', acc, self.distance)
                if acc > self.idm_parameter[2]:
                    acc = self.idm_parameter[2]
                if acc < self.idm_parameter[5]:
                    acc = self.idm_parameter[5]
            else:
                acc = MIN_ACCELERATION
            return acc
        else:
            return MAX_ACCELERATION
