import matplotlib.pyplot as plt
import numpy as np

def RGB_to_Hex(rgb):
    RGB = rgb.split(',')  # 将RGB格式划分开来
    color = '#'
    for i in RGB:
        num = int(i)
        # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color

def get_intention_line(ttcp_other, d_self, v_self, params):
    w1 = params[0]
    w2 = params[1]
    w3 = params[2]
    b = params[3]
    print(-w1/w2, -w3/w2, - b / w2)
    return -w1/w2 * ttcp_other - w3/w2 * (d_self - v_self * ttcp_other) / (ttcp_other ** 2) - b / w2

plt.figure()
ttcp_range = np.arange(1, 8, 0.1)
params_xxjh_lt = [-2.38, 2.41, 2.1, 5.86]
params_xxjh_gs = [-1.04, 2.07, 3.56, -8.88]
params = [-5.59928566e-01,  6.79648158e-01, -3.17712236e-04, 0.9568910811326777]  # [-1.38, 1.75, 0.22, 2.54]
# in pairs
params_av_lt = [-0.22172418,  0.46639683,  0.78988992, -1.135543396187552]  # av为左转车 此时对象为直行的直行车
params_hv_gs = [-0.42154014,  0.27299729,  0.78619943, 0.47534300404599333]

params_av_gs = [-0.48974414,  0.77623773, -0.00386497, 1.1687365746800147]
params_hv_lt = [-0.54283835,  0.50765787,  0.01317797, -2.2686705619664713]

params_av_sum = [-5.59928566e-01,  6.79648158e-01, -3.17712236e-04, 0.9568910811326777]
params_hv_sum = [-0.63628522,  0.53144675,  0.0027785, -0.9744559694469043]

params_hv1 = [-0.2875904,   0.42340031, -0.00098639, 1.025799569705723]
params_hv2 = [0.34524272, -0.26063108, -0.00493738, 1.1445414279115473]
f1 = get_intention_line(ttcp_range, 30, 10, params_av_lt)
f2 = get_intention_line(ttcp_range, 20, 5.6, params_hv_gs)
plt.plot(ttcp_range, f1, color='orange')
plt.plot(f2, ttcp_range)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.show()

