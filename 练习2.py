# 姓名 ：李星宇
# 开发时间 : 2022/7/2 18:04

import numpy as np
from matplotlib import pyplot as plt
"""利用PSO算法解决0-1背包问题：有N件物品和一个容量为v的背包，其中第i件物品的体积是c_i,
价值是w_i。求解将哪些物品放入背包里面可使物品的体积总和不超过背包的容量，且价值总和最大利
用PSO算法解决0-1背包问题：有N件物品和一个容量为v的背包，其中第i件物品的体积是c_i,价值是
w_i。求解将哪些物品放入背包里面可使物品的体积总和不超过背包的容量，且价值总和最大"""


#################初始化##################
N=100                  	#群体粒子个数
D=10                   	#粒子维数
T=200                  	#最大迭代次数
c1=1.5                 	#学习因子1
c2=1.5                 	#学习因子2
Wmax=0.8               	#惯性权重最大值
Wmin=0.4               	#惯性权重最小值
Vmax=10                	#速度最大值
Vmin=-10               	#速度最小值
V = 300                             	#背包容量
C = [95,75,23,73,50,22,6,57,89,98]  	#物品体积
W = [89,59,19,43,100,72,44,16,7,64] 	#物品价值
afa = 2                             	#惩罚函数系数

fit_T = np.zeros(T)
cur_weight = np.zeros(N)
t = np.zeros(T)

def f(x):
    for i in range(N):
        cur_weight[i] = np.dot(x[i], W)
        TotalSize = np.dot(x[i], C)
        if TotalSize > V:
            cur_weight[i] = cur_weight[i] - afa * (TotalSize - V)
    return cur_weight


def sigmoid(x):
    return 1/(1+np.exp(-x))


def update_x(x, v):
    for j in range(D):
        randnum = np.random.rand()
        if randnum < sigmoid(v[j]):
            x[j] = 1
        else:
            x[j] = 0


##########初始化种群个体（限定位置和速度）#########
x=np.random.randint(0, 2, [N, D])
v=np.random.rand(N,D) * (Vmax-Vmin)+Vmin
#############初始化个体最优位置和最优值##########
pbest = x
p_fit = f(x)
fit_argmax=np.argmax(f(x))
gbest = x[fit_argmax,:]
gbest_fit = p_fit[fit_argmax]

# ########按照公式依次迭代直到满足精度或者迭代次数######
for i in range(T):
    for j in range(N):
        w = Wmax - (Wmax - Wmax) * i / T
        v[j, :] = w * v[j, :]+c1 * np.random.rand() * (pbest[j, :]- x[j, :])+c2*np.random.rand()*(gbest-x[j, :])
        update_x(x[j, :], v[j, :])

        for k in range(D):
            if (v[j, k] > Vmax) or (v[j, k] < Vmin):
                v[j, k]=np.random.rand() * (Vmax-Vmax)+Vmax

    # 更新个体最优位置
    p_fit2 = f(x)
    flag = p_fit2 > p_fit
    for j in range(N):
        if flag[j]==True:
            pbest[j, :] = x[j, :]
    # 更新全局最优位置

    fit_argmax = np.argmax(f(pbest))
    t[i] = i
    if f(pbest)[fit_argmax]>gbest_fit:
        fit_T[i] = f(pbest)[fit_argmax]
        gbest = pbest[fit_argmax, :]
        gbest_fit = f(pbest)[fit_argmax]
    else:
        fit_T[i] = gbest_fit

plt.scatter(t, fit_T)
plt.plot(t, fit_T)
plt.xlabel("spoch")
plt.ylabel("f(x)")
plt.show()


