# 姓名 ：李星宇
# 开发时间 : 2022/7/2 18:04

import numpy as np
from matplotlib import pyplot as plt
"""利用PSO算法求函数f(x,y)=3cos(xy)+x+y^2的最小值"""

#############粒子群算法求函数极值#############
#################初始化##################
N=100                  	#群体粒子个数
D=2                    	#粒子维数
T=200                  	#最大迭代次数
c1=1.5                 	#学习因子1
c2=1.5                 	#学习因子2
Wmax=0.8               	#惯性权重最大值
Wmin=0.4               	#惯性权重最小值
Xmax=4                 	#位置最大值
Xmin=-4                	#位置最小值
Vmax=1                 	#速度最大值
Vmin=-1                	#速度最小值

fit = np.zeros(N)
fit_T = np.zeros(T)
t = np.zeros(T)
def f(x):
    for label, k in enumerate(x):
        i, j = tuple(k)
        fit[label] = 3 * np.cos(i*j)+i+j**2
    return fit


##########初始化种群个体（限定位置和速度）#########
x=np.random.rand(N,D) * (Xmax-Xmin)+Xmin
v=np.random.rand(N,D) * (Vmax-Vmin)+Vmin
#############初始化个体最优位置和最优值##########
pbest = x
p_fit = f(x)
fit_argmin=np.argmin(f(x))
gbest = x[fit_argmin,:]
gbest_fit = p_fit[fit_argmin]

# ########按照公式依次迭代直到满足精度或者迭代次数######
for i in range(T):
    for j in range(N):
        w = Wmax - (Wmax - Wmin) * i / T
        v[j, :] = w * v[j, :]+c1 * np.random.rand() * (pbest[j, :]- x[j, :])+c2*np.random.rand()*(gbest-x[j, :])
        x[j, :] = x[j, :] + v[j, :]

        for k in range(D):
            if (v[j, k] > Vmax) or (v[j, k] < Vmin):
                v[j, k]=np.random.rand() * (Vmax-Vmin)+Vmin
            if (x[j, k]>Xmax)  or  (x[j, k]< Xmin):
                x[j, k]=np.random.rand() * (Xmax-Xmin)+Xmin
    # 更新个体最优位置
    p_fit2 = f(x)
    flag = p_fit2 < p_fit
    for j in range(N):
        if flag[j]==True:
            pbest[j, :] = x[j, :]
    # 更新全局最优位置

    fit_argmin = np.argmin(f(pbest))
    t[i] = i
    if f(pbest)[fit_argmin]<gbest_fit:
        fit_T[i] = f(pbest)[fit_argmin]
        gbest = pbest[fit_argmin, :]
        gbest_fit = f(pbest)[fit_argmin]
    else:
        fit_T[i] = gbest_fit

plt.scatter(t, fit_T)
plt.plot(t, fit_T)
plt.xlabel("spoch")
plt.ylabel("f(x)")
plt.show()