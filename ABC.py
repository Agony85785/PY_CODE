import numpy as np
import math
from matplotlib import pyplot as plt
from numpy.matlib import repmat


def fitness_fuc(x):
    for i in range(N):
        cur_fitness[i] = 418.9829 * D + np.dot(x[i, :], np.sin(abs(x[i, :])**0.5))
    return cur_fitness

def fitness_fuc2(x):
    cur = 418.9829 * D + np.dot(x, np.sin(abs(x)**0.5))
    return cur

N = 60
D = 30
FES = 0
FESMAX = D * 10000  # 迭代次数
lu = np.array([-500*np.ones(D), 500*np.ones(D)])  # 取值范围
fit = np.zeros(N)
cur_fitness = np.zeros(N)
fit_result = np.zeros(int((FESMAX-FES)/10000))
epoch = np.array(range(0, FESMAX-FES, 10000))
x_result = np.zeros(D)



x = repmat(lu[0, :], N, 1) + np.random.rand(N, D) * repmat(lu[1,:] - lu[0,:], N, 1)
fitness = fitness_fuc(x)
kk = np.min(fitness)
limit = 100
number = np.zeros(N)
k = 0
FES = 0
t = 0
v = np.zeros(D)
while FES < FESMAX:
    t = t + 1
    for i in range(N):
        r = math.floor(np.random.rand() * N)
        while r == i:
            r = math.floor(np.random.rand() * N)
        v[:] = x[i,:]
        j = math.floor(np.random.rand() * D)
        v[j] = x[i, j] + (-1 + 2 * np.random.rand()) * (x[i, j] - x[r, j])  # 雇佣蜂搜索
        if v[j] > lu[1, j]:
            v[j] = np.max([2 * lu[1, j] - v[j], lu[0, j]])
        if v[j] < lu[0, j]:
            v[j] = np.min([2 * lu[0, j] - v[j], lu[1, j]])
        fnew = fitness_fuc2(v)
        if fnew < fitness[i]:
            x[i,:] = v
            fitness[i] = fnew
            number[i] = 0
        else:
            number[i] = number[i] + 1

    for i in range(N):  #  招募概率
        if fitness[i] > 0:
            fit[i] = 1 / (1 + fitness[i])
        else:
            fit[i] = 1 + abs(fitness[i])
    prob = 0.9 * (fit / np.max(fit)) + 0.1


    i = 0
    g = 0
    while g <= N:
        if np.random.rand() < prob[i]:
            g = g + 1
            r = math.floor(np.random.rand() * N)
            while r == i:
                r = math.floor(np.random.rand() * N)
            j = math.floor(np.random.rand() * D)
            v[:] = x[i,:]
            v[j] = x[i, j] + (-1 + 2 * np.random.rand()) * (x[i, j] - x[r, j])  # 观察蜂搜索
            if v[j] > lu[1, j]:
                v[j] = np.max([2 * lu[1, j] - v[j], lu[0, j]])
            if v[j] < lu[0, j]:
                v[j] = np.min([2 * lu[0, j] - v[j], lu[1, j]])
            fnew = fitness_fuc2(v)
            if fnew < fitness[i]:
                x[i, :] = v
                fitness[i] = fnew
                number[i] = 0
            else:
                number[i] = number[i] + 1
        i = i + 1
        if i == N :
            i = 0
    index = np.argmin(fitness)
    for i in range(N):
        if i != index and number[i] >= limit:
            x[i,:] = lu[0,:] + np.random.rand(D) * (lu[1, :] - lu[0,:])  # 侦查蜂搜索
            fitness[i] = fitness_fuc2(x[i,:])
            number[i] = 0


    for i in np.arange(2 * N):
        if FES % 10000 == 0:
            fit_result[k] = np.min(fitness)
            x_result = x[np.argmin(fitness), :]
            kk = fit_result[k]
            print('Algorithm: ', FES, 'Best: ', kk)
            k += 1
        FES += 1

print("最优点为:", tuple(x_result))
print("最小适应度值：", fit_result[-1])
plt.plot(epoch, fit_result)
plt.xlabel("spoch")
plt.ylabel("fitness")
plt.show()




