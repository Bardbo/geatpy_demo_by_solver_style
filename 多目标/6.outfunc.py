# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea

"""
这是一个离散决策变量的最小化目标的双目标优化问题。
min f1 = -25 * (x1 - 2)**2 - (x2 - 2)**2 - (x3 - 1)**2 - (x4 - 4)**2 - (x5 - 1)**2
min f2 = (x1 - 1)**2 + (x2 - 1)**2 + (x3 - 1)**2 + (x4 - 1)**2 + (x5 - 1)**2
s.t.
x1 + x2 >= 2
x1 + x2 <= 6
x1 - x2 >= -2
x1 - 3*x2 <= 2
4 - (x3 - 3)**2 - x4 >= 0
(x5 - 3)**2 + x4 - 4 >= 0
x1,x2,x3,x4,x5 ∈ {0,1,2,3,4,5,6,7,8,9,10}

描述:
    该案例是moea_demo1的另一个版本，展示了如何定义aimFunc()而不是evalVars()来计算目标函数和违反约束程度值。
    同时展示如何定义outFunc()，用于让算法在每一次进化时调用该outFunc()函数。
"""

def aimFunc(pop):  # 目标函数
    Vars = pop.Phen  # 得到决策变量矩阵
    x1 = Vars[:, [0]]
    x2 = Vars[:, [1]]
    x3 = Vars[:, [2]]
    x4 = Vars[:, [3]]
    x5 = Vars[:, [4]]
    f1 = -25 * (x1 - 2) ** 2 - (x2 - 2) ** 2 - (x3 - 1) ** 2 - (x4 - 4) ** 2 - (x5 - 1) ** 2
    f2 = (x1 - 1) ** 2 + (x2 - 1) ** 2 + (x3 - 1) ** 2 + (x4 - 1) ** 2 + (x5 - 1) ** 2
    # 利用可行性法则处理约束条件
    pop.CV = np.hstack([2 - x1 - x2,
                        x1 + x2 - 6,
                        -2 - x1 + x2,
                        x1 - 3 * x2 - 2,
                        (x3 - 3) ** 2 + x4 - 4,
                        4 - (x5 - 3) ** 2 - x4])
    pop.ObjV = np.hstack([f1, f2])  # 把求得的目标函数值赋值给种群pop的ObjV

Dim = 5 # 初始化Dim（决策变量维数）
M=2
problem = ea.Problem(name = 'MyProblem',  # 初始化name（函数名称，可以随意设置）
                        M=2,
                        Dim = Dim,  # 初始化Dim（决策变量维数）
                        maxormins = [1] * M,  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
                        varTypes = [1] * Dim,  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
                        lb = [0] * Dim,  # 决策变量下界
                        ub = [10] * Dim,  # 决策变量上界
                        lbin = [1] * Dim,  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
                        ubin = [1] * Dim,  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
                        aimFunc=aimFunc)


# 定义outFunc()函数
def outFunc(alg, pop):  # alg 和 pop为outFunc的固定输入参数，分别为算法对象和每次迭代的种群对象。
    print('第 %d 代' % alg.currentGen)
# 构建算法
algorithm = ea.moea_NSGA2_templet(problem,
                                    ea.Population(Encoding='RI', NIND=50),
                                    MAXGEN=200,  # 最大进化代数
                                    logTras=1,  # 表示每隔多少代记录一次日志信息，0表示不记录。
                                    outFunc=outFunc)
# 求解
res = ea.optimize(algorithm, verbose=False, drawing=1, outputMsg=True, drawLog=True, saveFlag=False)
print(res)