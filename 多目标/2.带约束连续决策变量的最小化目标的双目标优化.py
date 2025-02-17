# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea

"""
min f1 = X**2
min f2 = (X - 2)**2
s.t.
X**2 - 2.5 * X + 1.5 >= 0
10 <= Xi <= 10, (i = 1,2,3,...)

该案例展示了一个带约束连续决策变量的最小化目标的双目标优化问题的求解。
"""


# 构建问题
def evalVars(Vars):  # 目标函数
    f1 = Vars ** 2
    f2 = (Vars - 2) ** 2
    #        # 利用罚函数法处理约束条件
    #        exIdx = np.where(Vars**2 - 2.5 * Vars + 1.5 < 0)[0] # 获取不满足约束条件的个体在种群中的下标
    #        f1[exIdx] = f1[exIdx] + np.max(f1) - np.min(f1)
    #        f2[exIdx] = f2[exIdx] + np.max(f2) - np.min(f2)
    # 利用可行性法则处理约束条件
    CV = -Vars ** 2 + 2.5 * Vars - 1.5
    ObjV = np.hstack([f1, f2])
    return ObjV, CV

Dim = 1 # 初始化Dim（决策变量维数）
M=2
problem = ea.Problem(name = 'MyProblem',  # 初始化name（函数名称，可以随意设置）
                        Dim = Dim,  # 初始化Dim（决策变量维数）
                        M=M,
                        maxormins = [1] * M,  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
                        varTypes = [0] * Dim,  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
                        lb = [-10] * Dim,  # 决策变量下界
                        ub = [10] * Dim,  # 决策变量上界
                        lbin = [1] * Dim,  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
                        ubin = [1] * Dim,  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
                        evalVars=evalVars)
# 构建算法
algorithm = ea.moea_NSGA2_templet(problem,
                                    ea.Population(Encoding='RI', NIND=50),
                                    MAXGEN=200,  # 最大进化代数
                                    logTras=0)  # 表示每隔多少代记录一次日志信息，0表示不记录。
# 求解
res = ea.optimize(algorithm, verbose=False, drawing=1, outputMsg=True, drawLog=False, saveFlag=False)
print(res)
