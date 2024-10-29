# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea

"""
    该目标函数存在多个欺骗性很强的局部最优点。
    max f = 4*x1 + 2*x2 + x3
    s.t.
    2*x1 + x2 - 1 <= 0
    x1 + 2*x3 - 2 <= 0
    x1 + x2 + x3 - 1 == 0
    0 <= x1,x2 <= 1
    0 < x3 < 2

该案例展示了一个带等式约束的连续型决策变量最大化目标的单目标优化问题。
与soea_demo2不同之处是采用先验知识帮助进化优化。
"""

def evalVars(Vars):  # 目标函数
    x1 = Vars[:, [0]]
    x2 = Vars[:, [1]]
    x3 = Vars[:, [2]]
    f = 4 * x1 + 2 * x2 + x3
    # 采用可行性法则处理约束
    CV = np.hstack([2 * x1 + x2 - 1,
                        x1 + 2 * x3 - 2,
                        np.abs(x1 + x2 + x3 - 1)])
    return f, CV

def calReferObjV():  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值）
    referenceObjV = np.array([[2.5]])
    return referenceObjV

Dim = 3 # 初始化Dim（决策变量维数）
problem = ea.Problem(name = 'MyProblem',  # 初始化name（函数名称，可以随意设置）
                    M = 1,  # 初始化M（目标维数）
                    maxormins = [-1],  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
                    Dim = Dim,  # 初始化Dim（决策变量维数）
                    varTypes = [0] * Dim,  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
                    lb = [0, 0, 0],  # 决策变量下界
                    ub = [1, 1, 2],  # 决策变量上界
                    lbin = [1, 1, 0],  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
                    ubin = [1, 1, 0],  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
                        evalVars=evalVars,
                        calReferObjV=calReferObjV)
# 快速构建算法
algorithm = ea.soea_DE_currentToBest_1_bin_templet(problem,
                                                    ea.Population(Encoding='RI', NIND=20),
                                                    MAXGEN=400,  # 最大进化代数。
                                                    logTras=0)  # 表示每隔多少代记录一次日志信息，0表示不记录。
algorithm.mutOper.F = 0.7  # 差分进化中的参数F。
algorithm.recOper.XOVR = 0.7  # 交叉概率。
# 先验知识
prophetVars = np.array([[0.4, 0.2, 0.4]])  # 假设已知[0.4, 0.2, 0.4]为一组比较优秀的变量。
# 求解
res = ea.optimize(algorithm, prophet=prophetVars, verbose=True, drawing=1, outputMsg=True, drawLog=True, saveFlag=False)
print(res)
