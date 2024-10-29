# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea

"""
max f = x * np.sin(10 * np.pi * x) + 2.0
s.t.
-1 <= x <= 2

该案例是soea_demo1的拓展，展示了如何利用多种群来进行单目标优化。
在执行脚本中，通过调用带"multi"字样的进化算法类来进行多种群进化优化。
"""

def evalVars(Vars):  # 定义目标函数（含约束）
    ObjV = Vars * np.sin(10 * np.pi * Vars) + 2.0  # 计算目标函数值
    return ObjV

Dim = 1 # 初始化Dim（决策变量维数）
problem = ea.Problem(name = 'MyProblem',  # 初始化name（函数名称，可以随意设置）
                        M = 1,  # 初始化M（目标维数）
                        maxormins = [-1],  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
                        Dim = Dim, # 初始化Dim（决策变量维数）
                        varTypes = [0] * Dim,  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
                        lb = [-1],  # 决策变量下界
                        ub = [2],  # 决策变量上界
                        lbin = [1] * Dim,  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
                        ubin = [1] * Dim, # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
                        evalVars=evalVars)
# 种群设置
Encoding = 'RI'  # 编码方式
NINDs = [5, 10, 15, 20]  # 种群规模
population = []  # 创建种群列表
for i in range(len(NINDs)):
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器。
    population.append(ea.Population(Encoding, Field, NINDs[i]))  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）。
# 构建算法
algorithm = ea.soea_multi_SEGA_templet(problem,
                                        population,
                                        MAXGEN=30,  # 最大进化代数。
                                        logTras=1,  # 表示每隔多少代记录一次日志信息，0表示不记录。
                                        trappedValue=1e-6,  # 单目标优化陷入停滞的判断阈值。
                                        maxTrappedCount=5)  # 进化停滞计数器最大上限值。
# 求解
res = ea.optimize(algorithm, verbose=True, drawing=1, outputMsg=True, drawLog=False, saveFlag=False)
print(res)
