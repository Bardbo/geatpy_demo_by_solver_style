# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea

"""
    该案例展示了一个利用单目标进化算法实现句子匹配的应用实例。
"""

# 定义需要匹配的句子
strs = 'Tom is a little boy, isn\'t he? Yes he is, he is a good and smart child and he is always ready to ' \
    'help others, all in all we all love him very much.'
words = [ord(c) for c in strs] # 把字符串转成ASCII码
print('句子长度: ', len(words))

def evalVars(Vars):  # 目标函数
    diff = np.sum((Vars - words) ** 2, 1)
    f = np.array([diff]).T
    return f

Dim = len(words) # 初始化Dim（决策变量维数）
problem = ea.Problem(name = 'MyProblem',  # 初始化name（函数名称，可以随意设置）
                    M = 1,  # 初始化M（目标维数）
                    maxormins = [1],  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
                    Dim = Dim,  # 初始化Dim（决策变量维数）
                    varTypes = [1] * Dim,  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
                    lb = [32] * Dim, # 决策变量下界
                    ub = [122] * Dim,  # 决策变量上界
                    lbin = [1] * Dim,  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
                    ubin = [1] * Dim,  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
                        evalVars=evalVars)

# 快速构建算法
algorithm = ea.soea_DE_rand_1_L_templet(problem,
                                        ea.Population(Encoding='RI', NIND=50),
                                        MAXGEN=2000,  # 最大进化代数
                                        logTras=1)  # 表示每隔多少代记录一次日志信息，0表示不记录。
# 求解
res = ea.optimize(algorithm, verbose=True, drawing=1, outputMsg=False, drawLog=False, saveFlag=False)
print('最佳目标函数值：%s' % res['ObjV'][0][0])
print('搜索到的句子为：')
for num in res['Vars'][0, :]:
    print(chr(int(num)), end='')
