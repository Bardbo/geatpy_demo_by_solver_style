# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea
import matplotlib.pyplot as plt

"""
    有十座城市：A, B, C, D, E, F, G, H, I, J，坐标如下：
        X      Y
    [[0.4,  0.4439],
     [0.2439,0.1463],
     [0.1707,0.2293],
     [0.2293,0.761],
     [0.5171,0.9414],
     [0.8732,0.6536],
     [0.6878,0.5219],
     [0.8488,0.3609],
     [0.6683,0.2536],
     [0.6195,0.2634]]
    某旅行者从A城市出发，想逛遍所有城市，并且每座城市去且只去一次，最后要返回出发地，
而且需要从G地拿重要文件到D地，另外要从F地把公司的车开到E地，那么他应该如何设计行程方案，才能用
最短的路程来满足他的旅行需求？
    分析：在这个案例中，旅行者从A地出发，把其他城市走遍一次后回到A地，因此我们只需要考虑中间途
径的9个城市的访问顺序即可。这9个城市需要排列组合选出满足约束条件的最优的排列顺序作为最终的路线方案。

该案例展示了一个带约束的单目标旅行商问题的求解。
"""

places = np.array([[0.4, 0.4439],
                    [0.2439, 0.1463],
                    [0.1707, 0.2293],
                    [0.2293, 0.761],
                    [0.5171, 0.9414],
                    [0.8732, 0.6536],
                    [0.6878, 0.5219],
                    [0.8488, 0.3609],
                    [0.6683, 0.2536],
                    [0.6195, 0.2634]])

def evalVars(x):  # 目标函数
    # 添加从0地出发且最后回到出发地
    X = np.hstack([np.zeros((x.shape[0], 1)), x, np.zeros((x.shape[0], 1))]).astype(int)
    ObjV = []  # 存储所有种群个体对应的总路程
    for i in range(X.shape[0]):
        journey = places[X[i], :]  # 按既定顺序到达的地点坐标
        distance = np.sum(np.sqrt(np.sum(np.diff(journey.T) ** 2, 0)))  # 计算总路程
        ObjV.append(distance)
    f = np.array([ObjV]).T
    # 找到违反约束条件的个体在种群中的索引，保存在向量exIdx中（如：若0、2、4号个体违反约束条件，则编程找出他们来）
    exIdx1 = np.where(np.where(x == 3)[1] - np.where(x == 6)[1] < 0)[0]
    exIdx2 = np.where(np.where(x == 4)[1] - np.where(x == 5)[1] < 0)[0]
    exIdx = np.unique(np.hstack([exIdx1, exIdx2]))
    CV = np.zeros((x.shape[0], 1))
    CV[exIdx] = 1  # 把求得的违反约束程度矩阵赋值给种群pop的CV
    return f, CV

Dim = 9 # 初始化Dim（决策变量维数）
problem = ea.Problem(name = 'MyProblem',  # 初始化name（函数名称，可以随意设置）
                        M = 1,  # 初始化M（目标维数）
                        maxormins = [1],  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
                        Dim = Dim,  # 初始化Dim（决策变量维数）
                        varTypes = [1] * Dim,  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
                        lb = [1] * Dim,  # 决策变量下界
                        ub = [9] * Dim,  # 决策变量上界
                        lbin = [1] * Dim,  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
                        ubin = [1] * Dim,  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
                        evalVars=evalVars)
# 构建算法
algorithm = ea.soea_SEGA_templet(problem,
                                    ea.Population(Encoding='P', NIND=50),
                                    MAXGEN=200,  # 最大进化代数
                                    logTras=1)  # 表示每隔多少代记录一次日志信息，0表示不记录。
algorithm.mutOper.Pm = 0.5  # 变异概率
# 求解
res = ea.optimize(algorithm, verbose=True, drawing=1, outputMsg=True, drawLog=False, saveFlag=False)
# 绘制路线图
if res['success']:
    print('最短路程为：%s' % res['ObjV'][0][0])
    print('最佳路线为：')
    best_journey = np.hstack([0, res['Vars'][0, :], 0])
    for i in range(len(best_journey)):
        print(int(best_journey[i]), end=' ')
    print()
    # 绘图
    plt.figure()
    plt.plot(places[best_journey.astype(int), 0], places[best_journey.astype(int), 1], c='black')
    plt.plot(places[best_journey.astype(int), 0], places[best_journey.astype(int), 1], 'o',
                c='black')
    for i in range(len(best_journey)):
        plt.text(places[int(best_journey[i]), 0], places[int(best_journey[i]), 1],
                    chr(int(best_journey[i]) + 65), fontsize=20)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('roadmap.svg', dpi=600, bbox_inches='tight')
    plt.show()
else:
    print('没找到可行解。')
