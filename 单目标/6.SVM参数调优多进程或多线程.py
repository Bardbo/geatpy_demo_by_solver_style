# -*- coding: utf-8 -*-
import multiprocessing as mp
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import cross_val_score

import geatpy as ea

"""
本问题需要用到一个外部数据集，存放在同目录下的iris.data中，
并且把iris.data按3:2划分为训练集数据iris_train.data和测试集数据iris_test.data。
有关该数据集的详细描述详见http://archive.ics.uci.edu/ml/datasets/Iris
"""

PoolType = 'Thread' # Thread or Process

# 目标函数计算中用到的一些数据
fp = open('iris_train.data')
datas = []
data_targets = []
for line in fp.readlines():
    line_data = line.strip('\n').split(',')
    data = []
    for i in line_data[0:4]:
        data.append(float(i))
    datas.append(data)
    data_targets.append(line_data[4])
fp.close()
data = preprocessing.scale(np.array(datas))  # 训练集的特征数据（归一化）
dataTarget = np.array(data_targets)

def subAimFunc(args):
    i = args[0]
    Vars = args[1]
    data = args[2]
    dataTarget = args[3]
    C = Vars[i, 0]
    G = Vars[i, 1]
    svc = svm.SVC(C=C, kernel='rbf', gamma=G).fit(data, dataTarget)  # 创建分类器对象并用训练集的数据拟合分类器模型
    scores = cross_val_score(svc, data, dataTarget, cv=30)  # 计算交叉验证的得分
    ObjV_i = [scores.mean()]  # 把交叉验证的平均得分作为目标函数值
    return ObjV_i

def test(C, G):  # 代入优化后的C、Gamma对测试集进行检验
    # 读取测试集数据
    fp = open('iris_test.data')
    datas = []
    data_targets = []
    for line in fp.readlines():
        line_data = line.strip('\n').split(',')
        data_ = []
        for i in line_data[0:4]:
            data_.append(float(i))
        datas.append(data_)
        data_targets.append(line_data[4])
    fp.close()
    data_test = preprocessing.scale(np.array(datas))  # 测试集的特征数据（归一化）
    dataTarget_test = np.array(data_targets)  # 测试集的标签数据
    svc = svm.SVC(C=C, kernel='rbf', gamma=G).fit(data, dataTarget)  # 创建分类器对象并用训练集的数据拟合分类器模型
    dataTarget_predict = svc.predict(data_test)  # 采用训练好的分类器对象对测试集数据进行预测
    print("测试集数据分类正确率 = %s%%" % (
            len(np.where(dataTarget_predict == dataTarget_test)[0]) / len(dataTarget_test) * 100))


if __name__ == "__main__":
    if PoolType == 'Thread':
        pool = ThreadPool(2)  # 设置池的大小
    elif PoolType == 'Process':
        num_cores = int(mp.cpu_count())  # 获得计算机的核心数
        pool = ProcessPool(num_cores)  # 设置池的大小

    def evalVars(Vars):  # 目标函数，采用多线程加速计算
        N = Vars.shape[0]
        args = list(
            zip(list(range(N)), [Vars] * N, [data] * N, [dataTarget] * N))
        if PoolType == 'Thread':
            f = np.array(list(pool.map(subAimFunc, args)))
        elif PoolType == 'Process':
            result = pool.map_async(subAimFunc, args)
            result.wait()
            f = np.array(result.get())
        return f

    Dim = 2 # 初始化Dim（决策变量维数）
    problem = ea.Problem(name = 'MyProblem',  # 初始化name（函数名称，可以随意设置）
                        M = 1,  # 初始化M（目标维数）
                        maxormins = [-1],  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
                        Dim = Dim,  # 初始化Dim（决策变量维数）
                        varTypes = [0, 0],  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
                        lb = [2 ** (-8)] * Dim,  # 决策变量下界
                        ub = [2 ** 8] * Dim,  # 决策变量上界
                        lbin = [1] * Dim,  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
                        ubin = [1] * Dim,  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
                            evalVars=evalVars)

    # 构建算法
    algorithm = ea.soea_DE_rand_1_bin_templet(problem,
                                                ea.Population(Encoding='RI', NIND=50),
                                                MAXGEN=30,  # 最大进化代数。
                                                logTras=1,  # 表示每隔多少代记录一次日志信息，0表示不记录。
                                                trappedValue=1e-6,  # 单目标优化陷入停滞的判断阈值。
                                                maxTrappedCount=10)  # 进化停滞计数器最大上限值。
    # 求解
    res = ea.optimize(algorithm, verbose=True, drawing=1, outputMsg=True, drawLog=False, saveFlag=False)
    # 检验结果
    if res['success']:
        test(C=res['Vars'][0, 0], G=res['Vars'][0, 1])
