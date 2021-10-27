# @Time : Created on 2021/8/15 16:19 
# @File : 检验方法.py 
# @Author : xiaozhi

import pandas as pd
import numpy as np
import ling_model as lm
import networkx as nx
# Z检验方法
# 使用BA网络和WS网络测试
dfBA = pd.read_csv('../IEEE.csv')
# BAG = nx.from_pandas_edgelist(dfBA, source='node1', target='node2')
dfws = pd.read_csv('../Internet.csv')
# wsG = nx.from_pandas_edgelist(dfws)
ling_0_arr = []

for i in range(50):
    G = lm.create_MultilayerGraphFromPandasEdgeList(dfBA, dfws, source_df1='node1', target_df1='node2', source_df2='node1',
                                                target_df2='node2')
    N = int(len(G.nodes) / 2)
    newG = lm.random_changenode_model(G, N, 1000)
    # pear_val = lm.pearson_correlation_coefficient(newG, N)
    mutual_information_val = lm.overall_of_overlap_network(newG, N)
    ling_0_arr.append(mutual_information_val)
    print(f'第{i}次计算的值为{mutual_information_val}')

# ling_0_arr = []
# with open('./mutual_information样本标准差0阶零模型结果数据.txt', 'r', encoding='utf-8') as f:
#     for line in f:
#         val = float(line.strip().split('为')[1])
#         ling_0_arr.append(val)

np_array = np.array(ling_0_arr)
mean = np.average(np_array)
print(f'均值mean：{mean}')
sd = np.std(np_array, ddof=1)
print(f'样本标准差值SD：{sd}')
#
# 数据写入文件
f=open('mutual_information样本标准差节点置乱零模型结果数据.txt','w')

for i in range(50):
    f.write(f'{ling_0_arr[i]}\n')
f.close()
print(ling_0_arr)