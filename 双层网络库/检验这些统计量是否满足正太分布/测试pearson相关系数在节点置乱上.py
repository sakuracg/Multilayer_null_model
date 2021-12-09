# @Time : Created on 2021/12/3 11:07
# @File : 测试pearson相关系数在节点置乱上.py
# @Author : xiaozhi
import ling_model as lm
import pandas as pd

# 首先先读取实证网络数据
# 接着针对call层进行一阶零模型随机化
# 接着计算两层的pearson相关系数
# 重复该过程10000次
# 将结果写入文件

filename1 = 'IEEE.csv'
filename2 = 'Internet.csv'

dfBA = pd.read_csv(filename1)
dfws = pd.read_csv(filename2)
count = 514
while count <= 10000:
    G = lm.create_MultilayerGraphFromPandasEdgeList(dfBA, dfws, source_df1='node1', target_df1='node2',
                                                        source_df2='node1',
                                                        target_df2='node2')
    N = int(len(G.nodes) / 2)
    newG = lm.random_changenode_model(G, N, nswap=20000)
    val = lm.pearson_correlation_coefficient(newG, N)

    with open('res1.txt', 'a') as f:
        f.write(str(val))
        f.write('\n')
    print('count value: {}'.format(count))
    count += 1