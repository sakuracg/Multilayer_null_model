# @Time : Created on 2021/12/5 11:02 
# @File : 0阶-----Pearson相关系数求足够随机化次数.py 
# @Author : xiaozhi

import ling_model as lm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 首先先读取实证网络数据
# 接着针对call层进行一阶零模型随机化
# 接着计算两层的pearson相关系数
# 重复该过程10000次
# 将结果写入文件
filename1 = 'IEEE.csv'
filename2 = 'Internet.csv'

dfBA = pd.read_csv(filename1)
dfws = pd.read_csv(filename2)


# 思路是每交换10次求一次值，看有效交换多少次后值达到稳定了
# nswap = 0
# while nswap <= 10000:
#     G = lm.create_MultilayerGraphFromPandasEdgeList(dfBA, dfws, source_df1='node1', target_df1='node2',
#                                                         source_df2='node1',
#                                                         target_df2='node2')
#     N = int(len(G.nodes) / 2)
#     newG = lm.random_0_model(G, N, nswap=nswap, max_tries=400000)
#     val = lm.pearson_correlation_coefficient(newG, N)
#
#     with open('0阶-----Pearson相关系数求足够随机化次数.txt', 'a') as f:
#         f.write(str(val))
#         f.write('\n')
#     print('nswap:{}---pearson value: {}'.format(nswap, val))
#     nswap += 10

### 画图
# pdData = pd.read_csv('0阶-----Pearson相关系数求足够随机化次数.txt', names=['value']) # 路径的斜线必须是/   不能是\   不然会报错
# fmri = pd.DataFrame()
# fmri['x'] = [n for n in range(0, 6630, 10)]
# fmri['value'] = list(pdData['value'].values)
#
# sns.set(style="ticks",palette="muted")
# sns.set_context("notebook",font_scale=0.8)
# plt.figure(figsize=(5.5,4),dpi=144,facecolor=None,edgecolor=None,frameon=False)
#
# sns.lineplot(x="x", y="value",markers=True, dashes=False, data=fmri)
#
# plt.rcParams['font.sans-serif']='SimHei'#调节字体显示为中文
# plt.rcParams['axes.unicode_minus']=False
# plt.xlabel('有效交换次数',fontsize=7,family = 'SimHei')  #添加x轴的名称
# plt.ylabel('Pearson相关系数值',fontsize=7,family = 'SimHei')  #添加y轴的名称
# plt.xticks(fontproperties = 'Times New Roman', size = 7)
# plt.yticks(fontproperties = 'Times New Roman', size = 7)
# # plt.legend(handles=[l1,l2],labels=['线1的标注','线1的标注'],loc=0,prop={'family' : 'SimHei', 'size'   : 8},ncol=1)
# plt.show()
# plt.savefig('.png',dpi=600)
# plt.savefig('04.svg',dpi=600)