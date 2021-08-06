# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:05:57 2020

@author: Dell
"""

import numpy as np
import pylab as pl
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import *                                 #支持中文
import networkx as nx
import random
import copy
import pandas as pd
from matplotlib.font_manager import FontProperties
import time

def creatG(filename):
    df = pd.read_csv(filename)
    G = nx.from_pandas_edgelist(df,'node1','node2',create_using=nx.MultiGraph())
    return G,df


G_cas,df_cas = creatG(filename ='cas.csv')

G_cas_pdup,df_cas_pdup = creatG(filename ='cas_callpdup.csv')
G_cas_pddown,df_cas_pddown = creatG(filename ='cas_callpddown.csv')
G_cas_pdrandom,df_cas_pdrandom = creatG(filename ='cas_callpdrandom.csv')

def si(G0,b,time,e):
#参数G0为网络，
#N：节点个数，
#b: 传染率，
#g: 恢复率
#time:传播时间
    G = copy.deepcopy(G0)
    nodes = list(G.nodes())#节点列表
    N = len(nodes)#节点个数
    I=1#选择一个感染节点
    #I=int(N*0.01)
    S=N-I#剩余为易感节点
#    Ilist=random.sample(nodes,I)#随机选取感染节点
    #Ilist=nodes[:I] 

    Ilist = e      
    G.add_nodes_from(Ilist,state='I') #添加节点的初始属性状态；I:感染、S、未感染
    Slist=[item for item in nodes if item not in set(Ilist)]
    G.add_nodes_from(Slist, state='S')
    sList=[]
    iList=[]
    iiList=[1/len(nodes)]
    iList.append(len(Ilist))
    sList.append(len(Slist))
    a=0
    for t in range(time):
        I_list1=[]#用于存放各时刻新感染节点
        for n in Ilist:
            for nei in G.neighbors(n):#寻找感染节点的邻居节点
                if G.nodes[nei]['state']=='S' and random.random()<b:#如果感染节点的邻居节点的状态为：'S',且小于传染率    
                    G.nodes[nei]['state']='I'#则该邻居节点被感染
                    I_list1.append(nei)#新感染节点加入列表
                    Slist.remove(nei)#未感染者列表删除该感染节点
        Ilist.extend(I_list1)#感染者列表加入新感染者列表
        sList.append(len(Slist))#各个时刻易感染者数量集合
        iList.append(len(Ilist))#各个时刻感染者数量集合
        iiList.append(len(Ilist)/N)
        a=a+1
#        print('A',(a*100)/len(range(time)))
    return iiList#返回各个统计量
#
print('start SI')
start = time.time()
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
            
nodes_Baw = list(G_cas.nodes())#节点列表

list_origin = [si(G_cas,0.1,150,[i]) for i in nodes_Baw[50:150]]
list_pdup = [si(G_cas_pdup,0.1,150,[i]) for i in nodes_Baw[50:150]]
list_pddown = [si(G_cas_pddown,0.1,150,[i]) for i in nodes_Baw[50:150]]
list_pdrandom = [si(G_cas_pdrandom,0.1,150,[i]) for i in nodes_Baw[50:150]]
iList_cas=list(np.array(list_origin).mean(axis=0))
iList_cas_pdup=list(np.array(list_pdup).mean(axis=0))
iList_cas_pddown=list(np.array(list_pddown).mean(axis=0))
iList_cas_pdrandom=list(np.array(list_pdrandom).mean(axis=0))
pl.plot(iList_cas, 'k-',label='原始双层网络')
pl.plot(iList_cas_pdup, 'rd-',markersize = 5,label='层间度相关系数增加双层网络')
pl.plot(iList_cas_pddown, 'bo-',markersize = 5,label='层间度相关系数减小双层网络')
pl.plot(iList_cas_pddown, 'ys-',markersize = 5,label='层间度相关系数随机双层网络')
pl.legend(loc="best",fontsize=10)
pl.xlabel('step',fontsize=10)
pl.ylabel('I(t)',fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('SI_pd.png',dpi = 750)
print(time.time()-start)
print('over SI')