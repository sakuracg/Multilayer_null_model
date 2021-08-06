# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 20:42:23 2017
revised on Oct 15 2017
@author:xiaoke   
"""

import networkx as nx
import random
import copy
"""
selfloop:config_model\random_1k
"""

def dict_degree_nodes(degree_node_list):
#返回的字典为{度：[节点1，节点2，..]}，其中节点1和节点2有相同的度
    D = {}      
    for degree_node_i in degree_node_list:
        if degree_node_i[0] not in D:
            D[degree_node_i[0]] = [degree_node_i[1]]
        else:
            D[degree_node_i[0]].append(degree_node_i[1])
    return D 

def ER_model(G0):
    n=len(G0.nodes())
    m=len(G0.edges())
    p=2.0*m/(n*n)
    G = nx.random_graphs.erdos_renyi_graph(n, p, directed=False)
    return G

def config_model(G0):
    degree_seq = list(G0.degree().values()) 
    G = nx.configuration_model(degree_seq) 
    return G

def random_0k(G0, nswap=1, max_tries=100, connected=1):  
# 基于随机断边重连的0阶零模型
# G0：待改变结构的网络
# node_community_list：是网络中节点的社团归属信息
# nswap：是改变成功的系数，默认值为1
# max_tries：是尝试改变的次数，默认值为100
# connected：是否需要保证网络的联通特性，参数为1需要保持，参数为0不需要保持

    if G0.is_directed():
        raise nx.NetworkXError("It is only allowed for undirected networks")
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G0) < 3:
        raise nx.NetworkXError("Graph has less than three nodes.")  
        
    G = copy.deepcopy(G0)      

    n = 0
    swapcount = 0
    edges = G.edges()   
    nodes = G.nodes()
    while swapcount < nswap:
        n=n+1
        u,v = random.choice(edges)      #随机选网络中的一条要断开的边
        x,y = random.sample(nodes,2)    #随机找两个不相连的节点
        if len(set([u,v,x,y]))<4:
            continue
        if (x,y) not in edges and (y,x) not in edges:
            G.remove_edge(u,v)              #断旧边
            G.add_edge(x,y)                 #连新边
            edges.remove((u,v))
            edges.append((x,y))
            
            if connected==1:                            #判断是否需要保持联通特性，为1的话则需要保持该特性        
            	if not nx.is_connected(G):              #保证网络是全联通的:若网络不是全联通网络，则撤回交换边的操作
                    G.add_edge(u,v)
                    G.add_edge(x,y)
                    G.remove_edge(u,y)
                    G.remove_edge(x,v)
                    continue 
            swapcount=swapcount+1       

        if n >= max_tries:
            e=('Maximum number of swap attempts (%s) exceeded '%n +'before desired swaps achieved (%s).'%nswap)
            print(e)
            break
    return G

def random_1k(G0,nswap=1, max_tries=100,connected=1): 
# 保证度分布特性不变的情况下随机交换连边
# G0：待改变结构的网络
# nswap：是改变成功的系数，默认值为1
# max_tries：是尝试改变的次数，默认值为100
# connected：是否需要保证网络的联通特性，参数为1需要保持，参数为0不需要保持

    if not nx.is_connected(G0):
        raise nx.NetworkXError("非连通图，必须为连通图")
    if G0.is_directed():
        raise nx.NetworkXError("仅适用于无向图")
    if nswap > max_tries:
        raise nx.NetworkXError("交换次数超过允许的最大次数")
    if len(G0) < 4:
        raise nx.NetworkXError("节点数太少，至少要含四个节点")    
         
    tn = 0   #尝试次数
    swapcount = 0   #有效交换次数
    
    G = copy.deepcopy(G0)
    keys,degrees = zip(*G.degree().items()) 
    cdf = nx.utils.cumulative_distribution(degrees) 
   
    while swapcount < nswap:       #有效交换次数小于规定交换次数      
        if tn >= max_tries:
            e=('尝试次数 (%s) 已超过允许的最大次数'%tn + '有效交换次数（%s)'%swapcount)
            print (e)
            break     
        tn += 1 

        # 在保证度分布不变的情况下，随机选取两条连边u-v，x-y
        (ui,xi)=nx.utils.discrete_sequence(2,cdistribution=cdf)
        if ui==xi:
            continue
        u = keys[ui] 
        x = keys[xi]
        v = random.choice(list(G[u]))
        y = random.choice(list(G[x]))
        
        if len(set([u,v,x,y])) == 4: #保证是四个独立节点
            if (y not in G[u]) and (v not in G[x]):       #保证新生成的连边是原网络中不存在的边
                G.add_edge(u,y)                           #增加两条新连边
                G.add_edge(v,x)
                
                G.remove_edge(u,v)                        #删除两条旧连边
                G.remove_edge(x,y)
                		
                if connected==1:                            #判断是否需要保持联通特性，为1的话则需要保持该特性        
                    if not nx.is_connected(G):              #保证网络是全联通的:若网络不是全联通网络，则撤回交换边的操作
                    	G.add_edge(u,v)
                    	G.add_edge(x,y)
                    	G.remove_edge(u,y)
                    	G.remove_edge(x,v)
                    	continue 
                swapcount=swapcount+1              
    return G       

def random_2k(G0,nswap=1, max_tries=100,connected=1): 
# 保证2k特性不变和网络联通的情况下，交换社团内部的连边
# G0：待改变结构的网络
# nswap：是改变成功的系数，默认值为1
# max_tries：是尝试改变的次数，默认值为100
# connected：是否需要保证网络的联通特性，参数为1需要保持，参数为0不需要保持

    if not nx.is_connected(G0):
        raise nx.NetworkXError("非连通图，必须为连通图")
    if G0.is_directed():
        raise nx.NetworkXError("仅适用于无向图")
    if nswap > max_tries:
        raise nx.NetworkXError("交换次数超过允许的最大次数")
    if len(G0) < 3:
        raise nx.NetworkXError("节点数太少，至少要含三个节点")    
         
    tn = 0   #尝试次数
    swapcount = 0   #有效交换次数
    
    G = copy.deepcopy(G0)
    keys,degrees = zip(*G.degree().items()) 
    cdf = nx.utils.cumulative_distribution(degrees) 
   
    while swapcount < nswap:       #有效交换次数小于规定交换次数      
        if tn >= max_tries:
            e=('尝试次数 (%s) 已超过允许的最大次数'%tn + '有效交换次数（%s)'%swapcount)
            print (e)
            break     
        tn += 1 

        # 在保证度分布不变的情况下，随机选取两条连边u-v，x-y
        (ui,xi)=nx.utils.discrete_sequence(2,cdistribution=cdf)
        if ui==xi:
            continue
        u = keys[ui] 
        x = keys[xi]
        v = random.choice(list(G[u]))
        y = random.choice(list(G[x]))
        
        if len(set([u,v,x,y])) == 4: #保证是四个独立节点
            if G.degree(v)==G.degree(y):#保证节点的度匹配特性不变
                if (y not in G[u]) and (v not in G[x]):       #保证新生成的连边是原网络中不存在的边
                    G.add_edge(u,y)                           #增加两条新连边
                    G.add_edge(v,x)
                
                    G.remove_edge(u,v)                        #删除两条旧连边
                    G.remove_edge(x,y)
                		
                    if connected==1:                            #判断是否需要保持联通特性，为1的话则需要保持该特性        
                        if not nx.is_connected(G):              #保证网络是全联通的:若网络不是全联通网络，则撤回交换边的操作
                        	G.add_edge(u,v)
                        	G.add_edge(x,y)
                        	G.remove_edge(u,y)
                        	G.remove_edge(x,v)
                        	continue 
                    swapcount=swapcount+1              
    return G       

def random_25k(G0,nswap=1, max_tries=100,connected=1): 
# 保证2.5k特性不变和网络联通的情况下，交换社团内部的连边
# G0：待改变结构的网络
# nswap：是改变成功的系数，默认值为1
# max_tries：是尝试改变的次数，默认值为100
# connected：是否需要保证网络的联通特性，参数为1需要保持，参数为0不需要保持

    if not nx.is_connected(G0):
        raise nx.NetworkXError("非连通图，必须为连通图")
    if G0.is_directed():
        raise nx.NetworkXError("仅适用于无向图")
    if nswap > max_tries:
        raise nx.NetworkXError("交换次数超过允许的最大次数")
    if len(G0) < 3:
        raise nx.NetworkXError("节点数太少，至少要含三个节点")    
         
    tn = 0   #尝试次数
    swapcount = 0   #有效交换次数
    
    G = copy.deepcopy(G0)
    keys,degrees = zip(*G.degree().items()) 
    cdf = nx.utils.cumulative_distribution(degrees) 
   
    while swapcount < nswap:       #有效交换次数小于规定交换次数      
        if tn >= max_tries:
            e=('尝试次数 (%s) 已超过允许的最大次数'%tn + '有效交换次数（%s)'%swapcount)
            print (e)
            break     
        tn += 1 

        # 在保证度分布不变的情况下，随机选取两条连边u-v，x-y
        (ui,xi)=nx.utils.discrete_sequence(2,cdistribution=cdf)
        if ui==xi:
            continue
        u = keys[ui] 
        x = keys[xi]
        v = random.choice(list(G[u]))
        y = random.choice(list(G[x]))
        
        if len(set([u,v,x,y])) == 4: #保证是四个独立节点
            if G.degree(v)==G.degree(y):#保证节点的度匹配特性不变
                if (y not in G[u]) and (v not in G[x]):       #保证新生成的连边是原网络中不存在的边
                    G.add_edge(u,y)                           #增加两条新连边
                    G.add_edge(v,x)
                
                    G.remove_edge(u,v)                        #删除两条旧连边
                    G.remove_edge(x,y)
                		
                    degree_node_list = map(lambda t:(t[1],t[0]), G0.degree([u,v,x,y]+list(G[u])+list(G[v])+list(G[x])+list(G[y])).items())  
                	#先找到四个节点以及他们邻居节点的集合，然后取出这些节点所有的度值对应的节点，格式为（度，节点）形式的列表
                    D = dict_degree_nodes(degree_node_list) # 找到每个度对应的所有节点，具体形式为
                		
                    for i in range(len(D)):
                        avcG0 = nx.average_clustering(G0, nodes=D.values()[i], weight=None, count_zeros=True)
                        avcG = nx.average_clustering(G, nodes=D.values()[i], weight=None, count_zeros=True)
                        i += 1
                        
                    if avcG0 != avcG:   #若置乱前后度相关的聚类系数不同，则撤销此次置乱操作
                        G.add_edge(u,v)
                        G.add_edge(x,y)
                        G.remove_edge(u,y)
                        G.remove_edge(x,v)
                        break
                    		
                    if connected==1:                            #判断是否需要保持联通特性，为1的话则需要保持该特性        
                    	if not nx.is_connected(G):              #保证网络是全联通的:若网络不是全联通网络，则撤回交换边的操作
                    		G.add_edge(u,v)
                    		G.add_edge(x,y)
                    		G.remove_edge(u,y)
                    		G.remove_edge(x,v)
                    		continue 
                    		
                    swapcount=swapcount+1                
    return G       


def random_3k(G0,nswap=1,max_tries=100,connected=1): 
# 保证3k特性不变和网络联通的情况下，交换社团内部的连边
# G0：待改变结构的网络
# nswap：是改变成功的系数，默认值为1
# max_tries：是尝试改变的次数，默认值为100
# connected：是否需要保证网络的联通特性，参数为1需要保持，参数为0不需要保持

    if not nx.is_connected(G0):
        raise nx.NetworkXError("非连通图，必须为连通图")
    if G0.is_directed():
        raise nx.NetworkXError("仅适用于无向图")
    if nswap > max_tries:
        raise nx.NetworkXError("交换次数超过允许的最大次数")
    if len(G0) < 3:
        raise nx.NetworkXError("节点数太少，至少要含三个节点")    
         
    tn = 0   #尝试次数
    swapcount = 0   #有效交换次数
    
    G = copy.deepcopy(G0)
    keys,degrees = zip(*G.degree().items()) 
    cdf = nx.utils.cumulative_distribution(degrees) 
   
    while swapcount < nswap:       #有效交换次数小于规定交换次数      
        if tn >= max_tries:
            e=('尝试次数 (%s) 已超过允许的最大次数'%tn + '有效交换次数（%s)'%swapcount)
            print (e)
            break     
        tn += 1 

        # 在保证度分布不变的情况下，随机选取两条连边u-v，x-y
        (ui,xi)=nx.utils.discrete_sequence(2,cdistribution=cdf)
        if ui==xi:
            continue
        u = keys[ui] 
        x = keys[xi]
        v = random.choice(list(G[u]))
        y = random.choice(list(G[x]))
        
        if len(set([u,v,x,y])) == 4: #保证是四个独立节点
            if G.degree(v)==G.degree(y):#保证节点的度匹配特性不变
                if (y not in G[u]) and (v not in G[x]):       #保证新生成的连边是原网络中不存在的边
                    G.add_edge(u,y)                           #增加两条新连边
                    G.add_edge(v,x)
                
                    G.remove_edge(u,v)                        #删除两条旧连边
                    G.remove_edge(x,y)
                		
                    node_list=[u,v,x,y]+list(G[u])+list(G[v])+list(G[x])+list(G[y]) #找到四个节点以及他们邻居节点的集合
                    avcG0 = nx.clustering(G0, nodes=node_list) #计算旧网络中4个节点以及他们邻居节点的聚类系数
                    avcG = nx.clustering(G, nodes=node_list)   #计算新网络中4个节点以及他们邻居节点的聚类系数
                        
                    if avcG0 != avcG:                          #保证涉及到的四个节点聚类系数相同:若聚类系数不同，则撤回交换边的操作
                        G.add_edge(u,v)
                        G.add_edge(x,y)
                        G.remove_edge(u,y)
                        G.remove_edge(x,v)
                        continue
                    if connected==1:                            #判断是否需要保持联通特性，为1的话则需要保持该特性        
                    	if not nx.is_connected(G):              #保证网络是全联通的:若网络不是全联通网络，则撤回交换边的操作
                    		G.add_edge(u,v)
                    		G.add_edge(x,y)
                    		G.remove_edge(u,y)
                    		G.remove_edge(x,v)
                    		continue 
                    swapcount=swapcount+1              
    return G       

def rich_club_create(G0, k=1, nswap=1, max_tries=100, connected=1):
    """
    任选两条边(富节点和非富节点的连边)，若富节点间无连边，非富节点间无连边，则断边重连
    达到最大尝试次数或全部富节点间都有连边，循环结束
    """
	# G0：待改变结构的网络
   # k 为富节点度值的门限值
	# nswap：是改变成功的系数，默认值为1
	# max_tries：是尝试改变的次数，默认值为100
	# connected：是否需要保证网络的联通特性，参数为1需要保持，参数为0不需要保持
    
    if not nx.is_connected(G0):
        raise nx.NetworkXError("非连通图，必须为连通图")
    if G0.is_directed():
        raise nx.NetworkXError("仅适用于无向图")
    if nswap > max_tries:
        raise nx.NetworkXError("交换次数超过允许的最大次数")
    if len(G0) < 3:
        raise nx.NetworkXError("节点数太少，至少要含三个节点")    
         
    tn = 0   #尝试次数
    swapcount = 0   #有效交换次数   
    G = copy.deepcopy(G0)
    hubs = [e for e in G.nodes() if G.degree(e)>k]     #全部富节点
    hubs_edges = [e for e in G.edges() if G.degree()[e[0]]>k and G.degree()[e[1]]> k] #网络中已有的富节点和富节点的连边
    len_possible_edges = len(hubs)*(len(hubs)-1)/2        #全部富节点间都有连边的边数

    while swapcount<nswap and len(hubs_edges) < len_possible_edges:
        if tn >= max_tries:
            e=('尝试次数 (%s) 已超过允许的最大次数'%tn + '有效交换次数（%s)'%swapcount)
            print (e)
            break     
        tn += 1 
        
        u,y = random.sample(hubs,2)                       #任选两个富节点
        v = random.choice(list(G[u])) 
        x = random.choice(list(G[y]))
        if len(set([u,v,x,y]))==4:
            if G.degree()[v] > k or G.degree()[x] > k:
                continue   #另一端节点为非富节点         

        if (y not in G[u]) and (v not in G[x]):       #保证新生成的连边是原网络中不存在的边              
            G.add_edge(u,y)
            G.add_edge(x,v)
            
            G.remove_edge(u,v)
            G.remove_edge(x,y)
            hubs_edges.append((u,y)) #更新已存在富节点和富节点连边
            if connected==1:                            #判断是否需要保持联通特性，为1的话则需要保持该特性        
                if not nx.is_connected(G):              #保证网络是全联通的:若网络不是全联通网络，则撤回交换边的操作
                    G.add_edge(u,v)
                    G.add_edge(x,y)
                    G.remove_edge(u,y)
                    G.remove_edge(x,v)
                    hubs_edges.remove((u,y))
                    continue 
#        print(len(set(hubs_edges)))
                           
        if tn >= max_tries:
            print ('Maximum number of attempts (%s) exceeded '%tn)
            break
        swapcount=swapcount+1
    return G

def rich_club_break(G0, k=10, nswap=1, max_tries=100, connected=1):
    """
    富边：富节点和富节点的连边
    非富边：非富节点和非富节点的连边
    任选两条边(一条富边，一条非富边)，若富节点和非富节点间无连边，则断边重连
    达到最大尝试次数或无富边或无非富边，循环结束
    """
	# G0：待改变结构的网络
   # k 为富节点度值的门限值
	# nswap：是改变成功的系数，默认值为1
	# max_tries：是尝试改变的次数，默认值为100
	# connected：是否需要保证网络的联通特性，参数为1需要保持，参数为0不需要保持
    
    if not nx.is_connected(G0):
        raise nx.NetworkXError("非连通图，必须为连通图")
    if G0.is_directed():
        raise nx.NetworkXError("仅适用于无向图")
    if nswap > max_tries:
        raise nx.NetworkXError("交换次数超过允许的最大次数")
    if len(G0) < 3:
        raise nx.NetworkXError("节点数太少，至少要含三个节点")    
         
    tn = 0   #尝试次数
    swapcount = 0   #有效交换次数

    G = copy.deepcopy(G0)
    hubedges = []                                    #富边
    nothubedges = []                                 #非富边
    hubs = [e for e in G.nodes() if G.degree(e)>k] #全部富节点
    for e in G.edges():
        if e[0] in hubs and e[1] in hubs:
            hubedges.append(e)
        elif e[0] not in hubs and e[1] not in hubs:
            nothubedges.append(e)
            
    swapcount = 0
    while swapcount<nswap and hubedges and nothubedges:
        u,v = random.choice(hubedges)              #随机选一条富边
        x,y = random.choice(nothubedges)           #随机选一条非富边
        if len(set([u,v,x,y]))<4:
            continue
        if (y not in G[u]) and (v not in G[x]):    #保证新生成的连边是原网络中不存在的边              
            G.add_edge(u,y)
            G.add_edge(x,v)
            G.remove_edge(u,v)
            G.remove_edge(x,y)
            hubedges.remove((u,v))
            nothubedges.remove((x,y))
            if connected==1:
                if not nx.is_connected(G):                #不保持连通性，撤销
                    G.add_edge(u,v)
                    G.add_edge(x,y)
                    G.remove_edge(u,y)
                    G.remove_edge(x,v)
                    hubedges.append((u,v))
                    nothubedges.append((x,y))
                    continue
        if tn >= max_tries:
            print ('Maximum number of attempts (%s) exceeded '%tn)
            break
        swapcount=swapcount+1
    return G

def assort_mixing(G0, k=10, nswap=1, max_tries=100, connected=1):
    """
    随机选取两条边，四个节点，将这四个节点的度值从大到小排序，
    将度值较大的两个节点进行连接，度值较小的两个节点进行连接，
    最终形成了同配网络
    """
	# G0：待改变结构的网络
   # k 为富节点度值的门限值
	# nswap：是改变成功的系数，默认值为1
	# max_tries：是尝试改变的次数，默认值为100
	# connected：是否需要保证网络的联通特性，参数为1需要保持，参数为0不需要保持
    
    if not nx.is_connected(G0):
        raise nx.NetworkXError("非连通图，必须为连通图")
    if G0.is_directed():
        raise nx.NetworkXError("仅适用于无向图")
    if nswap > max_tries:
        raise nx.NetworkXError("交换次数超过允许的最大次数")
    if len(G0) < 3:
        raise nx.NetworkXError("节点数太少，至少要含三个节点")    
         
    tn = 0   #尝试次数
    swapcount = 0   #有效交换次数  
    
    G = copy.deepcopy(G0)
    keys = list(dict(G.degree()).keys())
    degrees= list(dict(G.degree()).values())
#    keys,degrees = zip(*G.degree().items()) 
    cdf = nx.utils.cumulative_distribution(degrees) 
   
    cnt = 0
    while swapcount < nswap:       #有效交换次数小于规定交换次数         
        tn += 1 

        # 在保证度分布不变的情况下，随机选取两条连边u-v，x-y
        (ui,xi)=nx.utils.discrete_sequence(2,cdistribution=cdf)
        if ui==xi:
            continue
        u = keys[ui] 
        x = keys[xi]
        v = random.choice(list(G[u]))
        y = random.choice(list(G[x]))
        
        if len(set([u,v,x,y]))<4:
            continue
        sortednodes = list(zip(*sorted(dict(G.degree([u,v,x,y])).items(),key=lambda d:d[1],reverse=True)))[0]
        if (sortednodes[0] not in G[sortednodes[1]]) and (sortednodes[2] not in G[sortednodes[3]]):    
        #保证新生成的连边是原网络中不存在的边
              
            G.add_edge(sortednodes[0],sortednodes[1])   #连新边 
            G.add_edge(sortednodes[2],sortednodes[3])
            G.remove_edge(x,y)                          #断旧边
            G.remove_edge(u,v)
            cnt+=1
            if connected==1:
                if not nx.is_connected(G):  
                    G.remove_edge(sortednodes[0],sortednodes[1]) 
                    G.remove_edge(sortednodes[2],sortednodes[3])
                    G.add_edge(x,y) 
                    G.add_edge(u,v)
                    continue
        if tn >= max_tries:
            e=('Maximum number of swap attempts (%s) exceeded '%tn + 'before desired swaps achieved (%s).'%nswap)
            print(e)
            break
        swapcount+=1
    print(cnt)        
    print(swapcount)    
    return G


def disassort_mixing(G0, k=10, nswap=1, max_tries=100, connected=1):
    """
    随机选取两条边，四个节点，将这四个节点的度值从大到小排序，
    将度值差异较大的两个节点进行连接，第一和第四两个节点相连，
    将度值差异较小的两个节点进行连接，第二和第三两个节点相连
    最终形成了异配网络
    """
	# G0：待改变结构的网络
   # k 为富节点度值的门限值
	# nswap：是改变成功的系数，默认值为1
	# max_tries：是尝试改变的次数，默认值为100
	# connected：是否需要保证网络的联通特性，参数为1需要保持，参数为0不需要保持
    
    if not nx.is_connected(G0):
        raise nx.NetworkXError("非连通图，必须为连通图")
    if G0.is_directed():
        raise nx.NetworkXError("仅适用于无向图")
    if nswap > max_tries:
        raise nx.NetworkXError("交换次数超过允许的最大次数")
    if len(G0) < 3:
        raise nx.NetworkXError("节点数太少，至少要含三个节点")    
         
    tn = 0   #尝试次数
    swapcount = 0   #有效交换次数  
    
    G = copy.deepcopy(G0)
    keys = list(dict(G.degree()).keys())
    degrees= list(dict(G.degree()).values())
#    keys,degrees = zip(*G.degree().items()) 
    cdf = nx.utils.cumulative_distribution(degrees) 
    cnt = 0
    while swapcount < nswap:       #有效交换次数小于规定交换次数         
        tn += 1 

        # 在保证度分布不变的情况下，随机选取两条连边u-v，x-y
        (ui,xi)=nx.utils.discrete_sequence(2,cdistribution=cdf)
        if ui==xi:
            continue
        u = keys[ui] 
        x = keys[xi]
        v = random.choice(list(G[u]))
        y = random.choice(list(G[x]))
        
        if len(set([u,v,x,y]))<4:
            continue
        sortednodes = list(zip(*sorted(dict(G.degree([u,v,x,y])).items(),key=lambda d:d[1],reverse=True)))[0]
        if (sortednodes[0] not in G[sortednodes[3]]) and (sortednodes[1] not in G[sortednodes[2]]):    
        #保证新生成的连边是原网络中不存在的边
              
            G.add_edge(sortednodes[0],sortednodes[3])   #连新边 
            G.add_edge(sortednodes[1],sortednodes[2])
            G.remove_edge(x,y)                          #断旧边
            G.remove_edge(u,v)
            cnt += 1
            if connected==1:
                if not nx.is_connected(G):  
                    G.remove_edge(sortednodes[0],sortednodes[3]) 
                    G.remove_edge(sortednodes[1],sortednodes[2])
                    G.add_edge(x,y) 
                    G.add_edge(u,v)
                    continue
        if tn >= max_tries:
            e=('Maximum number of swap attempts (%s) exceeded '%tn + 'before desired swaps achieved (%s).'%nswap)
            print(e)
            break
        swapcount+=1
    print(cnt)        
    print(swapcount)   
    return G
    
 
#下面的程序暂时未修改   
def random_1kd(G0, nswap=1, max_tries=100):  #有向网络基于随机断边重连的1阶零模型
    """
    随机取两条边 u->v 和 x->y, 若u->y,x->v不存在, 断边重连
    """
    if not G0.is_directed():
        raise nx.NetworkXError("Graph not directed")
    if nswap>max_tries:
            raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G0) < 4:
        raise nx.NetworkXError("Graph has less than four nodes.")
    G = copy.deepcopy(G0)
    n = 0
    swapcount = 0
    while swapcount < nswap:
        (u,v),(x,y) = random.sample(G.edges(),2)
        if len(set([u,v,x,y]))<4:            
            continue
        if (x,v) not in G.edges() and (u,y) not in G.edges():  #断边重连
            G.add_edge(u,y)
            G.add_edge(x,v)
            G.remove_edge(u,v)
            G.remove_edge(x,y)
            swapcount+=1
        if n >= max_tries:
            e=('Maximum number of swap attempts (%s) exceeded '%n + 'before desired swaps achieved (%s).'%nswap)
            print (e)
            break
        n+=1
    return G
 