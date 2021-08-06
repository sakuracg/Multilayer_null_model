# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:25:58 2020

@author: Dell
"""

import networkx as nx
import copy
import random   
import math                                       
import numpy as np
from operator import itemgetter

def rich_club_creat(G0, k, max_tries=100):
    """
    节点的强度 = 节点所有连边的权重值之和
    根据节点的强度将所有节点分成富节点和非富节点
    任选两条边(富节点和非富节点的连边)，若富节点间无连边，非富节点间无连边，则断边重连
    达到最大尝试次数或全部富节点间都有连边，循环结束
    强度大于k的节点为富节点
    """
    if len(G0) < 4:
        raise nx.NetworkXError("Graph has less than four nodes.")
    G = copy.deepcopy(G0)
    edges = G.edges()
    nodes = G.nodes()
    rnodes = [e for e in nodes if G.degree(e,weight='weight')>=k]     #全部富节点
    len_redges = len([e for e in edges if e[0] in rnodes and e[1] in rnodes]) #网络中已有的富节点和富节点的连边数
    len_possible_redges = len(rnodes)*(len(rnodes)-1)/2        #全部富节点间都有连边的边数
    n = 0
    while len_redges < len_possible_redges:
        u,x = random.sample(rnodes,2)                       #任选两个富节点
        candidate_v = [e for e in list(G[u]) if G.degree(e,weight='weight')<k]
        candidate_y = [e for e in list(G[x]) if G.degree(e,weight='weight')<k]
        if candidate_v != [] and candidate_y != []:
            v = random.choice(candidate_v) #非富节点
            y = random.choice(candidate_y)          
            if len(set([u,v,x,y])) < 4: #防止自环           
                continue
            if (x not in G[u]) and (y not in G[v]):
                G.add_edges_from([(u,x),(v,y)])
                G[u][x]['weight'] = G[u][v]['weight']  
                G[v][y]['weight'] = G[x][y]['weight'] 
                G.remove_edges_from([(u,v),(x,y)])
                len_redges += 1
        if n >= max_tries:
            print ('Maximum number of attempts (%s) exceeded '%n)
            break
        n += 1
    return G


def rich_club_break(G0, k, max_tries=100):
    """
    富边：富节点和富节点的连边
    非富边：非富节点和非富节点的连边
    任选两条边(一条富边，一条非富边)，若富节点和非富节点间无连边，则断边重连
    达到最大尝试次数或无富边或无非富边，循环结束
    """    
    if len(G0) < 4:
        raise nx.NetworkXError("Graph has less than four nodes.")
    G = copy.deepcopy(G0)
    edges = G.edges()
    nodes = G.nodes()
    rnodes = [e for e in nodes if G.degree(e,weight='weight')>=k]     #全部富节点
    redges = [e for e in edges if e[0] in rnodes and e[1] in rnodes] #网络中已有的富节点和富节点的连边
    pedges = [e for e in edges if e[0] not in rnodes and e[1] not in rnodes] #网络中已有的非富节点和非富节点的连边
#    len_redges = len(redges)
#    len_pedges = len(pedges)
    n = 0
    while redges and pedges:
        u,v = random.choice(redges)              #随机选一条富边
        x,y = random.choice(pedges)           #随机选一条非富边              
        if (x,u) not in edges and (u,x) not in edges and (v,y) not in edges and (y,v) not in edges:              
            G.add_edges_from([(u,x),(v,y)])
            G[u][x]['weight'] = G[u][v]['weight']  
            G[v][y]['weight'] = G[x][y]['weight'] 
            G.remove_edges_from([(u,v),(x,y)])
            edges.extend([(u,x),(v,y)])
            edges.remove((u,v))
            edges.remove((x,y))
            redges.remove((u,v))
            pedges.remove((x,y))
        if n >= max_tries:
            print ('Maximum number of attempts (%s) exceeded '%n)
            break
        n += 1
    return G            
def assort_mixing(G0, nswap=1, max_tries=100):
    """
    让强度大的节点和强度大的节点相连
    """       
    if nswap>max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G0) < 4:
        raise nx.NetworkXError("Graph has less than four nodes.")
    G = copy.deepcopy(G0)
    n=0
    swapcount=0
#    nodes = G.nodes()
    edges = G.edges()
    while swapcount < nswap:
        (u,v),(x,y) = random.sample(edges,2) #任选两条边
        if len(set([u,v,x,y]))<4:
            continue
        a,b,c,d = zip(*sorted(G.degree([u,v,x,y],weight='weight').items(),key=lambda d:d[1],reverse=True))[0]
        if (a,b) not in edges and (b,a)not in edges and (c,d) not in edges and (d,c)not in edges:
            G.add_edges_from([(a,b),(c,d)])
            G[a][b]['weight'] = G[u][v]['weight']  
            G[c][d]['weight'] = G[x][y]['weight'] 
            G.remove_edges_from([(u,v),(x,y)])
            edges.extend([(a,b),(c,d)])
            edges.remove((u,v))
            edges.remove((x,y))
            swapcount += 1
        if n >= max_tries:
            e=('Maximum number of swap attempts (%s) exceeded '%n +
            'before desired swaps achieved (%s).'%nswap)
            print (e)
            break
        n += 1       
    return G
    
    
def disassort_mixing(G0, nswap=1, max_tries=100):     #异配
    """
    让强度大的节点和强度小的节点相连
    """       
    if nswap>max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G0) < 4:
        raise nx.NetworkXError("Graph has less than four nodes.")
    G = copy.deepcopy(G0)
    n=0
    swapcount=0
#    nodes = G.nodes()
    edges = G.edges()
    while swapcount < nswap:
        (u,v),(x,y) = random.sample(edges,2) #任选两条边
        if len(set([u,v,x,y]))<4:
            continue
        a,b,c,d = zip(*sorted(G.degree([u,v,x,y],weight='weight').items(),key=lambda d:d[1],reverse=True))[0]
        if (a,d) not in edges and (d,a) not in edges and (c,b) not in edges and (b,c)not in edges:
            G.add_edges_from([(a,d),(b,c)])
            G[a][d]['weight'] = G[u][v]['weight']  
            G[b][c]['weight'] = G[x][y]['weight'] 
            G.remove_edges_from([(u,v),(x,y)])
            edges.extend([(a,d),(b,c)])
            edges.remove((u,v))
            edges.remove((x,y))
            swapcount += 1
        if n >= max_tries:
            e=('Maximum number of swap attempts (%s) exceeded '%n +
            'before desired swaps achieved (%s).'%nswap)
            print (e)
            break
        n += 1       
    return G