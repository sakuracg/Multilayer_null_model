from pymnet import *
import numpy as np
import random
import multinetx as mx
from ling_model import create_MultilayerGraph, write_data_toDat, random_25_model, random_3_model

N = 10

mg = create_MultilayerGraph(N, p_g1=0.4, p_g2=0.1, seed_g1=218, seed_g2=256, intra_layer_edges_weight=2, inter_layer_edges_weight=3)
write_data_toDat('data4-1.dat', mg, N, weight=None)

new_random_mg = random_3_model(mg, N, nswap=1, max_tries=1000)
# new_random_mg = random_2_model(mg, N, nswap=1, max_tries=1000)
write_data_toDat('data4-2.dat', new_random_mg, N, weight=None)

nt1 = read_ucinet('data4-1.dat')
fig = draw(nt1,
            show=False,
            layout="spring",
            layerColorRule={},
            defaultLayerColor="#EFF1F1",
            edgeColorRule={},
            defaultEdgeColor='#808080',
            nodeLabelDict={(0, 0):'A',(1, 0):'B',(2, 0):'C',(3, 0):'D',(4, 0):'E',(5, 0):'F',(6, 0):'G',(7, 0):'H',
                          (0, 1):'A',(1, 1):'B',(2, 1):'C',(3, 1):'D',(4, 1):'E',(5, 1):'F',(6, 1):'G',(7, 1):'H'},
            layerLabelDict={0:'α', 1:'β'},
            layerPadding=0.1)
fig.savefig('4-1.pdf')

nt2 = read_ucinet('data4-2.dat')
fig = draw(nt2,
            show=False,
            layout="spring",
            layerColorRule={},
            defaultLayerColor="#EFF1F1",
            edgeColorRule={},
            defaultEdgeColor='#808080',
            nodeLabelDict={(0, 0):'A',(1, 0):'B',(2, 0):'C',(3, 0):'D',(4, 0):'E',(5, 0):'F',(6, 0):'G',(7, 0):'H',
                          (0, 1):'A',(1, 1):'B',(2, 1):'C',(3, 1):'D',(4, 1):'E',(5, 1):'F',(6, 1):'G',(7, 1):'H'},
            layerLabelDict={0:'α', 1:'β'},
            layerPadding=0.1)
fig.savefig('4-2.pdf')

