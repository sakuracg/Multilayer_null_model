#!/usr/bin/env python
# coding=utf-8
# @Time : Created on 2021/6/21 16:03
# @File : D_value_test.py 
# @Author : xiaozhi
import ling_model as lm
import time
import numpy as np

jizhun_path1="./20201001.csv"
jizhun_path2="./20201002.csv"
# a_addr,b_addr, filename_single, filename_multi
start_time = time.time() * 1000
d_value = lm.D_value(jizhun_path1, jizhun_path2, 'single_01', 'multi_01')
end_time = time.time() * 1000
print(d_value)
print(end_time - start_time)