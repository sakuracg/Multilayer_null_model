# @Time : Created on 2021/8/3 16:43 
# @File : 将EU—Air总数据分成各个层.py 
# @Author : xiaozhi
# 准备数据，把欧盟航空数据都分散到各个文件
with open('./data/Auger_Network原始数据.txt', 'r') as f:
    for line in f:
        str_arr = line.strip().split(' ')
        layer = str_arr[0]
        with open('./data/Auger-txt/layer' + layer + '.txt', 'a') as file:
            file.write(str_arr[1] + ' ' + str_arr[2] + '\n')