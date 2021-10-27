# @Time : Created on 2021/7/30 20:15 
# @File : 相似度绘制箱线图—EUAir.py
# @Author : xiaozhi
import matplotlib.pyplot as plt

data_purity = []
data_f_measure = []
data_edge_jaccard = []
data_cos_val = []
data_pearson_val = []
data_degree_mean_val = []
data_cc_mean_val = []

# 从文件中读入数据
with open('../data/Auger各种度量结果/res_purity_new.txt', 'r') as f:
    for line in f:
        str = line.strip().split(' ')
        data_purity.append(float(str[1]))
with open('../data/Auger各种度量结果/res_f_measure_new.txt', 'r') as f:
    for line in f:
        str = line.strip().split(' ')
        data_f_measure.append(float(str[1]))

with open('../data/Auger各种度量结果/res_edge_jaccard.txt', 'r') as f:
    for line in f:
        str = line.strip().split(' ')
        data_edge_jaccard.append(float(str[1]))

with open('../data/Auger各种度量结果/res_cos_val.txt', 'r') as f:
    for line in f:
        str = line.strip().split(' ')
        data_cos_val.append(float(str[1]))

with open('../data/Auger各种度量结果/res_pearson_val.txt', 'r') as f:
    for line in f:
        str = line.strip().split(' ')
        data_pearson_val.append(float(str[1]))

with open('../data/Auger各种度量结果/res_degree_mean_val.txt', 'r') as f:
    for line in f:
        str = line.strip().split(' ')
        data_degree_mean_val.append(float(str[1]))

with open('../data/Auger各种度量结果/res_cc_mean_val.txt', 'r') as f:
    i = 1
    for line in f:
        str = line.strip().split(' ')
        data_cc_mean_val.append(float(str[1]))
print(len(data_purity))
x = []
for i in range(1, 16):
    for j in range(i+1, 17):
        x.append(f'({i},{j})')
print(len(x))
# plt.gcf().subplots_adjust(left=0.25)
# plt.boxplot((data_purity,data_f_measure, data_edge_jaccard,
#              data_cos_val, data_pearson_val, data_degree_mean_val,
#              data_cc_mean_val),
#             labels=["purity","f_measure", "edge_jaccard",
#                     "degree_cos_val", "degree_pearson_val",
#                     "degree_mean_val", "cc_mean_val"],
#             vert=False, flierprops={'markerfacecolor':'r', 'color':'g'})
# plt.title("Auger")
# plt.savefig("./data/Auger各种度量结果/res.png")

# plt.scatter(x[:15], data_purity[:15], marker = 'o',color = 'red', s = 40 ,label = 'purity')
# plt.scatter(x[:15], data_f_measure[:15], marker = 'o',color = 'blue', s = 40 ,label = 'f_measure')
# plt.scatter(x[:15], data_edge_jaccard[:15], marker = 'o',color = 'green', s = 40 ,label = 'edge_jaccard')
# plt.scatter(x[:15], data_cos_val[:15], marker = 'o',color = 'yellow', s = 40 ,label = 'degree_cos_val')
fig, ax = plt.subplots()
# plt.plot(x[:15], data_purity[:15], marker = 'o',color = 'red', label = 'purity')
# plt.plot(x[:15], data_f_measure[:15], marker = 'o',color = 'blue', label = 'f_measure')
# plt.plot(x[:15], data_edge_jaccard[:15], marker = 'o',color = 'green',label = 'edge_jaccard')
# plt.plot(x[:15], data_cos_val[:15], marker = 'o',color = 'yellow',label = 'degree_cos_val')
plt.plot(x[65:75], data_purity[65:75], marker = 'o',color = 'red', label = 'purity')
plt.plot(x[65:75], data_f_measure[65:75], marker = 'o',color = 'blue', label = 'f_measure')
plt.plot(x[65:75], data_edge_jaccard[65:75], marker = 'o',color = 'green',label = 'edge_jaccard')
plt.plot(x[65:75], data_cos_val[65:75], marker = 'o',color = 'yellow',label = 'degree_cos_val')
# plt.plot(x, data_purity, marker = 'o',color = 'red', label = 'purity')
# plt.plot(x, data_f_measure, marker = 'o',color = 'blue', label = 'f_measure')
# plt.plot(x, data_edge_jaccard, marker = 'o',color = 'green',label = 'edge_jaccard')
# plt.plot(x, data_cos_val, marker = 'o',color = 'yellow',label = 'degree_cos_val')
fig.set_figwidth(9)
plt.legend()
# plt.show()
plt.savefig("./data/Auger各种度量结果/6层比16.png")