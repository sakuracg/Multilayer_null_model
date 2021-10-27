# @Time : Created on 2021/7/30 20:16 
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
with open('../data/EU-Air各种度量结果/res_purity_new_2.txt', 'r') as f:
    for line in f:
        str = line.strip().split(' ')
        data_purity.append(float(str[1]))
with open('../data/EU-Air各种度量结果/res_f_measure_new.txt', 'r') as f:
    for line in f:
        str = line.strip().split(' ')
        data_f_measure.append(float(str[1]))

with open('../data/EU-Air各种度量结果/res_edge_jaccard.txt', 'r') as f:
    for line in f:
        str = line.strip().split(' ')
        data_edge_jaccard.append(float(str[1]))

with open('../data/EU-Air各种度量结果/res_cos_val.txt', 'r') as f:
    for line in f:
        str = line.strip().split(' ')
        data_cos_val.append(float(str[1]))

with open('../data/EU-Air各种度量结果/res_pearson_val.txt', 'r') as f:
    for line in f:
        str = line.strip().split(' ')
        data_pearson_val.append(float(str[1]))

with open('../data/EU-Air各种度量结果/res_degree_mean_val.txt', 'r') as f:
    for line in f:
        str = line.strip().split(' ')
        data_degree_mean_val.append(float(str[1]))

with open('../data/EU-Air各种度量结果/res_cc_mean_val.txt', 'r') as f:
    for line in f:
        str = line.strip().split(' ')
        data_cc_mean_val.append(float(str[1]))

plt.gcf().subplots_adjust(left=0.25)
plt.boxplot((data_purity,data_f_measure, data_edge_jaccard,
             data_cos_val, data_pearson_val, data_degree_mean_val,
             data_cc_mean_val),
            labels=["purity","f_measure", "edge_jaccard",
                    "degree_cos_val", "degree_pearson_val",
                    "degree_mean_val", "cc_mean_val"],
            vert=False, flierprops={'markerfacecolor':'r', 'color':'g'})
plt.title('EU-Air')
plt.savefig("./data/EU-Air各种度量结果/res.png")
plt.show()