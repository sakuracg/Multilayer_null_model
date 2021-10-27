# 富人俱乐部系数

import networkx as nx
import matplotlib.pyplot as plt
#
# fh = open("IEEE300.txt", 'rb')
# G = nx.read_edgelist(fh)
# fh.close()
# G0 = G.to_undirected()
#
#
# nx.draw(G0,with_labels=True)
# plt.show()

# data = [0, 10, 20, 30, 60, 100, 200, 307]
# data1 = [0, 10, 20, 30, 100, 150, 200, 240, 261]
# clu = [0.0650, 0.0495, 0.0175, -0.0264, -0.0719, -0.2036, -0.4281, -0.6683]
# clu1 = [0.0650, 0.0907, 0.1455, 0.1646, 0.3638, 0.4390, 0.5281, 0.5643, 0.6846]
#
# # 0
# # 5   0.0867
# # 10
# # 20  0.0
# #生成折线图
# plt.plot(data,clu,label="",color='b')
# plt.plot(data1,clu1,label="",color='b')
#
# #设置横坐标说明
# plt.xlabel("Number of exchanges")
# #设置纵坐标说明
# # plt.ylabel("Rich club coefficient")
# plt.ylabel("Degree correlation coefficient")
# #添加标题
# plt.title("BA and ws Internet Weaken")
# #显示网格
# plt.grid(True)
# # plt.legend()
# # plt.show()
# plt.savefig("./rich_coefficient.png")

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

sns.set(style="ticks", palette="muted")
sns.set_context("notebook", font_scale=0.8)
plt.figure(figsize=(5.5, 4), dpi=144, facecolor=None, edgecolor=None, frameon=False)
plt.rcParams['font.sans-serif'] = 'SimHei'  # 调节字体显示为中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
plt.rcParams['font.family'] = 'sans-serif'

name_list = ['层间度相关系数不变', '层间度相关系数增加', '层间度相关系数减小', '层间度相关系数随机']
cas_no_change = [0.462, 0.491, 0.491]
cas_up = [0.688, 0.680, 0.680]
calpu_down = [-0.549, -0.526, -0.526]
calpu_range = [0.022, -0.006, 0.026]
# cas_no_change = [0.462, 0.688, -0.549, 0.022]
# cas_up = [0.491, 0.680, -0.526, -0.006]
# calpu_down = [0.491, 0.680, -0.526, 0.026]
# x = list(range(len(cas_no_change)))
# total_width, n = 0.8, 4
# width = total_width / n

plt.figure(figsize=(8,6))
plt.subplot(221)
plt.barh(range(len(cas_no_change)), cas_no_change, height=0.4, color=['skyblue','orange','salmon'])
plt.xlabel('(a)层间度相关系数不变')
plt.yticks([])
plt.subplot(222)
plt.barh(range(len(cas_up)), cas_up, height=0.4, color=['skyblue','orange','salmon'])
plt.xlabel('(b)层间度相关系数增加')
plt.yticks([])
plt.subplot(223)
plt.barh(range(len(calpu_down)), calpu_down, height=0.4, color=['skyblue','orange','salmon'])
plt.xlabel('(c)层间度相关系数减小')
plt.yticks([])
ax = plt.subplot(224)
plt.barh(range(len(calpu_range)), calpu_range, height=0.4, color=['skyblue','orange','salmon'])
plt.xlabel('(d)层间度相关系数随机')
plt.yticks([])
red_patch1 = mpatches.Patch(color='skyblue', label='短信和电话层内度相关系数不变')
red_patch2 = mpatches.Patch(color='orange', label='短信和电话层内度相关系数都增大')
red_patch3 = mpatches.Patch(color='salmon', label='短信和电话层内度相关系数都减小')
ax.legend(handles=[red_patch1, red_patch2, red_patch3])
plt.tight_layout(pad=0.4, w_pad=2, h_pad=2)
# plt.bar(x, cas_no_change, width=width, label='短信和电话层内度相关系数不变', fc='y')
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, cas_up, width=width, label='短信和电话层内度相关系数都增大', tick_label=name_list, fc='r')
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, calpu_down, width=width, label='短信和电话层内度相关系数都减小(取绝对值)', tick_label=name_list, fc='b')
# plt.legend('短信和电话层内度相关系数不变', '短信和电话层内度相关系数都增大', '短信和电话层内度相关系数都减小')
plt.show()