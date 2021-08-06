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

data = [0, 10, 20, 30, 60, 100, 200, 307]
data1 = [0, 10, 20, 30, 100, 150, 200, 240, 261]
clu = [0.0650, 0.0495, 0.0175, -0.0264, -0.0719, -0.2036, -0.4281, -0.6683]
clu1 = [0.0650, 0.0907, 0.1455, 0.1646, 0.3638, 0.4390, 0.5281, 0.5643, 0.6846]

# 0
# 5   0.0867
# 10
# 20  0.0
#生成折线图
plt.plot(data,clu,label="",color='b')
plt.plot(data1,clu1,label="",color='b')

#设置横坐标说明
plt.xlabel("Number of exchanges")
#设置纵坐标说明
# plt.ylabel("Rich club coefficient")
plt.ylabel("Degree correlation coefficient")
#添加标题
plt.title("BA and ws Internet Weaken")
#显示网格
plt.grid(True)
# plt.legend()
# plt.show()
plt.savefig("./rich_coefficient.png")