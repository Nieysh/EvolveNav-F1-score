
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

x = np.array([1, 2, 3, 4, 5, 6])
SIFT = np.array([12.9749694, 13.9357018, 14.7440844, 16.482254, 18.720203, 18.687582])
AKAZE = np.array([12.1044724, 12.9757383, 13.9754183, 15.686206, 18.367847, 19.144531])
# ORB = np.array([12.0205495, 12.6509762, 13.1876223, 14.380781, 16.004548, 17.9298])

# label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
# color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
# 线型：-  --   -.  :    ,
# marker：.  ,   o   v    <    *    +    1

plt.figure(figsize=(10, 5))  # 设置图像大小
plt.grid(linestyle="--")  # 设置背景网格线为虚线
ax = plt.gca()  # gca就是get current axes 获取当前坐标轴
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框

plt.plot(x, SIFT, marker='o', color="blue", label="SIFT", linewidth=1.5)
plt.plot(x, AKAZE, marker='o', color="green", label="AKAZE", linewidth=1.5)
# plt.plot(x, ORB, marker='o', color="red", label="ORB", linewidth=1.5)

group_labels = ['Top 0-5%', 'Top 5-10%', 'Top 10-20%', 'Top 20-50%', 'Top 50-70%', ' Top 70-100%']  # x轴刻度的标识
plt.xticks(x, group_labels, fontsize=12, fontweight='bold')  # 默认字体大小为10
plt.yticks(fontsize=12, fontweight='bold')
plt.title("Example", fontsize=13, fontweight='bold')  # 默认字体大小为12
plt.xlabel("Performance Percentile", fontsize=13, fontweight='bold')
plt.ylabel("Accuracy", fontsize=13, fontweight='bold')
plt.xlim(0.9, 6.1)  # 设置x轴的范围
plt.ylim(10, 22)

# plt.legend()          #显示各曲线的图例
plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细

plt.savefig('./filename.svg', format='svg')  # 建议保存为svg格式,再用在线转换工具转为矢量图emf后插入word中
plt.show()
