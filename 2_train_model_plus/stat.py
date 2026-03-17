import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
matplotlib.use('Agg')
font_path_bold = './ARIALBD.TTF'
font_prop_bold = fm.FontProperties(fname=font_path_bold)
fm.fontManager.addfont(font_path_bold)

font_path = './ARIAL.TTF'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.size'] = 16
plt.rcParams['axes.unicode_minus']=False

# 读取数据
data = pd.read_csv("./evaluation/evaluate_with_pos_data.csv")

# 提取所需列
x = data["Positive MSE"]
y = data["Positive R"]
acc = data["ACC"]  # 读取 ACC 列

# 筛选红色点（ACC >= 0.9）
red_mask = acc >= 0.9
gray_mask = ~red_mask  # 其余点为灰色

# 绘制散点图
plt.figure(figsize=(7, 6))

# 先绘制灰色点
plt.scatter(x[gray_mask], y[gray_mask], color=(188/255, 190/255, 193/255), 
            s=8, linewidths=0.05, edgecolors='gray', zorder=1)

# 再绘制红色点
plt.scatter(x[red_mask], y[red_mask], color=(239/255, 118/255, 122/255), 
            s=8, linewidths=0.05, edgecolors='gray', zorder=1)

plt.xlabel("MSE")
plt.ylabel("r")
plt.axvline(x=0.8, color='black', linestyle='--', linewidth=1.2, zorder=2)
plt.axhline(y=0.75, color='black', linestyle='--', linewidth=1.2, zorder=2)

plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
# 保存图像
plt.savefig("./stat/mse_r.pdf")


# # 对 ACC 列进行排序
# data_sorted = data.sort_values(by="ACC", ascending=True).reset_index(drop=True)

# # 生成排名
# x = data_sorted.index + 1  # 排名从 1 开始
# y = data_sorted["ACC"]

# # 找到 ACC = 0.9 时的排名（最近的一个）
# threshold = 0.9
# closest_rank = (y >= threshold).idxmax() + 1  # 找到第一个满足条件的索引并转换为排名

# # 绘制散点图
# plt.figure(figsize=(7, 6))

# # 散点图（全部点默认灰色）
# plt.scatter(x, y, color=(188/255, 190/255, 193/255), s=8, linewidths=0.05, edgecolors='gray', zorder=1)

# # 设定高 ACC 的点为红色（可根据实际情况调整阈值）
# red_mask = y >= threshold
# plt.scatter(x[red_mask], y[red_mask], color=(239/255, 118/255, 122/255), s=8, linewidths=0.05, edgecolors='gray', zorder=3)

# # 添加阈值虚线
# plt.axhline(y=threshold, color='black', linestyle='--', linewidth=1.2, zorder=2)

# # 添加竖线（ACC = 0.9 对应的排名）
# plt.axvline(x=closest_rank, color='black', linestyle='--', linewidth=1.2, zorder=2)

# # 坐标轴标签
# plt.xlabel("Rank")
# plt.ylabel("ACC")
# plt.xticks([1, closest_rank, 400, 600, 800, len(data_sorted)])

# # 保存图像
# plt.savefig("./stat/acc_rank.pdf")