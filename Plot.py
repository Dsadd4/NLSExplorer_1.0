import numpy as np
import matplotlib.pyplot as plt

# 假设有三个任务，每个任务的向量在五个维度上的贡献不同
tasks = ['Task 1', 'Task 2', 'Task 3']
dimensions = ['Dim 1', 'Dim 2', 'Dim 3', 'Dim 4', 'Dim 5']

# 随机生成贡献数据 (可以替换为你的实际数据)
data = np.random.rand(3, 5)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 创建网格以绘制柱状图
xpos, ypos = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)

# 每个柱体的宽度
dx = dy = 0.8
dz = data.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)

# 设置轴标签
ax.set_xlabel('Tasks')
ax.set_ylabel('Dimensions')
ax.set_zlabel('Contribution')

# 设置轴刻度
ax.set_xticks(np.arange(len(tasks)) + dx / 2)
ax.set_xticklabels(tasks)
ax.set_yticks(np.arange(len(dimensions)) + dy / 2)
ax.set_yticklabels(dimensions)
plt.savefig('3d_a.svg')
plt.show()
