import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

dx = 2*np.pi/20
dy = 2*np.pi/20
x = np.arange(0, 2*np.pi+dx, dx)
y = np.arange(0, 2*np.pi+dy, dy)


mesh1 = np.zeros((len(x), len(y)))
# B.C.
mesh1[0] = 0
mesh1[-1] = 0
mesh1[:,0] = np.sin(2*x)+np.sin(5*x)+np.sin(7*x)
mesh1[:,-1] = 0

mesh2 = mesh1.copy()
mesh3 = mesh1.copy()

# Jacobi
max_iter = 200
for _ in range(max_iter):
    mesh1[1:-1,1:-1] = 0.25*(mesh1[:-2,1:-1]+mesh1[2:,1:-1]+mesh1[1:-1,:-2]+mesh1[1:-1,2:])

# Gauss-Seidel
for _ in range(max_iter):
    for i in range(1, len(x)-1):
        for j in range(1, len(y)-1):
            mesh2[i,j] = 0.25*(mesh2[i-1,j]+mesh2[i+1,j]+mesh2[i,j-1]+mesh2[i,j+1])

# SOR
omega = 1.5
for _ in range(max_iter):
    for i in range(1, len(x)-1):
        for j in range(1, len(y)-1):
            mesh3[i,j] = (1-omega)*mesh3[i,j]+omega*0.25*(mesh3[i-1,j]+mesh3[i+1,j]+mesh3[i,j-1]+mesh3[i,j+1])

x = np.arange(0, 2*np.pi+dx, dx)
y = np.arange(0, 2*np.pi+dy, dy)
X, Y = np.meshgrid(x, y)

fig = plt.figure()
# 创建3d图形的两种方式
# 将figure变为3d
ax = Axes3D(fig,auto_add_to_figure=False)
fig.add_axes(ax)
# plt.imshow(mesh1)
ax.plot_surface(X, Y, mesh1, rstride = 1, cstride = 1, cmap = plt.get_cmap('rainbow'))
plt.show()