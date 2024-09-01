import numpy as np
import matplotlib.pyplot as plt

# 参数设置
N = 69  # 链的长度
T = 240  # 总时间
h = 0.01  # 步长
timesteps = int(T / h)  # 总步数

# 非对称矩阵 A 的参数
a1, a2, a3 = 1, 1, 1

# 初始化非对称矩阵 A
A = np.zeros((N, N))
A[0,1] = a3
A[0,2] = -a1
for i in range(1,N-1):
    if i % 2 == 1:
        A[i,i-1] = -a3
        A[i,i+1] = a2
    if i % 2 == 0:
        A[i,i-2] = a1
        A[i,i-1] = - a2
        A[i,i+1] = a3
        A[i,i+2] = -a1
A[N-1,N-3] = a1
A[N-1,N-2] = -a2

print("Antisymmetric matrix A:")
print(A)

# 初始质量分布
m = np.full((N,), 0.01)
m[5] = m[11] = 0.01 * 0.45
m[6] = m[10] = 0.15 * 0.45
m[7] = m[9] = 0.05 * 0.45
m[8] = 0.3 * 0.45
m[21] = m[27] = 0.005 * 0.45
m[22] = m[26] = 0.07 * 0.45
m[23] = m[25] = 0.015 * 0.45
m[24] = 0.1 * 0.45

# 计算导数的函数
def dm_dt(m,A):
    dm_dt = np.zeros(N)
    for i in range(N):
        dm_dt[i] = m[i] * np.dot(A[i],m)
    return dm_dt
print(dm_dt(m,A))

# RK4 方法
def rk4_step(m, h, A):
    k1 = h * dm_dt(m, A)
    k2 = h * dm_dt(m + 0.5  * k1, A)
    k3 = h * dm_dt(m + 0.5  * k2, A)
    k4 = h * dm_dt(m + k3, A)
    return m + (1 / 6) * (k1 + 2*k2 + 2*k3 + k4)

m_t = np.zeros((timesteps + 1, N)) #定义m_t矩阵用于储存任意时刻的质量分布
m_t[0] = m #存入初始质量分布

#进行时间计算
for t in range(timesteps):
    m = rk4_step(m, h, A)
    m_t[t + 1] = m

#绘制质量在不同时间点的函数图
a = np.arange(1, N + 1)
plt.figure(figsize=(12, 6))
plt.plot(a, m_t[0, :], label='t=0',color='green') #绘制t=0的函数图像
plt.plot(a, m_t[int(120/h), :], label='t=120',color='blue') #绘制t=120的函数图像
plt.plot(a, m_t[int(240/h), :], label='t=240',color='red') #绘制t=240的函数图像
plt.xlabel('number')
plt.ylabel('mass')
plt.title('time-mass' )
plt.legend()
plt.grid()
plt.show()

#绘制3D图像
fig = plt.figure(dpi=300)
ax = fig.add_subplot(projection="3d")
for i in [240, 120, 0]:
    ax.plot(a, i * np.ones(N), m_t[int(i/h), :])
ax.set_xlabel("number")
ax.set_ylabel("time")
ax.set_zlabel("mass")
ax.set_yticks([1, 60, 120, 180, 240])
plt.title(f"time-mass 3D")
plt.legend()
plt.show()