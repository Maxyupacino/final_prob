import numpy as np
import matplotlib.pyplot as plt

# 参数设置
N = 13  # 链的长度
T = 5000  # 总时间
h = 0.1  # 步长
timesteps = int(T / h)  # 总步数

# 非对称矩阵 A 的参数
a1, a2, a3 = 1, 2, 1

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
A[12,10] = a1
A[12,11] = -a2

print("Antisymmetric matrix A:")
print(A)

# 初始质量分布
m = np.zeros(N)
for i in range(N):
    if i % 2 == 1:  # S是奇数时，m_S 非零
        m[i] = 0.155
    if i % 2 == 0:  # S是偶数时，m_S非零
        m[i] = 0.01
print(m)

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

# 记录质量随时间变化
m_evolution = np.zeros((timesteps + 1, N))
m_evolution[0] = m

# 迭代计算
for t in range(timesteps):
    m = rk4_step(m, h, A)
    m_evolution[t + 1] = m

# 绘制结果图
time = np.linspace(0, T, timesteps + 1)

plt.figure(figsize=(12, 6))
plt.plot(time, m_evolution[:, 0], label='m1 (left)')
plt.plot(time, m_evolution[:, 6], label='m7 (mid)')
plt.plot(time, m_evolution[:, -1], label='m13 (right)')
plt.xlabel('Time')
plt.ylabel('Mass')
plt.title('Mass Evolution Over Time')
plt.legend()
plt.grid()
plt.show()