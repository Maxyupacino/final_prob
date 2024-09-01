import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 参数设置
N = 13  # 链的长度
T = 5000  # 总时间
h = 0.1  # 步长
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

def simps_manual(y, x):
    """手动实现Simpson's法则计算积分"""
    if len(x) != len(y):
        raise ValueError("长度不匹配：x和y的长度必须相同")
    if len(x) % 2 == 0:
        raise ValueError("区间数必须为奇数")

    h = x[1] - x[0]  # 步长
    n = len(x) - 1  # 区间数

    integral = y[0] + y[-1]  # 边界项

    # 奇数项的总和
    odd_sum = np.sum(y[1:-1:2])

    # 偶数项的总和
    even_sum = np.sum(y[2:-2:2])

    integral += 4 * odd_sum + 2 * even_sum  # 中间项的加权
    integral *= h / 3  # 计算积分值

    return integral

mass_polarization = np.zeros(N)
for i in range(N):
    mass_polarization[i] = simps_manual(m_evolution[:, i], np.linspace(0, T, timesteps + 1)) / T

#定义拟合的函数形式（此处用指数函数拟合）
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

#计算拟合曲线
x_fit = np.arange(1, N + 1)
y_fit = mass_polarization

# 增大 maxfev 并提供初始猜测 p0
initial_guess = [1, 0.1, 0]
fit_params, _ = curve_fit(func, x_fit, y_fit, p0=initial_guess, maxfev=5000)
y_fit = func(x_fit, *fit_params)

# 绘制质量极化图
plt.figure(figsize=(8, 6))
plt.plot(range(1, N + 1), mass_polarization, marker='o', linestyle='-', label='Mass Polarization')
plt.plot(x_fit, y_fit, color='red', label='Fit')
plt.xlabel('Site')
plt.ylabel('Mass Polarization')
plt.title('Mass Polarization vs Site')
plt.legend()
plt.grid(True)
plt.show()