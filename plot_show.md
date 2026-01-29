这是一个使用 `matplotlib` 和 `numpy` 的完整 Python 示例。

为了让你更直观地理解，我构建了一个稍微复杂一点的场景：
1.  **全局路径**：不再是直线，而是一个**正弦波曲线**（模拟弯道）。
2.  **车辆状态**：车辆初始位置偏离了参考线，且航向也不对（模拟变道后或定位误差）。
3.  **规划目标**：车辆需要平滑地回归到参考曲线上，同时速度从当前速度加速到目标速度。

### 可视化代码

请在你的本地 Python 环境中运行此代码（需要安装 `matplotlib` 和 `numpy`）。

```python
import numpy as np
import matplotlib.pyplot as plt
import math

# --- 1. 基础数学工具类 (五次多项式) ---
class QuinticPolynomial:
    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        # 计算五次多项式系数: x(t) = a0 + a1*t + ... + a5*t^5
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[time ** 3, time ** 4, time ** 5],
                      [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                      [6 * time, 12 * time ** 2, 20 * time ** 3]])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
                      vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        
        try:
            x = np.linalg.solve(A, b)
            self.a3, self.a4, self.a5 = x[0], x[1], x[2]
        except np.linalg.LinAlgError:
            self.a3, self.a4, self.a5 = 0, 0, 0

    def calc_point(self, t):
        return self.a0 + self.a1 * t + self.a2 * t ** 2 + \
               self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

    def calc_first_derivative(self, t):
        return self.a1 + 2 * self.a2 * t + 3 * self.a3 * t ** 2 + \
               4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

    def calc_second_derivative(self, t):
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

# --- 2. 辅助函数：生成参考路径 ---
def generate_reference_path():
    # 生成一个正弦波形状的全局路径
    s_max = 100
    ds = 0.1
    s = np.arange(0, s_max, ds)
    
    # 路径坐标 (x, y)
    rx = s
    ry = 3.0 * np.sin(s / 10.0) 
    
    # 计算航向角 (yaw)
    ryaw = np.zeros_like(s)
    rk = np.zeros_like(s)
    
    for i in range(len(s) - 1):
        dx = rx[i+1] - rx[i]
        dy = ry[i+1] - ry[i]
        ryaw[i] = math.atan2(dy, dx)
    ryaw[-1] = ryaw[-2] # 补齐最后一个点
    
    return np.array(list(zip(rx, ry, ryaw, s)))

# --- 3. 核心规划逻辑 (Simulation) ---
def plan_and_visualize():
    # A. 准备数据
    global_path = generate_reference_path() # [x, y, yaw, s]
    
    # B. 定义车辆初始状态 (Cartesian)
    # 车在 x=5.0 处，y=-2.0 (偏离参考线)，朝向 0.5 rad，速度 2 m/s
    cur_x = 5.0
    cur_y = -2.0 
    cur_yaw = 0.5 
    cur_v = 2.0
    cur_a = 0.0
    
    # 规划参数
    TARGET_SPEED = 8.0  # 目标速度 m/s
    PLAN_T = 5.0        # 规划时长 5秒
    DT = 0.1            # 时间步长
    
    # C. 步骤1：定位与投影 (Finding Match Point)
    # 找到离车最近的参考点索引
    dx = global_path[:, 0] - cur_x
    dy = global_path[:, 1] - cur_y
    d = np.hypot(dx, dy)
    match_idx = np.argmin(d)
    
    ref_x, ref_y, ref_yaw, ref_s = global_path[match_idx]
    
    # D. 步骤2：Cartesian -> Frenet
    # 计算横向偏差 l (使用叉乘判断方向)
    cross_prod = math.cos(ref_yaw)*(cur_y - ref_y) - math.sin(ref_yaw)*(cur_x - ref_x)
    current_l = math.copysign(d[match_idx], cross_prod)
    
    # 近似计算横向速度 l_dot (实际应包含航向差)
    # l_dot = v * sin(delta_theta)
    current_l_d = cur_v * math.sin(cur_yaw - ref_yaw)
    current_l_dd = 0 # 简化假设
    
    # E. 步骤3：Frenet 轨迹生成 (Planning)
    # 3.1 横向规划：目标是 T 秒后 l=0, l_dot=0, l_ddot=0
    lat_qp = QuinticPolynomial(current_l, current_l_d, current_l_dd, 
                               0, 0, 0, PLAN_T)
    
    # 3.2 纵向规划：目标是 T 秒后 v=TARGET_SPEED (这里用简单的四次多项式或线性加速简化)
    # 这里我们直接线性插值速度作为演示，重点在于轨迹形状
    
    time_span = np.arange(0, PLAN_T, DT)
    
    # 存储结果
    traj_x, traj_y, traj_v, traj_l = [], [], [], []
    
    # 模拟沿参考线积分
    current_s_idx = match_idx
    
    for t in time_span:
        # 1. 获取当前时刻的 Frenet 状态
        l_t = lat_qp.calc_point(t)
        
        # 简单速度规划：匀加速
        acc = (TARGET_SPEED - cur_v) / PLAN_T
        v_t = cur_v + acc * t
        
        # 2. 计算当前时刻的 s (沿参考线前进的距离)
        # s(t) = s0 + v0*t + 0.5*a*t^2
        travel_dist = cur_v * t + 0.5 * acc * t**2
        
        # 3. 寻找 s 对应的参考点 (Frenet -> Cartesian 映射的关键)
        # 在 global_path 中找到对应的 index
        target_s = ref_s + travel_dist
        
        # 简单的向前搜索
        while current_s_idx < len(global_path)-1 and global_path[current_s_idx][3] < target_s:
            current_s_idx += 1
            
        r_x, r_y, r_theta, _ = global_path[current_s_idx]
        
        # 4. Frenet -> Cartesian 坐标变换
        # x = r_x - l * sin(r_theta)
        # y = r_y + l * cos(r_theta)
        x_t = r_x - l_t * math.sin(r_theta)
        y_t = r_y + l_t * math.cos(r_theta)
        
        traj_x.append(x_t)
        traj_y.append(y_t)
        traj_v.append(v_t)
        traj_l.append(l_t)

    # --- 4. 可视化 (Visualization) ---
    fig = plt.figure(figsize=(14, 8))
    
    # 子图1：XY 平面轨迹
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    ax1.plot(global_path[:, 0], global_path[:, 1], 'k--', label="Global Reference Path", alpha=0.5)
    ax1.plot(traj_x, traj_y, 'b-', linewidth=2, label="Planned Trajectory")
    
    # 画出起点和车辆方向
    ax1.plot(cur_x, cur_y, 'ro', label="Start Pos")
    ax1.arrow(cur_x, cur_y, math.cos(cur_yaw)*2, math.sin(cur_yaw)*2, 
              head_width=0.5, head_length=0.5, fc='r', ec='r')
    
    # 画几个轨迹中间的箭头表示方向
    for i in range(0, len(traj_x), 10):
        if i+1 < len(traj_x):
            dx_t = traj_x[i+1] - traj_x[i]
            dy_t = traj_y[i+1] - traj_y[i]
            yaw_t = math.atan2(dy_t, dx_t)
            ax1.arrow(traj_x[i], traj_y[i], math.cos(yaw_t), math.sin(yaw_t),
                      head_width=0.3, fc='b', ec='b', alpha=0.5)

    ax1.set_title("Trajectory Generation (XY Plane)")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.axis("equal")
    ax1.grid(True)
    ax1.legend()
    
    # 子图2：横向偏差 l(t)
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax2.plot(time_span, traj_l, 'g-', linewidth=2)
    ax2.set_title("Lateral Error Profile (Frenet l)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Lateral Distance l [m]")
    ax2.axhline(0, color='k', linestyle='--') # 目标线
    ax2.grid(True)
    
    # 子图3：速度规划 v(t)
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    ax3.plot(time_span, traj_v, 'm-', linewidth=2)
    ax3.set_title("Velocity Profile")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Speed [m/s]")
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plan_and_visualize()
```

### 图表解读

运行上述代码后，你会看到一个包含三个子图的窗口：

1.  **左侧大图 (Trajectory Generation - XY Plane)**:
    *   **黑色虚线**：全局参考路径（正弦波）。
    *   **红点/红箭头**：车辆的初始位置。你可以看到它并没有在黑线上，而且朝向也可能不正。
    *   **蓝色实线**：生成的轨迹。
    *   **观察点**：请注意蓝色线是如何从红点出发，平滑地切入并最终重合到黑色虚线上的。这展示了“路径跟踪”和“平滑回归”的效果。

2.  **右上小图 (Lateral Error Profile - Frenet l)**:
    *   这是 Frenet 坐标系下的横向规划结果。
    *   纵轴是 $l$（距离中心线的距离）。
    *   **观察点**：曲线是一条光滑的 S 型曲线（五次多项式特征），从初始误差（比如 -2米 或 -3米）在 5 秒内平滑地收敛到 0。这保证了变道或回正过程不会让乘客感到顿挫。

3.  **右下小图 (Velocity Profile)**:
    *   这是纵向规划的结果。
    *   **观察点**：速度从初始速度（2 m/s）线性增加到目标速度（8 m/s）。在实际 Apollo 代码中，这里可能是 S 型速度曲线以保证加速度连续，但线性加速在这里用于演示已足够。

### 关键点总结

*   **解耦**：我们在右上图只关心“偏离多少”，在右下图只关心“跑多快”，最后在左侧大图中将它们合成为“走到哪”。
*   **平滑**：因为使用了五次多项式（Quintic Polynomial），你可以看到轨迹没有折角，这意味着加速度和加加速度（Jerk）都是连续的，这对于控制模块（Control）来说非常友好。
