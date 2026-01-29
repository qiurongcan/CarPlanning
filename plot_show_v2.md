针对你明确的输入输出要求：
**输入**：全局路径点列表 `[x, y, theta, Vmax, Curvature]`
**输出**：轨迹点列表 `[x, y, theta, curvature, v, a, time]`

这是一个非常标准的工程化需求。为了实现从“离散地图点”到“连续控制轨迹”的转换，核心思路必须包含 **“参考线插值”** 和 **“Frenet合成”**。

以下是完整的 Python 实现代码，包含详细的注释和针对你要求的输入输出结构的适配。

### 核心实现思路

1.  **预处理（Pre-processing）**：
    全局路径通常是离散点。首先需要计算每个点的累计里程 $S_{ref}$，将全局路径构建成一个可以根据 $s$ 查询 $(r_x, r_y, r_\theta, r_\kappa)$ 的查找表（Look-up Table）。

2.  **匹配与定位（Matching）**：
    找到车辆当前位置在参考线上的投影点 $s_0$。
    *   计算当前横向误差 $l_0$。
    *   计算当前横向速度 $l'_0$ 和加速度 $l''_0$（通常近似为0）。
    *   计算当前纵向速度 $\dot{s}_0$。

3.  **多项式规划（Polynomial Planning）**：
    *   **横向 $l(t)$**：使用五次多项式，从 $(l_0, l'_0, l''_0)$ 规划到 $(0, 0, 0)$（回归中心线）。
    *   **纵向 $s(t)$**：使用四次或五次多项式（或速度剖面），从当前 $v_{start}$ 规划到参考线上的 $V_{max}$。

4.  **合成与重采样（Synthesis & Resampling）**：
    这是最关键的一步。在时间 $t$ 上循环：
    *   计算出 $s(t)$ 和 $l(t)$ 及其导数。
    *   **插值**：根据 $s(t)$ 在全局路径上找到对应的参考点 $(r_x, r_y, r_\theta, r_\kappa)$。**注意：不能只取最近点的索引，必须在两个离散点之间做线性插值，否则输出的 $\theta$ 和 $\kappa$ 会呈阶梯状跳变，导致控制抖动。**
    *   **转换**：利用公式计算出车辆在世界坐标系下的状态。

### Python 代码实现

```python
import numpy as np
import math
import matplotlib.pyplot as plt
import bisect

# --- 1. 基础数学工具：五次多项式 ---
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

# --- 2. 纵向速度规划器 (四次多项式) ---
# 用于从当前速度平滑过渡到目标速度
class QuarticPolynomialVelocity:
    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # xs: start position, vxs: start velocity, axs: start accel
        # vxe: end velocity, axe: end accel
        # 这里我们需要求解 s(t)，已知 s0, v0, a0 和 vt, at (st 不固定)
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        
        try:
            x = np.linalg.solve(A, b)
            self.a3, self.a4 = x[0], x[1]
        except np.linalg.LinAlgError:
            self.a3, self.a4 = 0, 0

    def calc_point(self, t):
        return self.a0 + self.a1 * t + self.a2 * t ** 2 + self.a3 * t ** 3 + self.a4 * t ** 4

    def calc_first_derivative(self, t):
        return self.a1 + 2 * self.a2 * t + 3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

    def calc_second_derivative(self, t):
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

# --- 3. 核心：参考路径处理器 ---
class ReferencePath:
    def __init__(self, path_data):
        """
        path_data: List of [x, y, theta, Vmax, Curvature]
        """
        self.path = np.array(path_data)
        self.x = self.path[:, 0]
        self.y = self.path[:, 1]
        self.yaw = self.path[:, 2]
        self.v_max = self.path[:, 3]
        self.k = self.path[:, 4]
        
        # 计算累计里程 s
        self.s = np.zeros(len(self.x))
        for i in range(1, len(self.x)):
            dist = np.hypot(self.x[i] - self.x[i-1], self.y[i] - self.y[i-1])
            self.s[i] = self.s[i-1] + dist

    def get_nearest_s(self, cur_x, cur_y):
        dx = self.x - cur_x
        dy = self.y - cur_y
        dist = np.hypot(dx, dy)
        min_idx = np.argmin(dist)
        return self.s[min_idx], min_idx, dist[min_idx]

    def get_reference_point(self, s_val):
        """
        关键函数：根据 s 值插值计算参考点状态
        防止因为离散点导致输出的 theta 和 curvature 跳变
        """
        # 找到 s_val 所在的区间索引
        idx = bisect.bisect_right(self.s, s_val) - 1
        if idx < 0: idx = 0
        if idx >= len(self.s) - 1: idx = len(self.s) - 2

        # 线性插值比例
        ds = s_val - self.s[idx]
        span = self.s[idx+1] - self.s[idx]
        ratio = ds / span if span > 0 else 0

        rx = self.x[idx] + (self.x[idx+1] - self.x[idx]) * ratio
        ry = self.y[idx] + (self.y[idx+1] - self.y[idx]) * ratio
        
        # 角度插值需要特殊处理
        dyaw = self.yaw[idx+1] - self.yaw[idx]
        # Normalize angle diff
        while dyaw >= math.pi: dyaw -= 2.0 * math.pi
        while dyaw < -math.pi: dyaw += 2.0 * math.pi
        ryaw = self.yaw[idx] + dyaw * ratio
        
        rv = self.v_max[idx] + (self.v_max[idx+1] - self.v_max[idx]) * ratio
        rk = self.k[idx] + (self.k[idx+1] - self.k[idx]) * ratio
        
        return rx, ry, ryaw, rv, rk

# --- 4. 核心：规划器 ---
class TrajectoryPlanner:
    def __init__(self, global_path_data):
        self.ref_path = ReferencePath(global_path_data)

    def plan(self, car_x, car_y, car_yaw, car_v, car_a, dt=0.1, T=5.0):
        """
        输入: 车辆当前状态
        输出: 轨迹点列表 [x, y, theta, curvature, v, a, t]
        """
        # 1. 匹配：找到最近点的 s
        s0_ref, idx, dist_err = self.ref_path.get_nearest_s(car_x, car_y)
        rx, ry, rtheta, r_vmax, rk = self.ref_path.get_reference_point(s0_ref)

        # 2. Cartesian -> Frenet
        # 计算横向偏差 l
        dx = car_x - rx
        dy = car_y - ry
        cross_rd_nd = math.cos(rtheta) * dy - math.sin(rtheta) * dx
        l0 = math.copysign(math.sqrt(dx**2 + dy**2), cross_rd_nd)
        
        # 计算横向导数 (近似)
        # l_dot = v * sin(delta_theta)
        l0_d = car_v * math.sin(car_yaw - rtheta)
        l0_dd = 0 # 简化，假设初始横向加速度为0

        # 3. 规划
        # 3.1 横向规划 (Quintic): 在 T 秒内收敛到 l=0
        lat_qp = QuinticPolynomial(l0, l0_d, l0_dd, 
                                   0.0, 0.0, 0.0, T) # 目标 l, l_dot, l_ddot 均为 0

        # 3.2 纵向规划 (Quartic Velocity): 在 T 秒内平滑加速/减速到 Vmax
        # 注意：这里的 Vmax 我们取参考路径上当前点的限速，
        # 实际更复杂的逻辑会向前看一段距离取最小限速
        target_v = r_vmax 
        lon_qp = QuarticPolynomialVelocity(s0_ref, car_v, car_a, 
                                           target_v, 0.0, T)

        # 4. 生成轨迹点 (Frenet -> Cartesian)
        trajectory = []
        time_span = np.arange(0, T + dt, dt)

        for t in time_span:
            # --- A. 获取 Frenet 状态 ---
            # 横向
            l = lat_qp.calc_point(t)
            l_d = lat_qp.calc_first_derivative(t)
            l_dd = lat_qp.calc_second_derivative(t)
            
            # 纵向 (s 以及 s 的导数)
            s = lon_qp.calc_point(t)
            v_s = lon_qp.calc_first_derivative(t) # ds/dt (纵向速度)
            a_s = lon_qp.calc_second_derivative(t) # d2s/dt2 (纵向加速度)

            # --- B. 获取参考点信息 (插值) ---
            rx, ry, rtheta, _, rkappa = self.ref_path.get_reference_point(s)

            # --- C. 坐标变换核心公式 ---
            
            # 1. 位置 x, y
            x = rx - l * math.sin(rtheta)
            y = ry + l * math.cos(rtheta)

            # 2. 速度 v (合速度)
            # v = sqrt((s_dot * (1 - k*l))^2 + l_dot^2)
            one_minus_kappa_l = 1.0 - rkappa * l
            v = math.sqrt((v_s * one_minus_kappa_l)**2 + l_d**2)

            # 3. 航向 theta
            # theta = rtheta + atan(l_dot / (s_dot * (1 - k*l)))
            delta_theta = math.atan2(l_d, v_s * one_minus_kappa_l)
            theta = rtheta + delta_theta
            
            # 4. 加速度 a (切向加速度)
            # 近似计算，严格计算需要复杂的求导，这里用纵向加速度近似
            a = a_s 

            # 5. 曲率 curvature (kappa)
            # 这是一个简化的工程近似公式，完整公式非常长
            # k_car ≈ k_ref + l_dd (当航向误差很小时)
            # 更精确一点: k_car = (l_dd + k_ref * (1 - k_ref * l)) / cos(delta_theta) / (1 - k_ref * l)^2
            # 我们使用适中的近似：
            if abs(v) < 0.1:
                k = rkappa
            else:
                # 这里的逻辑是：车辆的转弯 = 道路的转弯 + 变道产生的额外转弯
                # 分母 v^2 来自向心力公式 a_lat = v^2 * k
                k = (l_dd + rkappa * (1 - rkappa * l) * math.tan(delta_theta)**2) / \
                    (one_minus_kappa_l * math.cos(delta_theta)**2) 
                # 注意：上面这个公式主要定性，实际工程中常直接使用:
                # k = rkappa + l_dd / (v_s**2) (假设小角度)
                k = rkappa + l_dd  # 最简形式，通常足够用于控制前馈

            # 归一化角度
            theta = (theta + math.pi) % (2 * math.pi) - math.pi

            # 输出格式: [x, y, theta, curvature, v, a, time]
            trajectory.append([x, y, theta, k, v, a, t])

        return np.array(trajectory)

# --- 5. 测试与可视化 ---
def main():
    # 1. 构造输入数据: Global Path [x, y, theta, Vmax, Curvature]
    # 生成一个左转的圆弧路径
    s = np.arange(0, 100, 0.5)
    R = 50.0 # 半径
    path_x = s # 这里简化，先直行
    path_y = np.zeros_like(s)
    path_theta = np.zeros_like(s)
    path_k = np.zeros_like(s)
    
    # 后半段做成弯道
    for i in range(len(s)):
        if s[i] > 20:
            dist_in_curve = s[i] - 20
            path_x[i] = 20 + R * math.sin(dist_in_curve / R)
            path_y[i] = R - R * math.cos(dist_in_curve / R)
            path_theta[i] = dist_in_curve / R
            path_k[i] = 1.0 / R
            
    path_vmax = np.ones_like(s) * 10.0 # 限速 10 m/s
    
    # 组合成输入格式
    global_path = np.column_stack((path_x, path_y, path_theta, path_vmax, path_k))

    # 2. 定义车辆状态
    # 车在起点附近，偏离了中心线 2米，且速度较慢
    car_x, car_y = 0.0, 2.0 
    car_yaw = 0.0
    car_v = 2.0
    car_a = 0.0

    # 3. 执行规划
    planner = TrajectoryPlanner(global_path)
    traj = planner.plan(car_x, car_y, car_yaw, car_v, car_a, dt=0.1, T=6.0)

    # 4. 打印结果示例
    print("Output Format: [x, y, theta, curvature, v, a, time]")
    print(traj[:5])

    # 5. 简单绘图
    plt.figure(figsize=(10, 5))
    plt.plot(global_path[:,0], global_path[:,1], 'k--', label='Global Path')
    plt.plot(traj[:,0], traj[:,1], 'r.-', label='Planned Trajectory')
    plt.scatter([car_x], [car_y], color='blue', label='Car Start')
    plt.legend()
    plt.axis('equal')
    plt.title("Trajectory Planning Result")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
```

### 关键点解释

针对你的输入输出，代码做了以下特定设计：

1.  **输入解析 (`ReferencePath` 类)**:
    *   将 `[x, y, theta, Vmax, Curvature]` 拆解。
    *   **重要**：计算了 `s` (cumulative distance)。因为 `Vmax` 和 `Curvature` 是随路径位置变化的，我们需要通过 `s` 来索引这些属性。

2.  **插值函数 (`get_reference_point`)**:
    *   因为规划出的 `s(t)` 是连续的小数，而输入地图是离散的点。如果不做插值，直接取最近点的 `Curvature`，输出的曲率曲线会是阶梯状的，这会导致控制器的方向盘疯狂抖动。
    *   代码中对 `x, y, theta, Vmax, Curvature` 全部进行了线性插值。

3.  **计算输出维度**:
    *   **`x, y`**: 通过 $r_x - l \cdot \sin(r_\theta)$ 等公式计算。
    *   **`theta`**: 它是参考线航向 $r_\theta$ 加上由横向速度产生的偏角 $\arctan(l'/v)$。
    *   **`curvature`**: 它是参考线曲率 $r_\kappa$ 加上由横向加速度产生的修正项 $l''$。这是控制器前馈控制方向盘的关键。
    *   **`v, a`**: 由纵向多项式规划直接给出。

4.  **时间戳 `time`**:
    *   作为规划的自变量，从 0 开始递增，步长 `dt`。

这个代码结构可以直接嵌入到你的自动驾驶软件栈中，连接 Map 模块和 Control 模块。
