# CarPlanning
这是一个非常经典的自动驾驶**局部路径规划（Local Planning）**或**轨迹生成（Trajectory Generation）**问题。

虽然你提到“不需要做控制”，但“路径跟踪”这个词通常暗示了控制。不过，根据你要求的输出格式 `(x, y, yaw, curvature, v, a, t)`，这实际上是一个**轨迹规划层**的任务。你的目标是生成一条在时间 $t$ 上连续、平滑且符合车辆运动学的轨迹，供下游的控制器（如 MPC 或 PID）去执行。

最主流且符合你前面提到的 Apollo 代码风格的方法是使用 **Frenet 坐标系下的五次多项式（Quintic Polynomial）轨迹规划**。

以下是实现思路和 Python 代码示例。

### 核心思路：Frenet 坐标系解耦

1.  **定位与投影（Matching）：**
    找到小车当前位置 `(cur_x, cur_y)` 在全局参考路径上的投影点（匹配点）。这能让你算出小车当前相对于参考线的纵向位置 $s$ 和横向误差 $l$（以及横向误差的导数 $l', l''$）。

2.  **坐标转换（Cartesian to Frenet）：**
    将小车的状态 `(x, y, yaw, v, a)` 转换为 Frenet 状态 `(s, s_dot, s_ddot, l, l_prime, l_double_prime)`。

3.  **轨迹生成（Planning）：**
    在 Frenet 坐标系下，将问题拆分为两个一维问题：
    *   **横向规划（Lateral）：** 设计一个函数 $l(t)$，让小车从当前的偏差 $l_{current}$ 平滑地回归到 $0$（即回到参考线中心）。通常使用**五次多项式**来保证加速度和平滑度（Jerk）连续。
    *   **纵向规划（Longitudinal）：** 设计一个函数 $s(t)$，让小车从当前速度 $v_{current}$ 平滑加速/减速到目标速度 $V_{max}$。

4.  **坐标回转（Frenet to Cartesian）：**
    将规划出的 $s(t)$ 和 $l(t)$ 序列点，结合参考线的几何信息，转换回全局坐标系 `(x, y)`。

5.  **计算运动学参数：**
    利用微分关系计算出每一个时间步 $t$ 的 `yaw`, `curvature`, `v`, `a`。

---

### Python 实现代码

为了简化，这里假设全局路径是由离散点组成的，我们使用五次多项式来生成未来几秒的轨迹。

```python
import numpy as np
import math

class QuinticPolynomial:
    """
    五次多项式求解器：x(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    用于连接起始状态 (x0, v0, a0) 和 终止状态 (xt, vt, at) 在时间 T 内
    """
    def __init__(self, x0, v0, a0, xt, vt, at, T):
        self.a0 = x0
        self.a1 = v0
        self.a2 = a0 / 2.0

        A = np.array([[T**3, T**4, T**5],
                      [3 * T**2, 4 * T**3, 5 * T**4],
                      [6 * T, 12 * T**2, 20 * T**3]])
        b = np.array([xt - self.a0 - self.a1 * T - self.a2 * T**2,
                      vt - self.a1 - 2 * self.a2 * T,
                      at - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        return self.a0 + self.a1 * t + self.a2 * t**2 + self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5

    def calc_first_derivative(self, t):
        return self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2 + 4 * self.a4 * t**3 + 5 * self.a5 * t**4

    def calc_second_derivative(self, t):
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3

    def calc_third_derivative(self, t):
        return 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t**2


class LocalPlanner:
    def __init__(self, global_path):
        """
        global_path: list of [x, y, theta, Vmax, curvature]
        """
        self.global_path = np.array(global_path)
        
    def find_nearest_index(self, cur_x, cur_y):
        """
        找到全局路径上离当前位置最近的点的索引
        """
        dx = self.global_path[:, 0] - cur_x
        dy = self.global_path[:, 1] - cur_y
        distances = np.hypot(dx, dy)
        return np.argmin(distances)

    def cartesian_to_frenet(self, cur_x, cur_y, cur_yaw, cur_v, idx):
        """
        粗略的坐标转换（简化版），实际项目中需要更复杂的投影计算
        """
        rx, ry, rtheta, _, _ = self.global_path[idx]
        
        dx = cur_x - rx
        dy = cur_y - ry
        
        # 计算 Frenet 坐标系下的 l (横向偏差)
        # cross_product > 0 在左侧，< 0 在右侧
        cross_rd_nd = math.cos(rtheta) * dy - math.sin(rtheta) * dx
        l = math.copysign(math.sqrt(dx**2 + dy**2), cross_rd_nd)
        
        # 近似计算 s (纵向里程) - 这里简单用索引代替，实际应用需累加距离
        # 假设点之间距离较小且均匀，这里仅作逻辑演示
        s = idx  # 这里的单位不是米，实际代码中应该是 path_s[idx]
        
        # 简化的速度分解
        s_d = cur_v * math.cos(cur_yaw - rtheta)
        l_d = cur_v * math.sin(cur_yaw - rtheta)
        
        return s, s_d, l, l_d

    def plan_trajectory(self, cur_x, cur_y, cur_yaw, cur_v, cur_a=0):
        """
        主规划函数
        """
        # 1. 寻找匹配点
        idx = self.find_nearest_index(cur_x, cur_y)
        
        # 获取参考点信息
        ref_x, ref_y, ref_yaw, ref_vmax, ref_k = self.global_path[idx]
        
        # 2. 转换到 Frenet (s, l)
        # 假设当前位于 s0, l0，目标是 T 秒后回到 l=0 (中心线)
        # 这里的 s 单位需要注意，为了演示方便，我们假设 s 等同于时间推演
        
        # 当前状态
        c_l = math.sqrt((cur_x - ref_x)**2 + (cur_y - ref_y)**2) # 简化：距离误差
        # 判断左右
        cross = math.cos(ref_yaw)*(cur_y-ref_y) - math.sin(ref_yaw)*(cur_x-ref_x)
        c_l = c_l if cross > 0 else -c_l
        
        c_l_d = cur_v * math.sin(cur_yaw - ref_yaw) # 横向速度近似
        c_l_dd = 0 # 假设横向加速度为0
        
        # 目标状态 (Target)
        # 我们希望 T 秒后，横向误差 l=0，横向速度 l_d=0，横向加速度 l_dd=0
        T = 4.0  # 规划未来 4 秒
        target_l = 0
        target_l_d = 0
        target_l_dd = 0
        
        # 纵向目标：希望 T 秒后达到参考速度
        target_v = ref_vmax
        
        # 3. 生成多项式
        # 横向规划：从当前 l 变到 0
        lat_qp = QuinticPolynomial(c_l, c_l_d, c_l_dd, target_l, target_l_d, target_l_dd, T)
        
        # 纵向规划：从当前 v 变到 ref_vmax (使用四次或五次，这里用五次简化)
        # 假设 s0=0 (相对运动), 目标 s_dot = target_v
        # s 的规划通常用 Quartic (四次) 因为不需要固定 s_end，只需要固定 v_end
        # 这里为了演示方便，我们简单地假设加速度恒定或者直接积分速度
        
        trajectory_points = []
        dt = 0.1 # 时间步长
        
        # 预测未来 T 秒的轨迹
        for t in np.arange(0, T, dt):
            # --- A. 计算 Frenet 状态 ---
            
            # 横向状态 l(t)
            l_t = lat_qp.calc_point(t)
            l_d_t = lat_qp.calc_first_derivative(t)
            l_dd_t = lat_qp.calc_second_derivative(t)
            
            # 纵向状态 s(t) (简单线性加速模型作为演示)
            # 实际应使用纵向多项式
            acc = (target_v - cur_v) / T
            v_t = cur_v + acc * t
            s_dist = cur_v * t + 0.5 * acc * t**2 # 相对当前点的距离
            
            # --- B. 转换回 Global Cartesian ---
            # 我们需要找到 s_dist 对应的参考线上的点
            # 这是一个沿路径积分的过程
            
            # 简易做法：向前搜索参考路径
            # 实际中应使用 s 坐标插值
            dist_accum = 0
            lookahead_idx = idx
            while lookahead_idx < len(self.global_path) - 1:
                d = np.hypot(self.global_path[lookahead_idx+1][0] - self.global_path[lookahead_idx][0],
                             self.global_path[lookahead_idx+1][1] - self.global_path[lookahead_idx][1])
                if dist_accum + d > s_dist:
                    break
                dist_accum += d
                lookahead_idx += 1
            
            # 获取参考点属性
            r_x, r_y, r_theta, r_v, r_k = self.global_path[lookahead_idx]
            
            # 计算这一时刻的坐标 (x, y)
            # 公式: x = r_x - l * sin(r_theta)
            #       y = r_y + l * cos(r_theta)
            x_t = r_x - l_t * math.sin(r_theta)
            y_t = r_y + l_t * math.cos(r_theta)
            
            # --- C. 计算 Yaw, Curvature, V, A ---
            
            # 1. Yaw (航向角)
            # 近似公式：yaw ≈ r_theta + atan(l_d / v) 
            # 更精确的公式见 Apollo 代码：NormalizeAngle(rtheta + std::atan2(dl, 1 - l * rkappa));
            yaw_t = r_theta + math.atan2(l_d_t, v_t) # 简化版
            
            # 2. Curvature (曲率)
            # 这是一个复杂的几何关系，如果不使用 Apollo 的复杂公式，
            # 可以通过计算三个连续点的圆弧来求，或者利用 l_dd
            # 简化公式：k ≈ r_k + l_dd
            k_t = r_k # 简单假设跟随道路曲率，实际需加上 l_dd 的影响
            
            # 3. Velocity (速度)
            # v_t 已经在纵向规划中算出
            
            # 4. Acceleration (加速度)
            a_t = acc # 已经在纵向规划中算出
            
            # 存储结果 (x, y, yaw, curvature, v, a, t)
            trajectory_points.append([x_t, y_t, yaw_t, k_t, v_t, a_t, t])
            
        return np.array(trajectory_points)

# --- 测试数据 ---
# 生成一条直线的全局路径: x=0~100, y=0, theta=0
global_path_data = []
for i in range(100):
    global_path_data.append([i, 0, 0, 10.0, 0]) # x, y, theta, Vmax, k

planner = LocalPlanner(global_path_data)

# 假设车在 (0, 2) 位置（偏离中心线2米），朝向0，速度1m/s
# 我们希望它回到 y=0 的中心线
cur_x, cur_y, cur_yaw, cur_v = 0, 2, 0, 1

trajectory = planner.plan_trajectory(cur_x, cur_y, cur_yaw, cur_v)

# 打印前几个点看看结果
print("Format: [x, y, yaw, curvature, v, a, t]")
for p in trajectory[:5]:
    print(np.round(p, 3))
    
# 预期结果：
# y 应该从 2 逐渐变小趋向于 0
# t 应该从 0 增加
```

### 详细步骤解析 (针对你的 7 维输出)

代码中的逻辑是为了生成你需要的 7 个维度：

1.  **`t` (Time):**
    这是自变量。规划通常是基于时间的（例如未来 0s 到 4s，步长 0.1s）。

2.  **`x`, `y` (Position):**
    这是最关键的输出。
    *   首先通过纵向规划得到每一时刻小车在参考线上的“虚拟位置” $s(t)$，找到对应的参考点 $(r_x, r_y, r_{\theta})$。
    *   然后通过横向规划得到每一时刻的横向偏差 $l(t)$。
    *   最后合成：
        $$x(t) = r_x(s) - l(t) \cdot \sin(r_{\theta})$$
        $$y(t) = r_y(s) + l(t) \cdot \cos(r_{\theta})$$

3.  **`yaw` (Heading):**
    小车的朝向不仅仅取决于参考线的朝向 $r_{\theta}$，还取决于横向移动的变化。
    $$yaw(t) \approx r_{\theta} + \arctan\left(\frac{dl}{ds}\right)$$
    如果横向偏差变化很快（急变道），车头必须偏得更多。

4.  **`curvature` ($\kappa$):**
    这是给控制模块的重要前馈量，决定了方向盘转角。
    它是参考线曲率 $\kappa_{ref}$ 和横向运动产生的曲率修正的组合。如 Apollo 代码中所示，计算公式非常复杂，但在简化模型中，可以理解为：
    $$\kappa_{car} \approx \kappa_{ref} + \text{横向二阶导修正项}$$

5.  **`v` (Velocity):**
    由纵向规划直接给出。通常是一个平滑的速度剖面（S型曲线或线性），确保不超过 $V_{max}$。

6.  **`a` (Acceleration):**
    速度 $v(t)$ 对时间的导数。控制模块需要它来实现精确的纵向控制（油门/刹车）。

### 总结

要实现你想要的功能，**不要**试图直接在 $(x,y)$ 坐标系下画曲线，那样很难控制速度和加速度的平滑性。

**正确的流程是：**
1.  **XY -> SL**: 把车扔进 Frenet 坐标系。
2.  **Plan in SL**: 在 $S$ 轴规划速度，在 $L$ 轴规划偏移（用多项式拟合）。
3.  **SL -> XY**: 把规划好的 $S(t)$ 和 $L(t)$ 映射回世界坐标，同时计算出导数作为 $v, a, \kappa$。

这就是 Apollo、Autoware 等主流自动驾驶框架中 Planner 模块的核心工作原理。
