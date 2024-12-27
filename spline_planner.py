import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 三次样条曲线系数 三次曲线 四个系数 根据四个已知条件构建四个方程即可求解 依次返回从低阶到高阶对应的系数
def cubic_spline_coefficients(x0, dx0, xf, dxf, tf):
    return (x0, dx0, -2 * dx0 / tf - dxf / tf - 3 * x0 / tf ** 2 + 3 * xf / tf ** 2,
            dx0 / tf ** 2 + dxf / tf ** 2 + 2 * x0 / tf ** 3 - 2 * xf / tf ** 3)

# N是轨迹离散点的个数
# TODO 速度优化 然后 和之前的三次样条得到的path结合
def compute_spline_xyvaqrt(v0, dv0, vf, tf, path, N, offset):
    t = torch.arange(N+1).to(v0.device) * tf / N
    # 将时间张量的形状从（N+1）扩展到（N+1，1），方便后续广播计算
    # 生成【0 1 2 3】的整数张量
    # 两个张量计算涉及到了广播机制！！！
    tp = t[..., None] ** torch.arange(4).to(v0.device)
    dtp = t[..., None] ** torch.tensor([0, 0, 1, 2]).to(v0.device) * torch.arange(4).to(v0.device)
    
    coefficients = cubic_spline_coefficients(v0, dv0, vf, 0, tf)
    # 将返回的系数堆叠为一个张量 形状为（4，），unsqueeze(-1)是在最后一个维度增加一个维度，变为（4， 1），方便后续矩阵运算
    coefficients = torch.stack(coefficients).unsqueeze(-1)
    # 矩阵相乘
    # TODO shape：timestep， 1
    v = tp @ coefficients
    a = dtp @ coefficients
    # 计算累计的弧长
    s = torch.cumsum(v * tf / N, dim=0)
    s = torch.cat((torch.zeros(1, 1).to(v0.device), s[:-1]), dim=0)
    s += offset
    # 转化为long类型
    i = (s / 0.1).long()
    # TODO
    if i[-1] > path.shape[0] - 1:
        return

    x = path[i, 0]
    y = path[i, 1]
    yaw = path[i, 2]
    r = path[i, 3]
    # 增加t的维度，然后再最后一个维度上合并这些张量，squeeze(0)并从张量中移除第一个维度，如果该维度的大小为 1。
    # squeeze 方法用于去除张量中所有长度为1的维度。squeeze(0) 指定去除第一个维度（索引为0）的长度为1的维度。
    # 这通常用于在连接操作后，当第一个维度（批次维度）大小为1时，去除这个单一维度，使得张量的形状更加紧凑。
    return torch.cat((x, y, yaw, v, a, r, t.unsqueeze(-1)), -1).squeeze(0)


class SplinePlanner:
    def __init__(self, first_stage_horizion, horizon):
        self.spline_order = 3
        self.max_curve = 0.3
        self.max_lat_acc = 3.0
        self.acce_bound = [-5, 3]
        # TODO 速度的范围是0~15mps
        self.vbound = [0, 15.0]
        # 第一段轨迹的时间长度
        self.first_stage_horizion = first_stage_horizion
        # 整个轨迹（两段轨迹）的时间长度
        self.horizon = horizon
    # path是路径点
    def calc_trajectory(self, v0, a0, vf, tf, path, N_seg, offset=0):
        traj = compute_spline_xyvaqrt(v0, a0, vf, tf, path, N_seg, offset)

        return traj
    # 短期轨迹生成 dyn_filter动态约束过滤的标志位
    def gen_short_term_trajs(self, x0, tf, paths, dyn_filter):
        xf_set = []
        trajs = []
        
        # generate speed profile and trajectories
        for path in paths:
            path = torch.from_numpy(path).to(x0.device).type(torch.float)
            # 对末端速度进行了采样
            for v in self.v_grid:
                traj = self.calc_trajectory(x0[3], x0[4], v, tf, path, self.first_stage_horizion*10) # [x, y, yaw, v, a, r, t]
                if traj is None:
                    continue
                # 末端状态的xy信息，最后一行的索引0 1
                xf = traj[-1, :2]
                # 检查轨迹终点 xf 是否与已有终点过于接近（欧几里得距离小于 0.5）
                # 去重
                if xf_set and torch.cdist(xf.unsqueeze(0), torch.stack(xf_set)).min() < 0.5:
                    continue
                else:
                    xf_set.append(xf)
                    trajs.append(traj)
        # 堆叠在一起，张量的维度增加一个
        # TODO shape: path_size timesteps dim
        trajs = torch.stack(trajs)
        
        # remove trajectories that are not feasible
        if dyn_filter:
            feas_flag = self.feasible_flag(trajs)
            trajs = trajs[feas_flag]

        return trajs
    # 长期轨迹生成
    def gen_long_term_trajs(self, x0, tf, paths, dyn_filter):
        xf_set = []
        trajs = []
        
        # generate speed profile and trajectories
        for path in paths:
            path = torch.from_numpy(path).to(x0.device).type(torch.float)
            dist = torch.norm(path[:, :2] - x0[:2], dim=1)
            if dist.min() > 0.1:
                continue
            
            offset = torch.argmin(dist) * 0.1

            for v in self.v_grid:
                traj = self.calc_trajectory(x0[3], x0[4], v, tf, path, (self.horizon-self.first_stage_horizion)*10, offset) # [x, y, yaw, v, a, r, t]
                if traj is None:
                    continue

                xf = traj[-1, :2]
  
                if xf_set and torch.cdist(xf.unsqueeze(0), torch.stack(xf_set)).min() < 0.5:
                    continue
                else:
                    xf_set.append(xf)
                    trajs.append(traj)

        if len(trajs) == 0:
            return
        else:
            trajs = torch.stack(trajs)
        
        # remove trajectories that are not feasible
        if dyn_filter:
            feas_flag = self.feasible_flag(trajs)
            trajs = trajs[feas_flag]

        return trajs

    def feasible_flag(self, trajs):
        feas_flag = ((trajs[:, 1:, 3] >= self.vbound[0]) & 
                     (trajs[:, 1:, 3] <= self.vbound[1]) &
                     (trajs[:, 1:, 4] >= self.acce_bound[0]) & 
                     (trajs[:, 1:, 4] <= self.acce_bound[1]) &
                     (trajs[:, 1:, 5].abs() * trajs[:, 1:, 3] ** 2 <= self.max_lat_acc) &
                     (trajs[:, 1:, 5].abs() <= self.max_curve)
                    ).all(1)

        if feas_flag.sum() == 0:
            print("No feasible trajectory")
            feas_flag = torch.ones(trajs.shape[0], dtype=torch.bool).to(trajs.device)
        
        return feas_flag
    # 综合轨迹生成
    # TODO 速度分配
    def gen_trajectories(self, x0, tf, paths, speed_limit, is_root):
        # generate trajectories
        v0 = x0[3]

        if is_root:
            v_min = max(v0 - 4.0 * tf, 0.0)
            v_max = min(v0 + 2.4 * tf, speed_limit)
            # 速度采样 和论文描述基本符合
            self.v_grid = torch.linspace(v_min, v_max, 10).to(x0.device)
            trajs = self.gen_short_term_trajs(x0, tf, paths, dyn_filter=False)
        else:
            v_min = max(v0 - tf, 0.0)
            v_max = min(v0 + tf, speed_limit)
            # 速度采样 和论文描述基本符合
            self.v_grid = torch.linspace(v_min, v_max, 5).to(x0.device)
            trajs = self.gen_long_term_trajs(x0, tf, paths, dyn_filter=False)

        # adjust timestep
        if not is_root:
            # TODO ？？？？
            trajs[:, :, -1] += self.horizon - self.first_stage_horizion

        # remove the first time step
        # TODO
        trajs = trajs[:, 1:]

        return trajs
