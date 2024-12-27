import torch
import logging
import random
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
from debug_utils import *
# 初始化日志记录器，用于将日志信息输出到文件和控制台。
# def initLogging(log_file: str, level: str = "INFO"):
#     logging.basicConfig(filename=log_file, filemode='w',
#                         level=getattr(logging, level, None),
#                         # format='[%(levelname)s %(asctime)s] %(message)s',
#                         format='%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s - %(message)s'
#                         # datefmt='%m-%d %H:%M:%S'
#                         )
#     # 添加一个 StreamHandler，将日志信息输出到控制台(终端）。
#     # logging.getLogger().addHandler(logging.StreamHandler())

# 设置随机种子，确保代码运行的可重复性。
def set_seed(CUR_SEED):
    # 设置python numpy pytorch随机数生成器的种子。
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    # 强制使用确定性算法，确保每次运行的结果一致。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# NOTE 定义一个自定义数据集类，用于加载和处理自动驾驶轨迹数据。
class DrivingData(Dataset):
    def __init__(self, data_list, n_neighbors, n_candidates):
        # 存储数据文件路径的列表
        self.data_list = data_list
        # 邻居车辆的最大数量。
        self._n_neighbors = n_neighbors
        # 自车候选轨迹的最大数量。
        self._n_candidates = n_candidates
        # 时间序列的长度（80 个时间步）。80 * 0.1 = 8s
        self._time_length = 80

    def __len__(self):
        return len(self.data_list)
    # 对自车轨迹进行预处理，确保其形状符合模型的输入要求。
    # 轨迹的数量和每个轨迹的时间步数需要满足设定的要求
    def process_ego_trajectory(self, ego_trajectory):
        # 初始化一个零矩阵，用于存储处理后的轨迹，形状为 [n_candidates, time_length, 6]。
        # TODO 6是 x y yaw vel acc curvature
        # 自车的轨迹树的分支数！！！
        trajectory = np.zeros((self._n_candidates, self._time_length, 6), dtype=np.float32)
        # 获取第一个维度的大小，如果轨迹的候选数量超过 n_candidates，则截断多余的候选。
        if ego_trajectory.shape[0] > self._n_candidates:
            ego_trajectory = ego_trajectory[:self._n_candidates]
        
        if ego_trajectory.shape[1] < self._time_length:
            # 如果轨迹的时间步少于 time_length，则用零填充。
            trajectory[:ego_trajectory.shape[0], :ego_trajectory.shape[1]] = ego_trajectory
        else:
            trajectory[:ego_trajectory.shape[0]] = ego_trajectory

        return trajectory
    # 当你使用 DataLoader 来迭代数据集时，它会调用数据集的 __getitem__ 方法来获取每个批次的数据。
    # 加载单个样本的数据，并返回处理后的数据。
    def __getitem__(self, idx):
        # 加载的数据是一个字典类型
        data = np.load(self.data_list[idx])
        ego = data['ego_agent_past']
        neighbors = data['neighbor_agents_past']
        route_lanes = data['route_lanes'] 
        map_lanes = data['map_lanes']
        map_crosswalks = data['map_crosswalks']
        ego_future_gt = data['ego_agent_future'] # Ego 未来的真实轨迹！！！
        # NOTE 切片操作 只考虑附近最近的self._n_neighbors个交通参与者
        neighbors_future_gt = data['neighbor_agents_future'][:self._n_neighbors]
        # 处理后的自车轨迹
        # 这里使用省略号...来表示“所有前面的维度”，这意味着你将选择所有维度上的所有元素，直到最后一个维度。然后，:6表示你将选择最后一个维度上的前6个元素。
        # 无效的位置 已经使用0进行了填充
        first_stage = self.process_ego_trajectory(data['first_stage_ego_trajectory'][..., :6])
        second_stage = self.process_ego_trajectory(data['second_stage_ego_trajectory'][..., :6])

        return ego, neighbors, map_lanes, map_crosswalks, route_lanes, ego_future_gt, neighbors_future_gt, first_stage, second_stage

# NOTE 计算模型预测结果的损失，包括轨迹误差、正则化损失和分类损失。
# neighbors:bt branch neighbor times(0.1s切片) 3
# ego：batchsize branch time dim 
# ego_regularization:bt times 3
# scores： bt branch
# weights:bt 8
# ego_gt：bt 80 3
# neighbors_gt:bt neighbors 80 3
# neighbors_valid:bt neighbors 80 3
def calc_loss(neighbors, ego, ego_regularization, scores, weights, ego_gt, neighbors_gt, neighbors_valid):
    # .sum(-1) 是对张量 ego 进行求和操作。这里的 -1 表示最后一个维度，也就是说，这个操作会沿着张量的最后一个维度（通常是特征维度）对所有元素进行求和。
    # torch.ne 是PyTorch中的一个函数，用于计算两个张量逐元素的“不等于”（not equal）。它返回一个与输入张量相同形状的布尔型张量，其中的每个元素表示两个输入张量相应位置的元素是否不相等。
    # 将 ego.sum(-1) 的结果与标量0进行比较，检查它们是否不等于0。
    # NOTE: bt branch timesteps(80)
    mask = torch.ne(ego.sum(-1), 0) # 生成掩码，标记自车轨迹中非零的时间步。
    # neighbors_valid 是一个布尔型张量或掩码，用于标识哪些邻居对象的轨迹是有效的
    # 如果 neighbors_valid 中的元素为 True，则对应的 neighbors[:, 0] 元素保持不变；如果为 False，则对应的 neighbors[:, 0] 元素会被乘以 0，从而在后续计算中被过滤掉。
    # NOTE 取第一个分支的原因：论文：III-D 模型训练中提到了
    print_tensor_shape_log(neighbors, "neighbors", DEBUG)
    print_tensor_shape_log(neighbors[:, 0], "neighbors[:, 0]", DEBUG)
    # bt neighbors 80 3
    neighbors = neighbors[:, 0] * neighbors_valid # 过滤无效的邻居轨迹。
    # 设置为 'none' 意味着不对损失值进行任何聚合操作，而是返回每个样本的损失值。
    # 这允许开发者对每个样本的损失进行更细粒度的控制和分析。cmp_loss这个张量的维度和neighbors是一样的！！！
    # NOTE:condition motion predict!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # bt neighbors 80 3
    # NOTE 论文：III-D 模型训练中提到了
    cmp_loss = F.smooth_l1_loss(neighbors, neighbors_gt, reduction='none')
    # NOTE None的作用
    print_tensor_shape_log(cmp_loss, "cmp_loss", DEBUG)
    print_tensor_shape_log(mask, "mask", DEBUG)
    print_tensor_shape_log(mask[:, 0, None, :, None], "mask[:, 0, None, :, None]", DEBUG)

    # 将 cmp_loss 与 mask 相乘，mask 用于过滤掉无效的邻居轨迹。这里使用 None 来增加维度，使得 mask 的形状与 cmp_loss 匹配。 true:1 false:0
    # bt neighbors 80 3
    cmp_loss = cmp_loss * mask[:, 0, None, :, None]
    # 对 cmp_loss 进行求和，并除以 mask 中有效元素的和，得到平均损失。
    cmp_loss = cmp_loss.sum() / mask[:, 0].sum()
    # TODO 为啥取第一个分支
    regularization_loss = F.smooth_l1_loss(ego_regularization, ego_gt, reduction='none')
    regularization_loss = regularization_loss * mask[:, 0, :, None]
    regularization_loss = regularization_loss.sum() / mask[:, 0].sum()
    # 交叉熵损失（Cross Entropy Loss）:
    #  创建一个全零的标签张量，用于交叉熵损失的计算。
    # TODO 类别都是0 也就是第一个分支
    label = torch.zeros(scores.shape[0], dtype=torch.long).to(scores.device)
    # 输入的尺寸分别为(bt, branch) (bt)   
    print_tensor_shape_log(scores, "scores", DEBUG)
    print_tensor_shape_log(label, "label", DEBUG)
    irl_loss = F.cross_entropy(scores, label)
    # 计算权重的平方的均值，这是一种正则化项，用于防止模型过拟合。
    weights_regularization = torch.square(weights).mean()
    # 标量
    loss = cmp_loss + irl_loss + 0.1 * regularization_loss + 0.01 * weights_regularization

    return loss

# 评价指标计算
# 它用于计算规划（planning）和预测（prediction）任务的评估指标。这些指标通常用于衡量自动驾驶系统中轨迹预测和规划模型的性能。
def calc_metrics(plan_trajectory, prediction_trajectories, scores, ego_future, neighbors_future, neighbors_future_valid):
    '''
    plan_trajectory: bt branch max_timesteps(80) dim(6)
    prediction_trajectories: bt branch neighbor times(0.1s切片) 3
    scores： bt branch
    ego_future: bt timesteps(80) xy heading
    neighbors_future: bt neighbor timesteps(80) xy heading
    neighbors_future_valid： bt neighbor timesteps(80) xy heading
    '''
    # 选择评分最高的候选轨迹。
    # dim=-1 通常用于在最后一个维度上寻找最大值的索引
    # 找到每个邻居每个预测轨迹中评分最高的索引。
    best_idx = torch.argmax(scores, dim=-1)
    # .shape[0]:第一个维度的大小
    # plan_trajectory[x, x]:张量的索引
    # NOTE plan_trajectory: bt max_timesteps(80) dim(6)
    plan_trajectory = plan_trajectory[torch.arange(plan_trajectory.shape[0]), best_idx]
    # NOTE bt neighbor times(0.1s切片) 3
    prediction_trajectories = prediction_trajectories[torch.arange(prediction_trajectories.shape[0]), best_idx]
    # 对应位置的元素想乘 过滤无效的邻居轨迹
    prediction_trajectories = prediction_trajectories * neighbors_future_valid
    # NOTE bt max_timesteps(80)
    plan_distance = torch.norm(plan_trajectory[:, :, :2] - ego_future[:, :, :2], dim=-1)
    # 计算坐标差的欧几里得范数（即距离）
    # NOTE bt neighbor max_timesteps
    prediction_distance = torch.norm(prediction_trajectories[:, :, :, :2] - neighbors_future[:, :, :, :2], dim=-1)

    # planning
    plannerADE = torch.mean(plan_distance)
    plannerFDE = torch.mean(plan_distance[:, -1])

    # prediction
    # 包括 ADE（Average Displacement Error，平均位移误差）和 FDE（Final Displacement Error，终点位移误差）
    # 分别计算邻居车辆的平均误差（ADE）和终点误差（FDE）。
    # NOTE bt neighbor
    predictorADE = torch.mean(prediction_distance, dim=-1)
    # TODO 使用掩码过滤无效的邻居轨迹。
    # NOTE bt neighbor 如果掩码中某个位置为 True，则 predictorADE 中对应位置的值会被选中；否则会被忽略。
    predictorADE = torch.masked_select(predictorADE, neighbors_future_valid[:, :, 0, 0])
    predictorADE = torch.mean(predictorADE)
    # NOTE bt neighbor
    predictorFDE = prediction_distance[:, :, -1]
    # NOTE 选对应位置为true的值
    predictorFDE = torch.masked_select(predictorFDE, neighbors_future_valid[:, :, 0, 0])
    # 结果是一个标量（0维张量），表示 predictorFDE 中所有元素的平均值
    predictorFDE = torch.mean(predictorFDE)
    # 用于从标量张量中提取其值，并将其转换为 Python 的原生数据类型（如 float 或 int）。
    return plannerADE.item(), plannerFDE.item(), predictorADE.item(), predictorFDE.item()
