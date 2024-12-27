import scipy
import numpy as np
import matplotlib.pyplot as plt
from common_utils import *

from nuplan.planning.simulation.path.path import AbstractPath
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType, STATIC_OBJECT_TYPES
from nuplan.planning.simulation.planner.utils.breadth_first_search import BreadthFirstSearch
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusData, TrafficLightStatusType
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.metrics.utils.expert_comparisons import principal_value
# 对路径进行检查和精简，移除相邻点之间距离过近的点。
def check_path(path):
    # path：输入路径，通常是一个二维数组，形状为(N, 2)，表示路径上的点集，每个点包含(x, y)坐标。
    refine_path = [path[0]]
        
    for i in range(1, path.shape[0]):
        # 计算当前点与前一个点之间的欧几里得距离（np.linalg.norm）。
        if np.linalg.norm(path[i] - path[i-1]) < 0.1:
            continue
        else:
            refine_path.append(path[i])
        
    line = np.array(refine_path)

    return line


def calculate_path_heading(path):
    # 使用np.arctan2计算每两个相邻点之间的朝向角度（弧度制）。范围为[-π, π]。
    heading = np.arctan2(path[1:, 1] - path[:-1, 1], path[1:, 0] - path[:-1, 0])
    heading = np.append(heading, heading[-1])

    return heading


def trajectory_smoothing(trajectory):
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    h = trajectory[:, 2]

    window_length = 40
    # 对x、y和h分别进行平滑，减少轨迹中的抖动或不连续性
    # savgol_filter：一种平滑滤波器，基于Savitzky-Golay算法，可以在保留数据趋势的同时减少噪声。
    # window_length=40：滑动窗口的长度，表示每次平滑时考虑的点数。
    # 返回的list的shape和输入的列表一样
    x = scipy.signal.savgol_filter(x, window_length=window_length, polyorder=3)
    y = scipy.signal.savgol_filter(y, window_length=window_length, polyorder=3)
    h = scipy.signal.savgol_filter(h, window_length=window_length, polyorder=3)
#    将平滑后的x、y和h重新组合成一个二维数组，形状为(N, 3)。
    return np.column_stack([x, y, h])

# 该函数的作用是将任意角度值(弧度制)归一化到[-π, π]范围
def wrap_to_pi(theta):
    return (theta+np.pi) % (2*np.pi) - np.pi
