import torch
import scipy
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from shapely import Point, LineString
from shapely.geometry.base import CAP_STYLE
from planner_utils import *
from common_utils import *
from bezier_path import calc_4points_bezier_path
from path_planner import calc_spline_course

from nuplan.database.nuplan_db.query_session import execute_one, execute_many
from nuplan.database.nuplan_db.nuplan_scenario_queries import *
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo
# convert_absolute_to_relative_poses：将全局坐标系中的轨迹转换为相对于参考点的局部坐标系。
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses
# path_to_linestring：将路径数据转换为线段表示，便于后续几何计算
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring
# Agents：用于描述场景中其他交通参与者（如车辆、行人等）的状态。
# TrajectorySampling：轨迹采样相关的工具，用于从连续轨迹中提取关键点。
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
# 将全局坐标的SE2状态（2D平移+旋转）转换为局部坐标。
from nuplan.common.geometry.torch_geometry import global_state_se2_tensor_to_local
# AgentInternalIndex 和 EgoInternalIndex：定义了交通参与者和自车状态的特征索引，用于访问特定特征（如位置、速度等）。
# 其他函数（如filter_agents_tensor、pad_agent_states等）：与交通参与者的状态处理相关，包括过滤、填充、打包等操作。
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    AgentInternalIndex,
    EgoInternalIndex,
    sampled_past_ego_states_to_tensor,
    sampled_past_timestamps_to_tensor,
    compute_yaw_rate_from_state_tensors,
    filter_agents_tensor,
    pack_agents_tensor,
    pad_agent_states
)

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
# 将一组坐标转换为局部参考系。
from nuplan.common.geometry.torch_geometry import vector_set_coordinates_to_local_frame
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import *
# interpolate_points：对点进行插值，用于补齐或均匀分布点集。
from nuplan.planning.training.preprocessing.utils.vector_preprocessing import interpolate_points

# 这段代码是一个自动驾驶系统中场景数据处理的模块，主要用于从地图和传感器数据中提取特征，并将其转换为固定大小的张量，供神经网络或其他算法使用。

# 从给定的交通参与者对象中提取特定类型的特征（如车辆、行人等），并将其转换为张量。
########## Network input features ##########
# 参数：
# tracked_objects：场景中所有已检测的交通参与者。
# TODO track_token_ids：一个字典，用于将每个交通参与者的唯一标识符映射到整数ID。 一一对应！！！！！！！！！！！！！！！！！！！！！！！！！
# # object_types：需要提取的交通参与者类型（如车辆、行人、自行车等）。
def _extract_agent_tensor(tracked_objects, track_token_ids, object_types):
    """
    Extracts the relevant data from the agents present in a past detection into a tensor.
    Only objects of specified type will be transformed. Others will be ignored.
    The output is a tensor as described in AgentInternalIndex
    :param tracked_objects: The tracked objects to turn into a tensor.
    TODO :track_token_ids: A dictionary used to assign track tokens to integer IDs.
    :object_type: TrackedObjectType to filter agents by.
    :return: The generated tensor and the updated track_token_ids dict.
    """
    # 只提取object_type中包含的物体类型
    agents = tracked_objects.get_tracked_objects_of_types(object_types)
    agent_types = []
    output = torch.zeros((len(agents), AgentInternalIndex.dim()), dtype=torch.float32)
    max_agent_id = len(track_token_ids)
    # NOTE list 类型
    for idx, agent in enumerate(agents):
        # token
        if agent.track_token not in track_token_ids: # 判断这个字典的键中有无track_token
            track_token_ids[agent.track_token] = max_agent_id
            max_agent_id += 1
        # TODO
        track_token_int = track_token_ids[agent.track_token]
        # TODO output中 第一个元素代表跟踪的objects的id（用int依次定义的）
        # TODO 只考虑其他交通参与者的位姿 尺寸 速度
        output[idx, AgentInternalIndex.track_token()] = float(track_token_int)
        output[idx, AgentInternalIndex.vx()] = agent.velocity.x
        output[idx, AgentInternalIndex.vy()] = agent.velocity.y
        output[idx, AgentInternalIndex.heading()] = agent.center.heading
        output[idx, AgentInternalIndex.width()] = agent.box.width
        output[idx, AgentInternalIndex.length()] = agent.box.length
        output[idx, AgentInternalIndex.x()] = agent.center.x
        output[idx, AgentInternalIndex.y()] = agent.center.y
        agent_types.append(agent.tracked_object_type)
    # 返回值：包含交通参与者特征的张量、更新后的ID映射字典，以及交通参与者类型列表。
    return output, track_token_ids, agent_types

# 将过去的检测结果（交通参与者状态）转换为张量列表。
def sampled_tracked_objects_to_tensor_list(past_tracked_objects):
    """
    Tensorizes the agents features from the provided past detections.
    For N past detections, output is a list of length N, with each tensor as described in `_extract_agent_tensor()`.
    :param past_tracked_objects: The tracked objects to tensorize.
    :return: The tensorized objects.
    """
    # 物体类型 只针对这三种类型的障碍物（他车 也被称为other agent）
    object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE]
    output = []
    output_types = []
    # NOTE 跟踪的物体的id
    track_token_ids = {}
    # NOTE for循环的是 时间
    for i in range(len(past_tracked_objects)):
        tensorized, track_token_ids, agent_types = _extract_agent_tensor(past_tracked_objects[i], track_token_ids, object_types)
        output.append(tensorized)
        output_types.append(agent_types)

    return output, output_types

# 将全局坐标系中的速度向量转换为局部坐标系。
# 偏航角度 逆时针为正
def global_velocity_to_local(velocity, anchor_heading):
    velocity_x = velocity[:, 0] * torch.cos(anchor_heading) + velocity[:, 1] * torch.sin(anchor_heading)
    velocity_y = velocity[:, 1] * torch.cos(anchor_heading) - velocity[:, 0] * torch.sin(anchor_heading)

    return torch.stack([velocity_x, velocity_y], dim=-1)

# 将交通参与者的状态从全局坐标系转换为相对于自车的局部坐标系。
def convert_absolute_quantities_to_relative(agent_state, ego_state, agent_type='ego'):
    """
    Converts the agent' poses and relative velocities from absolute to ego-relative coordinates.

    :param agent_state: The agent states to convert, in the AgentInternalIndex schema.
    :param ego_state: The ego state to convert, in the EgoInternalIndex schema.
    :return: The converted states, in AgentInternalIndex schema.
    """
    ego_pose = torch.tensor(
        [
            float(ego_state[EgoInternalIndex.x()].item()),
            float(ego_state[EgoInternalIndex.y()].item()),
            float(ego_state[EgoInternalIndex.heading()].item()),
        ],
        dtype=torch.float64,
    )

    if agent_type == 'ego':
        agent_global_poses = agent_state[:, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]]
        agent_global_velocities = agent_state[:, [AgentInternalIndex.vx(), AgentInternalIndex.vy()]]

        transformed_poses = global_state_se2_tensor_to_local(agent_global_poses, ego_pose, precision=torch.float64)
        # TODO 速度也进行了转换，加速度没有转换！！！
        transformed_velocities = global_velocity_to_local(agent_global_velocities, ego_pose[-1])
        agent_state[:, EgoInternalIndex.x()] = transformed_poses[:, 0].float()
        agent_state[:, EgoInternalIndex.y()] = transformed_poses[:, 1].float()
        agent_state[:, EgoInternalIndex.heading()] = transformed_poses[:, 2].float()
        agent_state[:, AgentInternalIndex.vx()] = transformed_velocities[:, 0].float()
        agent_state[:, AgentInternalIndex.vy()] = transformed_velocities[:, 1].float()
    else:
        agent_global_poses = agent_state[:, [AgentInternalIndex.x(), AgentInternalIndex.y(), AgentInternalIndex.heading()]]
        agent_global_velocities = agent_state[:, [AgentInternalIndex.vx(), AgentInternalIndex.vy()]]
        transformed_poses = global_state_se2_tensor_to_local(agent_global_poses, ego_pose, precision=torch.float64)
        transformed_velocities = global_velocity_to_local(agent_global_velocities, ego_pose[-1])
        agent_state[:, AgentInternalIndex.x()] = transformed_poses[:, 0].float()
        agent_state[:, AgentInternalIndex.y()] = transformed_poses[:, 1].float()
        agent_state[:, AgentInternalIndex.heading()] = transformed_poses[:, 2].float()
        agent_state[:, AgentInternalIndex.vx()] = transformed_velocities[:, 0].float()
        agent_state[:, AgentInternalIndex.vy()] = transformed_velocities[:, 1].float()

    return agent_state


def agent_past_process(past_ego_states, past_time_stamps, past_tracked_objects, tracked_objects_types, num_agents):
    """
    This function process the data from the raw agent data.
    :param past_ego_states: The input tensor data of the ego past.
    :param past_time_stamps: The input tensor data of the past timestamps.
    :param past_time_stamps: The input tensor data of other agents in the past.
    :return: ego_agent_array, other_agents_array.
    """
    agents_states_dim = Agents.agents_states_dim()
    ego_history = past_ego_states
    time_stamps = past_time_stamps
    agents = past_tracked_objects
    # anchor_ego_state：获取自车的参考状态（最后一帧的状态）。也就是最后一个时间刻的状态
    # squeeze() 方法用于去除张量中所有长度为1的维度。这在处理具有单一时间步的历史状态时很有用，因为它可以去除单一时间步维度，使得张量的形状更加紧凑。
    # clone() 方法用于创建张量的一个副本。这在你需要保留原始张量不变，同时对副本进行操作时非常有用。
    anchor_ego_state = ego_history[-1, :].squeeze().clone()
    # 将自车的历史状态从全局坐标系转换为相对于参考状态的局部坐标系。
    ego_tensor = convert_absolute_quantities_to_relative(ego_history, anchor_ego_state)
    # 过滤并提取交通参与者的历史状态。
    # shape is (timesteps, agents, states)
    agent_history = filter_agents_tensor(agents, reverse=True)
    # TODO agent_types：获取最后一帧中交通参与者的类型。
    agent_types = tracked_objects_types[-1]

    """
    Model input feature representing the present and past states of the ego and agents, including:
    ego: <np.ndarray: num_frames, 7>
        The num_frames includes both present and past frames.
        TODO 3 + 2 + 2
        The last dimension is the ego pose (x, y, heading) velocities (vx, vy) acceleration (ax, ay) at time t.
    agents: <np.ndarray: num_frames, num_agents, 8>
        Agent features indexed by agent feature type.
        The num_frames includes both present and past frames.
        The num_agents is padded to fit the largest number of agents across all frames.
        The last dimension is the agent pose (x, y, heading) velocities (vx, vy, yaw rate) and size (length, width) at time t.
    """
    # NOTE 当前时刻没有需要关注的交通参与者 agents
    if agent_history[-1].shape[0] == 0:
        # Return zero tensor when there are no agents in the scene
        agents_tensor = torch.zeros((len(agent_history), 0, agents_states_dim)).float()
    else:
        local_coords_agent_states = []
        # TODO pad_agent_states：填充交通参与者状态，使得每一帧的交通参与者数量与当前帧一样，并且 顺序也是一样的！！！！！！！！。
        # TODO 此时，padded_agent_states中每一时刻的agents的顺序就一样了！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        padded_agent_states = pad_agent_states(agent_history, reverse=True)
        # for 循环的也是时间
        for agent_state in padded_agent_states:
            local_coords_agent_states.append(convert_absolute_quantities_to_relative(agent_state, anchor_ego_state, 'agent'))
    
        # Calculate yaw rate
        yaw_rate_horizon = compute_yaw_rate_from_state_tensors(padded_agent_states, time_stamps)
    
        agents_tensor = pack_agents_tensor(local_coords_agent_states, yaw_rate_horizon)

    '''
    Post-process the agents tensor to select a fixed number of agents closest to the ego vehicle.
    agents: <np.ndarray: num_agents, num_frames, 11>]].
        Agent type is one-hot encoded: [1, 0, 0] vehicle, [0, 1, 0] pedestrain, [0, 0, 1] bicycle 
            and added to the feature of the agent
        The num_agents is padded or trimmed to fit the predefined number of agents across.
        The num_frames includes both present and past frames.
    '''
    # agents_tensor.shape[0] is num_frames
    agents = np.zeros(shape=(num_agents, agents_tensor.shape[0], agents_tensor.shape[-1]+3), dtype=np.float32)

    # sort agents according to distance to ego
    # 已跟踪的他车（tracked agents）的状态，已经转换到自车坐标系下了，所以到自车的距离，也就是他车的坐标的norm
    # TODO dim=-1 指定了计算范数的维度，-1 表示当前时刻 Now，即每个代理的 x 和 y 坐标。
    distance_to_ego = torch.norm(agents_tensor[-1, :, :2], dim=-1)
    # TODO 返回输入张量的元素从小到大排序后的索引列表
    indices = list(torch.argsort(distance_to_ego).numpy())[:num_agents]

    # fill agent features into the array
    # agent的类型 作为特征feature 加入到agent的feature中！！！！
    # NOTE 按照距离自车从近到远的顺序
    for i, j in enumerate(indices):
        agents[i, :, :agents_tensor.shape[-1]] = agents_tensor[:, j, :agents_tensor.shape[-1]].numpy()
        if agent_types[j] == TrackedObjectType.VEHICLE:
            agents[i, :, agents_tensor.shape[-1]:] = [1, 0, 0]
        elif agent_types[j] == TrackedObjectType.PEDESTRIAN:
            agents[i, :, agents_tensor.shape[-1]:] = [0, 1, 0]
        else:
            agents[i, :, agents_tensor.shape[-1]:] = [0, 0, 1]

    return ego_tensor.numpy().astype(np.float32), agents, indices
# 获取相关索引的状态 比如AgentInternalIndex.x(), AgentInternalIndex.y(), AgentInternalIndex.heading()这几个函数 返回的均是int类型的！！！！！！！！！！！！！！！！！！！

# 处理未来交通参与者状态，将其转换为局部坐标系，并生成固定大小的特征张量。
def agent_future_process(anchor_ego_state, future_tracked_objects, num_agents, agent_index):
    anchor_ego_state = torch.tensor([anchor_ego_state.rear_axle.x, anchor_ego_state.rear_axle.y, anchor_ego_state.rear_axle.heading, 
                                     anchor_ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
                                     anchor_ego_state.dynamic_car_state.rear_axle_velocity_2d.y,
                                     anchor_ego_state.dynamic_car_state.rear_axle_acceleration_2d.x,
                                     anchor_ego_state.dynamic_car_state.rear_axle_acceleration_2d.y])
    # 提取每一帧中 出现在第一帧中的物体
    agent_future = filter_agents_tensor(future_tracked_objects)
    local_coords_agent_states = []
    # NOTE for 循环的是时间
    for agent_state in agent_future:
        # TODO shape:nums_frame agent_nums agent_states
        local_coords_agent_states.append(convert_absolute_quantities_to_relative(agent_state, anchor_ego_state, 'agent'))
    # NOTE local_coords_agent_states中每一帧的agents的顺序是不一样的 还没有对齐
    padded_agent_states = pad_agent_states_with_zeros(local_coords_agent_states)

    # fill agent features into the array
    # num_agents nums_frame states
    # TODO 顺序对齐了！！！ 按照agent_index的顺序进行对齐，past_agent的顺序也是这个顺序，所以，执行完下面的代码之后 agents的past 和 future的状态中，
    # TODO agent的顺序都是对齐的。同一行对应是某一个代理的过去 现在 未来的状态
    agent_futures = np.zeros(shape=(num_agents, padded_agent_states.shape[0]-1, 3), dtype=np.float32)
    for i, j in enumerate(agent_index):
        agent_futures[i] = padded_agent_states[1:, j, [AgentInternalIndex.x(), AgentInternalIndex.y(), AgentInternalIndex.heading()]].numpy()

    return agent_futures

# 主要功能是对交通参与者的轨迹数据进行填充，使得每一帧的交通参与者数量与第一帧一致。
def pad_agent_states_with_zeros(agent_trajectories):
    # shape is agents_numbers, dim
    key_frame = agent_trajectories[0]
    # track_id_idx 是交通参与者的唯一标识符（track_token）在状态向量中的索引位置。   注意是状态向量
    # 每个交通参与者都有一个唯一的 track_token，用于区分不同的交通参与者。
    # 通过这个索引，可以找到每个交通参与者在不同帧中的对应数据。
    track_id_idx = AgentInternalIndex.track_token() # 返回0 也就是第零列是agent的唯一标识符 tracked token！！！！！！！！！！！！！！！！！！！！！！！！！！
    # print(f'hailan-debug--track_id_idx: {track_id_idx}')
    #     frame = torch.tensor([  
    #     [1, 10, 20],  # 第一个交通参与者，track_token 为 1  
    #     [2, 30, 40],  # 第二个交通参与者，track_token 为 2  
    #     [3, 50, 60],  # 第三个交通参与者，track_token 为 3  
    # ])  
    # 第一列是agen的标识符 也就是track_token
    # TODO shape is timesteps, key_frame[0], key_frame[1]
    pad_agent_trajectories = torch.zeros((len(agent_trajectories), key_frame.shape[0], key_frame.shape[1]), dtype=torch.float32)
    for idx in range(len(agent_trajectories)):
        frame = agent_trajectories[idx]
        # mapped_rows 是当前帧中所有交通参与者的 track_token（唯一标识符）。
        mapped_rows = frame[:, track_id_idx]
        # print(f'hailan-debug---mapped_rows: {mapped_rows}')
        # 遍历参考帧中的每个交通参与者（row_idx 表示参考帧中交通参与者的索引）。

        # 其实这个地方这样写 是建立在 agent的标识符是从0依次增加的 但是不如官方写的更加稳健！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # TODO 遍历自车当前时刻的所有检测到的objects
        # 0 1 2 3 4 5 6 ……
        for row_idx in range(key_frame.shape[0]):
            if row_idx in mapped_rows:
                # 使用布尔索引从 frame 中提取满足条件的交通参与者状态。
                # 如果当前帧中存在 track_token 等于 row_idx 的交通参与者，则返回其状态。
                # 如果不存在，则返回一个空张量。

                # 结果为：tensor([False, True, False])  
                # 使得 每一帧中的agent的顺序都和第一frame是一样的！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
                pad_agent_trajectories[idx, row_idx] = frame[frame[:, track_id_idx]==row_idx]
    # 返回一个固定大小的张量 pad_agent_trajectories，每一帧的交通参与者数量与第一帧（key_frame）的数量一致。如果某些交通参与者在某些帧中不存在，则用零填充。
    return pad_agent_trajectories

# 将地图特征（如车道、导航车道 人行横道）转换为固定大小的张量，支持插值和填充。
def convert_feature_layer_to_fixed_size(ego_pose, feature_coords, feature_tl_data, max_elements, max_points,
                                         traffic_light_encoding_dim, interpolation):
    """
    Converts variable sized map features to fixed size tensors. Map elements are padded/trimmed to max_elements size.
        Points per feature are interpolated to maintain max_points size.
    :param ego_pose: the current pose of the ego vehicle.
    :param feature_coords: Vector set of coordinates for collection of elements in map layer.
        TODO [num_elements, num_points_in_element (variable size), 2]
    :param feature_tl_data: Optional traffic light status corresponding to map elements at given index in coords.
       TODO [num_elements, traffic_light_encoding_dim (4)]
    :param max_elements: Number of elements to pad/trim to.
    :param max_points: Number of points to interpolate or pad/trim to.
    :param traffic_light_encoding_dim: Dimensionality of traffic light data.
    :param interpolation: Optional interpolation mode for maintaining fixed number of points per element.
        None indicates trimming and zero-padding to take place in lieu of interpolation. Interpolation options: 'linear' and 'area'.
    :return
        coords_tensor: The converted coords tensor.
        tl_data_tensor: The converted traffic light data tensor (if available).
        avails_tensor: Availabilities tensor identifying real vs zero-padded data in coords_tensor and tl_data_tensor.
    :raise ValueError: If coordinates and traffic light data size do not match.!!!!!!!!!!!!!!!!!
    """
    if feature_tl_data is not None and len(feature_coords) != len(feature_tl_data):
        raise ValueError(f"Size between feature coords and traffic light data inconsistent: {len(feature_coords)}, {len(feature_tl_data)}")

    # trim or zero-pad elements to maintain fixed size
    coords_tensor = torch.zeros((max_elements, max_points, 2), dtype=torch.float32)
    # avails_tensor：存储有效性标志。
    avails_tensor = torch.zeros((max_elements, max_points), dtype=torch.bool)
    tl_data_tensor = (
        torch.zeros((max_elements, max_points, traffic_light_encoding_dim), dtype=torch.float32)
        if feature_tl_data is not None else None
    )

    # get elements according to the mean distance to the ego pose
    # 计算每个特征到自车的最小距离，并按距离排序，选择最近的max_elements个特征。
    mapping = {}
    for i, e in enumerate(feature_coords):
        # 分别是index value
        dist = torch.norm(e - ego_pose[None, :2], dim=-1).min()
        mapping[i] = dist
        print(f"11111111type is {dist}   {type(dist)}")
    # NOTE mapping 变成了一个列表，列表中的每个元素都是一个元组，元组中包含了原始字典的键和值。(idx tensor)
    mapping = sorted(mapping.items(), key=lambda item: item[1])
    # 只取前max_element个最近的地图元素
    sorted_elements = mapping[:max_elements]

    # pad or trim waypoints in a map element
    for idx, element_idx in enumerate(sorted_elements):
        element_coords = feature_coords[element_idx[0]]
    
        # interpolate to maintain fixed size if the number of points is not enough
        element_coords = interpolate_points(element_coords, max_points, interpolation=interpolation)
        coords_tensor[idx] = element_coords
        # TODO 广播机制
        avails_tensor[idx] = True  # specify real vs zero-padded data 指定实数据与零填充数据

        if tl_data_tensor is not None and feature_tl_data is not None:
            # TODO 用到了广播机制
            # 将一个形状为[traffic_light_encoding_dim]的一维张量赋值给另一个形状为[max_points, traffic_light_encoding_dim]的二维张量时，确实会使用到广播机制。
            # 广播机制允许PyTorch在某些维度上自动扩展较小的张量，以便它们可以与较大张量的维度相匹配，从而进行数学运算或赋值操作。
            tl_data_tensor[idx] = feature_tl_data[element_idx[0]]

    return coords_tensor, tl_data_tensor, avails_tensor

# 提取自车周围的地图特征（如车道、边界线）和交通灯信息
def get_neighbor_vector_set_map(
    map_api: AbstractMap,
    map_features: List[str],
    point: Point2D,
    radius: float,
    route_roadblock_ids: List[str],
    traffic_light_status_data: List[TrafficLightStatusData],
) -> Tuple[Dict[str, MapObjectPolylines], Dict[str, LaneSegmentTrafficLightData]]:
    """
    Extract neighbor vector set map information around ego vehicle.
    :param map_api: map to perform extraction on.
    map_api：

    一个抽象的地图 API（AbstractMap），用于从地图中提取各种特征数据。
    提供地图相关的查询功能，例如获取车道、多边形对象、交通灯等。
    :param map_features: Name of map features to extract.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about vector map query range.
    :param route_roadblock_ids: List of ids of roadblocks/roadblock connectors (lane groups) within goal route.
    :param traffic_light_status_data: A list of all available data at the current time step.
    :return:
    映射 Map xxx to yyy
        coords: Dictionary mapping feature name to polyline vector sets.
        traffic_light_data: Dictionary mapping feature name to traffic light info corresponding to map elements
            in coords.
    :raise ValueError: if provided feature_name is not a valid VectorFeatureLayer.
    """
    coords: Dict[str, MapObjectPolylines] = {}
    traffic_light_data: Dict[str, LaneSegmentTrafficLightData] = {}
    feature_layers: List[VectorFeatureLayer] = []
    # 将输入的地图特征名称转换为对应的特征层，如果名称无效则抛出异常。
    for feature_name in map_features:
        try:
            # VectorFeatureLayer[feature_name] 的含义是 通过键（key）访问枚举成员
            # 如果 VectorFeatureLayer 是一个枚举类，[] 的作用是通过成员名称（字符串）访问对应的枚举值。
            feature_layers.append(VectorFeatureLayer[feature_name])
        except KeyError:
            raise ValueError(f"Object representation for layer: {feature_name} is unavailable")

    # extract lanes
    if VectorFeatureLayer.LANE in feature_layers:
        # 提取车道中心线、左边界、右边界以及对应的id
        lanes_mid, lanes_left, lanes_right, lane_ids = get_lane_polylines(map_api, point, radius)
        # lane baseline paths baseline就是道路中心线
        # TODO 车道中心线
        # TODO coords[VectorFeatureLayer.LANE.name]和traffic_light_data[VectorFeatureLayer.LANE.name]两个对象的数据成员（list类型）的长度十一月的
        coords[VectorFeatureLayer.LANE.name] = lanes_mid

        # lane traffic light data
        traffic_light_data[VectorFeatureLayer.LANE.name] = get_traffic_light_encoding(
            lane_ids, traffic_light_status_data
        )

        # lane boundaries
        if VectorFeatureLayer.LEFT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.LEFT_BOUNDARY.name] = MapObjectPolylines(lanes_left.polylines)
        if VectorFeatureLayer.RIGHT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.RIGHT_BOUNDARY.name] = MapObjectPolylines(lanes_right.polylines)

    # extract route
    # 提取自车的导航路线！！！
    if VectorFeatureLayer.ROUTE_LANES in feature_layers:
        route_polylines = get_route_lane_polylines_from_roadblock_ids(map_api, point, radius, route_roadblock_ids)
        # TODO 导航车道的中心线
        coords[VectorFeatureLayer.ROUTE_LANES.name] = route_polylines
    # extract generic map objects
    for feature_layer in feature_layers:
        # 语义地图特征层
        # TODO
        # 遍历 feature_layers 中的每个特征层。
        # 使用 VectorFeatureLayerMapping.semantic_map_layer(feature_layer) 确定特征层对应的语义地图层。
        # 将多边形数据存储到 coords 字典中。
        if feature_layer in VectorFeatureLayerMapping.available_polygon_layers():
            polygons = get_map_object_polygons(
            # Returns associated SemanticMapLayer for feature extraction, if exists.
                map_api, point, radius, VectorFeatureLayerMapping.semantic_map_layer(feature_layer)
            )
            # TODO 人行横道
            coords[feature_layer.name] = polygons
            print(f'hailan-data_utils--------feature_layer.name is : {feature_layer.name}')

    return coords, traffic_light_data

# map_process 函数用于处理地图数据，将原始矢量化的地图数据（如车道、交通灯信息等）转换为固定大小的张量，并将其从全局坐标系转换到自车坐标系，同时生成适合后续模型使用的格式化地图数据。
def map_process(anchor_state, coords, traffic_light_data, map_features, max_elements, max_points, interpolation_method):
    """
    This function process the data from the raw vector set map data.
    :param anchor_state: The current state of the ego vehicle.
    :param coords: The input data of the vectorized map coordinates.！！！！！！！！！！！
    :param traffic_light_data: The input data of the traffic light data.
    :return: dict of the map elements.
    """

    # convert data to tensor list
    anchor_state_tensor = torch.tensor([anchor_state.x, anchor_state.y, anchor_state.heading], dtype=torch.float32)
    list_tensor_data = {}
    # 字典类型
    # 遍历地图特征并将其转换为张量
    # TODO 车道特征有:车道 导航车道 人行横道等等
    # coords：输入的矢量化地图数据，包含所有地图特征（车道 导航车道 人行横道）的坐标信息。
    for feature_name, feature_coords in coords.items():
        list_feature_coords = []

        # Pack coords into tensor list
        # 将特征数据转换为矢量化格式，返回每个地图元素的坐标集合。
        for element_coords in feature_coords.to_vector():
            # NOTE element_coords也是一个列表
            list_feature_coords.append(torch.tensor(element_coords, dtype=torch.float32))
        list_tensor_data[f"coords.{feature_name}"] = list_feature_coords
        print(f'hailan---data_utils------coords.feature_name is : {feature_name}')

        # Pack traffic light data into tensor list if it exists
        if feature_name in traffic_light_data:
            list_feature_tl_data = []
            # traffic_light_data[feature_name]：当前地图特征的交通灯数据。
            for element_tl_data in traffic_light_data[feature_name].to_vector():
                # NOTE element_tl_data也是一个列表
                list_feature_tl_data.append(torch.tensor(element_tl_data, dtype=torch.float32))
            list_tensor_data[f"traffic_light_data.{feature_name}"] = list_feature_tl_data
            print(f'hailan---data_utils------coords.traffic_light_data is : {feature_name}')


    """
    # 矢量集地图数据结构
    Vector set map data structure, including:
    coords: Dict[str, List[<np.ndarray: num_elements, num_points, 2>]].
            The (x, y) coordinates of each point in a map element across map elements per sample.
    traffic_light_data: Dict[str, List[<np.ndarray: num_elements, num_points, 4>]].
    one-shot！！！！！！！！！！！！！！！！！！！
            One-hot encoding of traffic light status for each point in a map element across map elements per sample.
            Encoding: green [1, 0, 0, 0] yellow [0, 1, 0, 0], red [0, 0, 1, 0], unknown [0, 0, 0, 1]
    TODO 可用性 表示数据是否是特征数据 或者是零填充的！！！！
    availabilities: Dict[str, List[<np.ndarray: num_elements, num_points>]].
    # TODO 布尔指示符，表示给定索引处的点是否有特征数据，或是否为零填充
            Boolean indicator of whether feature data is available for point at given index or if it is zero-padded.
    """
    # 筛选需要提取的地图特征！！！！！！！！！！！！
    tensor_output = {}
    # traffic_light_encoding_dim：交通灯状态的编码维度（如绿灯、黄灯、红灯、未知灯状态的独热编码）。
    traffic_light_encoding_dim = LaneSegmentTrafficLightData.encoding_dim()
    # map_features：需要处理的地图特征列表。
    # 需要处理的地图特征列表 需要处理的地图特征列表
    for feature_name in map_features:
        if f"coords.{feature_name}" in list_tensor_data:
            feature_coords = list_tensor_data[f"coords.{feature_name}"]

            feature_tl_data = (
                list_tensor_data[f"traffic_light_data.{feature_name}"]
                if f"traffic_light_data.{feature_name}" in list_tensor_data
                else None
            )
            # avails 表示可用性!!!!!!!!!!!!!!!
            coords, tl_data, avails = convert_feature_layer_to_fixed_size(
                    anchor_state_tensor,
                    feature_coords,
                    feature_tl_data,
                    max_elements[feature_name],
                    max_points[feature_name],
                    traffic_light_encoding_dim,
                    interpolation=interpolation_method  # apply interpolation only for lane features
                    if feature_name
                    in [
                        # .name 提供了枚举成员的名称（字符串形式），可以用于日志记录、字典键、数据序列化等场景。
                        # 例如，在代码中，VectorFeatureLayer.LANE 是一个枚举成员，而 VectorFeatureLayer.LANE.name 是一个字符串 'LANE'，更适合用作字典的键或输出到日志中。
                        VectorFeatureLayer.LANE.name,
                        VectorFeatureLayer.LEFT_BOUNDARY.name,
                        VectorFeatureLayer.RIGHT_BOUNDARY.name,
                        VectorFeatureLayer.ROUTE_LANES.name,
                        VectorFeatureLayer.CROSSWALK.name
                    ]
                    else None,
            )
            # 转换到自车局部坐标系下!!!!!!!!!!!!!!!!
            coords = vector_set_coordinates_to_local_frame(coords, avails, anchor_state_tensor)

            tensor_output[f"vector_set_map.coords.{feature_name}"] = coords
            tensor_output[f"vector_set_map.availabilities.{feature_name}"] = avails

            if tl_data is not None:
                tensor_output[f"vector_set_map.traffic_light_data.{feature_name}"] = tl_data

    """
    Post-precoss the map elements to different map types. Each map type is a array with the following shape.
    N: number of map elements (fixed for a given map feature)
    P: number of points (fixed for a given map feature)
    F: number of features
    """

    for feature_name in map_features:
        if feature_name == "LANE":
            polylines = tensor_output[f'vector_set_map.coords.{feature_name}'].numpy()
            traffic_light_state = tensor_output[f'vector_set_map.traffic_light_data.{feature_name}'].numpy()
            avails = tensor_output[f'vector_set_map.availabilities.{feature_name}'].numpy()
            vector_map_lanes = polyline_process(polylines, avails, traffic_light_state)

        elif feature_name == "CROSSWALK":
            polylines = tensor_output[f'vector_set_map.coords.{feature_name}'].numpy()
            avails = tensor_output[f'vector_set_map.availabilities.{feature_name}'].numpy()
            vector_map_crosswalks = polyline_process(polylines, avails)

        elif feature_name == "ROUTE_LANES":
            polylines = tensor_output[f'vector_set_map.coords.{feature_name}'].numpy()
            avails = tensor_output[f'vector_set_map.availabilities.{feature_name}'].numpy()
            vector_map_route_lanes = polyline_process(polylines, avails)

        else:
            pass
    # TODO 从上面的代码中可以看出，红绿灯数据是在地图元素LANE中获得的
    # 提取三种元素 均是在自车坐标系下  Ego frame
    vector_map_output = {'map_lanes': vector_map_lanes, 'map_crosswalks': vector_map_crosswalks, 'route_lanes': vector_map_route_lanes}

    return vector_map_output

# polylines shape is：(max_elements, max_points, 2)
# 用于处理多段线数据，计算每段线的方向角并将其与坐标数据合并。
def polyline_process(polylines, avails, traffic_light=None):
    # 最终返回的数据最后的特征维度是3或者7 3是 x y heading 
    dim = 3 if traffic_light is None else 7 # 这里之所以是7是因为 如果是交通灯 会使用四位 one-hot表示的红绿灯状态
    new_polylines = np.zeros(shape=(polylines.shape[0], polylines.shape[1], dim), dtype=np.float32)

    for i in range(polylines.shape[0]):
        # TODO 为true的话，表示有特征数据，而不是使用0进行填充的
        if avails[i][0]: 
            polyline = polylines[i]
            polyline_heading = wrap_to_pi(np.arctan2(polyline[1:, 1]-polyline[:-1, 1], polyline[1:, 0]-polyline[:-1, 0]))
            polyline_heading = np.insert(polyline_heading, -1, polyline_heading[-1])[:, np.newaxis]
            if traffic_light is None:
                new_polylines[i] = np.concatenate([polyline, polyline_heading], axis=-1)
            else:
                new_polylines[i] = np.concatenate([polyline, polyline_heading, traffic_light[i]], axis=-1)  

    return new_polylines


########## Path planning functions ##########
# 生成候选路径。它通过深度优先搜索（DFS）在道路网络中寻找可能的路径，并将这些路径转换为多段线形式，供后续路径规划模块使用。
def get_candidate_paths(edges, ego_state, candidate_lane_edge_ids):
    # get all paths
    paths = []
    # edges：起始道路边缘的集合，表示自车所在的起始位置。
    # depth_first_search：深度优先搜索函数，用于在道路网络中搜索所有可能的路径。
    # candidate_lane_edge_ids：候选车道边缘的 ID 列表，限制搜索范围。
    # 作用：从每个起始边缘出发，搜索所有可能的路径，并将结果存储在 paths 中。
    for edge in edges:
        paths.extend(depth_first_search(edge, candidate_lane_edge_ids))

    # extract path polyline
    candidate_paths = []
    # path：深度优先搜索返回的路径，由一系列道路边缘组成。
    # edge.baseline_path.discrete_path：每个边缘的离散化路径点（多段线）。
    # 作用：将路径中的每个边缘的多段线拼接起来，形成完整的路径多段线。
    for i, path in enumerate(paths): # 枚举 index value
        path_polyline = []
        for edge in path:
            path_polyline.extend(edge.baseline_path.discrete_path)
        # 路径点集合
        path_polyline = check_path(np.array(path_to_linestring(path_polyline).coords))
        # 计算自车位置与路径上所有点的距离。
        dist_to_ego = scipy.spatial.distance.cdist([(ego_state.rear_axle.x, ego_state.rear_axle.y)], path_polyline)
        # dist_to_ego.argmin()：找到距离自车最近的路径点索引。
        # 最近点之前的点不要，只要最近点之后的点，可以理解为只要自车前方的点！！！！！！！！！！
        # 作用：将路径修剪为从自车位置开始的部分，去掉自车之前的路径点。
        path_polyline = path_polyline[dist_to_ego.argmin():]
        if len(path_polyline) < 3:
            continue
        # path_len：路径长度，假设每个点之间的距离为 0.25 米。
        path_len = len(path_polyline) * 0.25
        polyline_heading = calculate_path_heading(path_polyline)
        # x y heading
        path_polyline = np.stack([path_polyline[:, 0], path_polyline[:, 1], polyline_heading], axis=1)

        candidate_paths.append((path_len, dist_to_ego.min(), path_polyline)) # 三元组

    # trim paths by length
    max_path_len = max([v[0] for v in candidate_paths])
    # acceptable_path_len：可接受的最小路径长度，限制为 MAX_LEN/2。
    acceptable_path_len = MAX_LEN/2 if max_path_len > MAX_LEN/2 else max_path_len
    # 作用：筛选出长度足够的路径，去掉过短的路径。
    paths = [v for v in candidate_paths if v[0] >= acceptable_path_len]

    return paths

# 函数用于对候选路径进行采样、拟合和评估，生成最终的规划路径。
def generate_paths(paths, obstacles, ego_state):
    new_paths = []
    path_distance = []
    # 三元组
    for (path_len, dist, path_polyline) in paths:
        # sampled_index：根据路径点数选择采样点的索引
        # 对路径进行分段采样，选择若干目标点用于拟合贝塞尔曲线！！！！！！！！！！
        # 类似于Lattice采样！！！！
        if len(path_polyline) > 81:
            sampled_index = np.array([5, 10, 15, 20]) * 4
        elif len(path_polyline) > 61:
            sampled_index = np.array([5, 10, 15]) * 4
        elif len(path_polyline) > 41:
            sampled_index = np.array([5, 10]) * 4
        elif len(path_polyline) > 21:
            sampled_index = [20]
        else:
            sampled_index = [1]
     
        target_states = path_polyline[sampled_index].tolist()
        for j, state in enumerate(target_states):
            # calc_4points_bezier_path：使用贝塞尔曲线拟合自车到目标点的路径。
            # first_stage_path：从自车位置到目标点的拟合路径。
            # second_stage_path：目标点之后的原始路径。
            # 作用：将拟合路径和原始路径拼接，生成完整的分段路径。
            first_stage_path = calc_4points_bezier_path(ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading,
                                                        state[0], state[1], state[2], 3, sampled_index[j])[0]
            # 这个地方 应该是工程化处理了 跟踪到对应的车道中心线 后面的path使用车道线的点！！！
            second_stage_path = path_polyline[sampled_index[j]+1:, :2]
            path_polyline = np.concatenate([first_stage_path, second_stage_path], axis=0)
            new_paths.append(path_polyline)  
            path_distance.append(dist)   

    # evaluate paths
    candiate_paths = {}
    for path, dist in zip(new_paths, path_distance):
        cost = calculate_cost(path, dist, obstacles)
        candiate_paths[cost] = path

    # sort paths by cost
    candidate_paths = []
    # 选择代价最低的前三条路径。
    for cost in sorted(candiate_paths.keys())[:3]:
        path = candiate_paths[cost]
        # 经过曲线拟合 按照索引，均匀选取十个点作为waypoints
        path = post_process(path, ego_state)
        # path 已经在自车坐标系下了
        candidate_paths.append(path)

    return candidate_paths
    
def calculate_cost(path, dist, obstacles):
    # path curvature
    curvature = calculate_path_curvature(path[0:100]) # ？？？？ 工程上的优化？？？？？？
    curvature = np.max(curvature)

    # lane change
    # dist：路径与自车的最小距离
    lane_change = dist # 这个dist实际上是车道中心线到自车的最近距离！！！！！

    # check obstacles
    obstacles = check_obstacles(path[0:100:10], obstacles)
        
    # final cost
    # why？
    cost = 10 * obstacles + 1 * lane_change  + 0.1 * curvature

    return cost

def post_process(path, ego_state):
    path = transform_to_ego_frame(path, ego_state)
    index = np.arange(0, len(path), 10)
    x = path[:, 0][index]
    y = path[:, 1][index]

    # spline interpolation
    rx, ry, ryaw, rk = calc_spline_course(x, y)
    spline_path = np.stack([rx, ry, ryaw, rk], axis=1)
    ref_path = spline_path[:MAX_LEN*10]

    return ref_path
# 参数方程求曲率的公式
def calculate_path_curvature(path):
    dx = np.gradient(path[:, 0])
    dy = np.gradient(path[:, 1])
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    curvature = np.abs(dx * d2y - d2x * dy) / (dx**2 + dy**2)**(3/2)

    return curvature
# 函数用于检查路径是否与障碍物相交，评估路径的安全性。
def check_obstacles(path, obstacles):
    # LineString(path)：
    # 将路径点转换为 LineString 对象（线段），用于几何计算。
    # .buffer((WIDTH/2), cap_style=CAP_STYLE.square)：
    # 对路径进行缓冲，生成一个宽度为 WIDTH 的多边形区域，模拟车辆的占用空间。
    # WIDTH/2：车辆宽度的一半。
    # cap_style=CAP_STYLE.square：缓冲区域的端点样式为方形。
    expanded_path = LineString(path).buffer((WIDTH/2), cap_style=CAP_STYLE.square)

    for obstacle in obstacles:
        obstacle_polygon = obstacle.geometry
        # 检查路径的缓冲区域是否与障碍物相交。
        if expanded_path.intersects(obstacle_polygon):
            return 1

    return 0

# 函数用于从起始道路块中选择与自车位置最近的道路边缘，作为路径搜索的起点。
# 道路块 道路块
def get_candidate_edges(ego_state, starting_block):
    # edges：存储候选道路边缘。
    # edges_distance：存储每个边缘与自车位置的最近距离。
    # ego_point：自车的后轴中心点坐标。
    edges = []
    edges_distance = []
    ego_point = (ego_state.rear_axle.x, ego_state.rear_axle.y)
    # 起始道路块的内部边缘集合。
    for edge in starting_block.interior_edges:
        # 多边形
        edges_distance.append(edge.polygon.distance(Point(ego_point)))
        # 如果边缘与自车的距离小于 4 米，将其加入候选边缘列表。
        if edge.polygon.distance(Point(ego_point)) < 4:
            edges.append(edge)
        
    # if no edge is close to ego, use the closest edge
    if len(edges) == 0:
        edges.append(starting_block.interior_edges[np.argmin(edges_distance)])

    return edges

# depth_first_search 函数用于在道路网络中进行深度优先搜索，生成所有可能的路径。
def depth_first_search(starting_edge, candidate_lane_edge_ids, target_depth=MAX_LEN, depth=0):
    # depth：当前路径的深度（即路径长度）。
    # target_depth：搜索的最大深度。
    # 作用：如果当前路径的深度超过最大深度，返回当前边缘作为路径的终点。
    if depth >= target_depth:
        return [[starting_edge]]
    else:
        traversed_edges = []
        # starting_edge.outgoing_edges：
        # 当前边缘的所有后续边缘。
        # if edge.id in candidate_lane_edge_ids：
        # 过滤掉不在候选车道边缘 ID 列表中的边缘。
        child_edges = [edge for edge in starting_edge.outgoing_edges if edge.id in candidate_lane_edge_ids]

        if child_edges:
            for child in child_edges:
                # 当前边缘的长度，假设每个点之间的距离为 0.25 米。
                # NuPlan 工具包提供了对数据集的解析和操作工具，允许用户访问地图元素（如车道、边缘）及其几何信息。
                # baseline_path：车道的中心线，通常是一个多段线对象，包含一系列离散点。
                # discrete_path：baseline_path 的离散化表示，存储了中心线的点坐标。
                edge_len = len(child.baseline_path.discrete_path) * 0.25 # ？？？？？？？？？？？？？？
                traversed_edges.extend(depth_first_search(child, candidate_lane_edge_ids, depth=depth+edge_len))

        if len(traversed_edges) == 0:
            return [[starting_edge]]

        edges_to_return = []

        for edge_seq in traversed_edges:
            # 将当前边缘与后续路径拼接，形成完整路径。
            edges_to_return.append([starting_edge] + edge_seq)
        # 从当前边缘出发的所有可能路径。    
        return edges_to_return
    
# 坐标系转换 全局-->局部
def transform_to_ego_frame(path, ego_state):
    ego_x, ego_y, ego_h = ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading
    path_x, path_y = path[:, 0], path[:, 1]
    ego_path_x = np.cos(ego_h) * (path_x - ego_x) + np.sin(ego_h) * (path_y - ego_y)
    ego_path_y = -np.sin(ego_h) * (path_x - ego_x) + np.cos(ego_h) * (path_y - ego_y)
    ego_path = np.stack([ego_path_x, ego_path_y], axis=-1) # 列组合

    return ego_path


########## Visulazation functions ##########
def create_ego_raster(vehicle_state):
    # Extract ego vehicle dimensions
    vehicle_parameters = get_pacifica_parameters()
    ego_width = vehicle_parameters.width
    ego_front_length = vehicle_parameters.front_length
    ego_rear_length = vehicle_parameters.rear_length

    # Extract ego vehicle state
    x_center, y_center, heading = vehicle_state[0], vehicle_state[1], vehicle_state[2]
    ego_bottom_right = (x_center - ego_rear_length, y_center - ego_width/2)

    # Paint the rectangle
    rect = plt.Rectangle(ego_bottom_right, ego_front_length+ego_rear_length, ego_width, linewidth=2, color='r', alpha=0.6, zorder=3,
                        transform=mpl.transforms.Affine2D().rotate_around(*(x_center, y_center), heading) + plt.gca().transData, label='ego')
    plt.gca().add_patch(rect)


def create_agents_raster(agents):
    for i in range(agents.shape[0]):
        if agents[i, 0] != 0:
            x_center, y_center, heading = agents[i, 0], agents[i, 1], agents[i, 2]
            agent_length, agent_width = agents[i, 6],  agents[i, 7]
            agent_bottom_right = (x_center - agent_length/2, y_center - agent_width/2)
            # 定义矩形的变换，这涉及到两个部分：旋转和平移。
            # mpl.transforms.Affine2D().rotate_around(*(x_center, y_center), heading) 创建一个二维仿射变换，用于围绕点 (x_center, y_center) 旋转 heading 角度。
            # plt.gca().transData 获取当前坐标轴的数据变换，这通常用于确保矩形按照数据坐标系而非显示坐标系来绘制。
            # 洋红色
            rect = plt.Rectangle(agent_bottom_right, agent_length, agent_width, linewidth=2, color='m', alpha=0.6, zorder=3,
                                transform=mpl.transforms.Affine2D().rotate_around(*(x_center, y_center), heading) + plt.gca().transData, label='agent' if i == 0 else "")
            plt.gca().add_patch(rect)

# 用于绘制地图的栅格化表示，包括车道中心线（lanes）、人行横道（crosswalks）和规划路径上的车道（route_lanes）。
def create_map_raster(lanes, crosswalks, route_lanes):
    # lanes：车道中心线的坐标数据，通常是一个二维数组，形状为 (n, m, 2)，其中 n 是车道的数量，m 是每条车道的点数，2 表示每个点的 (x, y) 坐标。
    # 黑色 蓝色 绿色
    for i in range(lanes.shape[0]):
        lane = lanes[i]
        if lane[0][0] != 0:
            print('plot----lanes')
            plt.plot(lane[:, 0], lane[:, 1], 'k', linewidth=3, label='Lane Centerline' if i == 0 else "") # plot centerline 中心线！！！

    for j in range(crosswalks.shape[0]):
        crosswalk = crosswalks[j]
        if crosswalk[0][0] != 0:
            print('plot----crosswalks')
            plt.plot(crosswalk[:, 0], crosswalk[:, 1], 'b:', linewidth=6, label='Crosswalk' if j == 0 else "") # plot crosswalk

    for k in range(route_lanes.shape[0]):
        route_lane = route_lanes[k]
        if route_lane[0][0] != 0:
            print('plot----route_lanes')
            plt.plot(route_lane[:, 0], route_lane[:, 1], 'g', linewidth=8, label='Route Lane' if k == 0 else "") # plot route_lanes


def draw_trajectory(ego_trajectory, agent_trajectories):
    # plot ego 红色 紫色
    plt.plot(ego_trajectory[:, 0], ego_trajectory[:, 1], 'r:p', linewidth=6, zorder=3)

    # plot others
    for i in range(agent_trajectories.shape[0]):
        if agent_trajectories[i, -1, 0] != 0:
            trajectory = agent_trajectories[i]
            # 设置矩形的绘图顺序，zorder值越大，矩形越在上层绘制。
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'm--^', linewidth=8, zorder=3)


def draw_plans(trajectory_plans, stage=1):
    # 黄色虚线 青色虚线
    f = 'y--' if stage == 1 else 'c--+'
 
    for i in range(trajectory_plans.shape[0]):
        trajectory = trajectory_plans[i]
        plt.plot(trajectory[:, 0], trajectory[:, 1], f, linewidth=2, zorder=5-stage, label='' if i !=  0 else ('first' if stage == 0 else "second"))