import os
import math
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_utils import *
from trajectory_tree_planner import *
from common_utils import get_filter_parameters, get_scenario_map

from nuplan.planning.utils.multithreading.worker_pool import Task
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping



# define data processor
# 在 Python 2 中，继承自 object 表示这是一个新式类。而在 Python 3 中，所有类默认都是新式类，因此不需要显式地继承自 object！
# object 是所有类的根类。使用 object 作为基类可以确保 DataProcessor 类具有所有基本的对象功能（如实例化、属性管理、方法等）。
class DataProcessor(object):
    # 场景是nuplan中的一个重要概念
    def __init__(self, scenarios):
        self._scenarios = scenarios
        # 2s + 8s = 10s
        self.past_time_horizon = 2 # [seconds]
        self.num_past_poses = 10 * self.past_time_horizon # 0.1s一个轨迹点
        # 两个阶段，对应的时刻分别是3s和8s
        self.future_time_horizon = 8 # [seconds]
        self.max_target_speed = 15 # [m/s]
        self.first_stage_horizon = 3 # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon
        self.num_agents = 20 # 最多只考虑附近20个他车
        # 所需提取的地图特征的名称
        self._map_features = ['LANE', 'ROUTE_LANES', 'CROSSWALK'] # name of map features to be extracted.
        # 在处理特征层（feature layer）时，每个特征层中允许提取的最大元素数量
        self._max_elements = {'LANE': 40, 'ROUTE_LANES': 10, 'CROSSWALK': 5} # maximum number of elements to extract per feature layer.
        # 在特征层中，每个特征允许提取的最大点数的限制
        self._max_points = {'LANE': 50, 'ROUTE_LANES': 50, 'CROSSWALK': 30} # maximum number of points per feature to extract per feature layer.
        # 查询半径的范围是相对于当前姿态（或位置）
        self._radius = 80 # [m] query radius scope relative to the current pose.
        # 在处理地图元素时，为了保持固定大小的地图元素，在进行插值时使用的插值方法
        self._interpolation_method = 'linear' # Interpolation method to apply when interpolating to maintain fixed size map elements.

    # 定义一个方法用于获取自车的状态
    # 张量是深度学习的基础 tensor
    def get_ego_agent(self):
        self.anchor_ego_state = self.scenario.initial_ego_state
        # 获取过去的历史轨迹
        past_ego_states = self.scenario.get_ego_past_trajectory(
            iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
        )
        # 将过去的状态与当前状态结合。
        sampled_past_ego_states = list(past_ego_states) + [self.anchor_ego_state]
        # 将 sampled_past_ego_states 转换为张量tensor格式；为了后续处理
        # NOTE shape is (N, 7) this is 21, 7
        past_ego_states_tensor = sampled_past_ego_states_to_tensor(sampled_past_ego_states)

        past_time_stamps = list(
            self.scenario.get_past_timestamps(
                iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
            )
        ) + [self.scenario.start_time]
        # 也转化为张量形式
        past_time_stamps_tensor = sampled_past_timestamps_to_tensor(past_time_stamps)

        return past_ego_states_tensor, past_time_stamps_tensor
    
    def get_neighbor_agents(self):
        # 这个属性可能包含当前时刻的所有被跟踪的对象（例如，车辆、行人等）。
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects
        # 列表推导式
        past_tracked_objects = [
            # 获取tracked_objects的成员变量（字段）tracked_objects
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_past_tracked_objects(
                iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
            )
        ]

        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        past_tracked_objects_tensor_list, past_tracked_objects_types = \
            sampled_tracked_objects_to_tensor_list(sampled_past_observations)

        return past_tracked_objects_tensor_list, past_tracked_objects_types
    # 得到地图
    def get_map(self):        
        ego_state = self.scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y) # x y位姿点
        # 获取路障 ID
        # 用于获取当前场景（scenario）中规划路径（route）上所有的 roadblock IDs
        # 一个 roadblock 通常包含多个车道段（lane segments）。
        # 它是一个逻辑上的区域，表示一段连续的道路。
        # 每个 roadblock 都有一个唯一的 ID，用于标识它。
        # 在自动驾驶中，route 是指车辆从起点到终点的规划路径。这个路径通常由多个 roadblock 和 lane segments 组成。route 是车辆在地图中行驶的目标路径。

        # Route 和 Roadblock 的关系：
        # 一个 route 是由多个 roadblock 组成的。
        # 例如，如果车辆需要从 A 点到 B 点，可能需要经过多个 roadblock，这些 roadblock 按照一定的顺序排列，形成了车辆的行驶路径。
        # TODO 
        route_roadblock_ids = self.scenario.get_route_roadblock_ids()
        # 获取交通信号灯状态
        traffic_light_data = self.scenario.get_traffic_light_status_at_iteration(0)
        # 获取邻域向量集地图 路块的id
        
        # NOTE 字典类型的 得到了包含自车附近车道的中心线 导航车道的中心线 以及人行横道的坐标点 都是MapObjectPolylines类型的数据
        # TODO to_vector函数：Collection of map object polylines, each represented as a list of x, y coords
        # NOTE [num_elements, num_points_in_element (variable size), 2].
        coords, traffic_light_data = get_neighbor_vector_set_map(
            self.map_api, self._map_features, ego_coords, self._radius, route_roadblock_ids, traffic_light_data
        )
        # 生成一个向量地图，存储在 vector_map 中
        # TODO 就是一个字典
        # TODO 已经转化在自车坐标系下
        vector_map = map_process(ego_state.rear_axle, coords, traffic_light_data, self._map_features, 
                                 self._max_elements, self._max_points, self._interpolation_method)

        return vector_map
    # 用于获取代理的未来轨迹，并将其转换为相对坐标系统，以便后续处理或分析
    def get_ego_agent_future(self):
        current_absolute_state = self.scenario.initial_ego_state
        # 获取未来轨迹 使用参数指定当前迭代（0）、未来位置的样本数量和时间范围
        trajectory_absolute_states = self.scenario.get_ego_future_trajectory(
            iteration=0, num_samples=self.num_future_poses, time_horizon=self.future_time_horizon
        )
        # 在自车坐标系下
        # Get all future poses of the ego relative to the ego coordinate system
        trajectory_relative_poses = convert_absolute_to_relative_poses(
            current_absolute_state.rear_axle, [state.rear_axle for state in trajectory_absolute_states]
        )

        return trajectory_relative_poses
    # 用于指定要获取未来信息的代理的索引。
    def get_neighbor_agents_future(self, agent_index):
        current_ego_state = self.scenario.initial_ego_state
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects

        # Get all future poses of of other agents
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_future_tracked_objects(
                iteration=0, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
            )
        ]
        # shape: time agent 位姿
        sampled_future_observations = [present_tracked_objects] + future_tracked_objects
        # NOTE 每一帧中的agent的顺序是不一样的
        # TODO 第一帧 也就是当前时刻，按照agents出现的顺序依次赋值他们的token（agent的标识符）为0 1 2 3 4…… 在 函数pad_agent_states_with_zeros用到了。
        # TODO 因为我们只取在第一帧中出现的障碍物，所以需要使用第一帧出现的token 来筛选出这些agents 
        future_tracked_objects_tensor_list, _ = sampled_tracked_objects_to_tensor_list(sampled_future_observations)
        # 函数内部有坐标转换等等 转到到当前自车坐标系下，也就是Ego坐标系
        agent_futures = agent_future_process(current_ego_state, future_tracked_objects_tensor_list, self.num_agents, agent_index)

        return agent_futures
    # 得到自车的候选轨迹
    # 包括第一阶段的前3s 和 第二阶段的前8s轨迹
    def get_ego_candidate_trajectories(self):
        planner = SplinePlanner(self.first_stage_horizon, self.future_time_horizon)

        # Gather information about the environment
        # NOTE 获得导航路段ids
        route_roadblock_ids = self.scenario.get_route_roadblock_ids() # 得到路障id
        observation = self.scenario.get_tracked_objects_at_iteration(0)
        ego_state = self.scenario.initial_ego_state
        # 导航路段的列表
        route_roadblocks = []

        for id_ in route_roadblock_ids:
            # SemanticMapLayer：通常指的是在地图或环境建模中，用于表示和处理语义信息的层。这种层的主要功能是将环境中的对象和区域标注为具有特定意义的类别，使得计算机能够理解和处理这些信息。
            block = self.map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            # 左边真，则返回左边的值，否则，返回右边的值
            block = block or self.map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            print(f'hailan------------------block type is {type(block)}')
            route_roadblocks.append(block)
        # TODO 嵌套列表推导式 每一个导航路段的内部边的ids！！！！！
        # TODO 候选车道边ids
        candidate_lane_edge_ids = [edge.id for block in route_roadblocks if block for edge in block.interior_edges]

        # Get obstaclesblock
        object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.BARRIER,
                        TrackedObjectType.CZONE_SIGN, TrackedObjectType.TRAFFIC_CONE,
                        TrackedObjectType.GENERIC_OBJECT]
        objects = observation.tracked_objects.get_tracked_objects_of_types(object_types)
        obstacles = []
        # TODO 添加障碍物
        for obj in objects:
            if obj.box.geometry.distance(ego_state.car_footprint.geometry) > 30: # 太远的障碍物不考虑
                continue
            # 如果车辆的速度小于0.01，或者不是车辆类型，它的边界框将被添加到 obstacles 列表中
            if obj.tracked_object_type == TrackedObjectType.VEHICLE:
                if obj.velocity.magnitude() < 0.01:
                    obstacles.append(obj.box)
            else:
                obstacles.append(obj.box)
        # Get starting block 起始地图区块
        # 用于存储距离代理最近的路障区块。
        starting_block = None
        cur_point = (self.scenario.initial_ego_state.rear_axle.x, self.scenario.initial_ego_state.rear_axle.y) # 元组
        # 用于后续计算最近的路障区块。
        closest_distance = math.inf
        # NOTE 遍历 route_roadblocks 中的每个区块和其内部边缘 找到最近的起始路段块
        for block in route_roadblocks:
            # TODO 好奇怪 为啥搜不到这个interior_edges成员变量！！！！！！！！！！！！！！！！！！！！！
            for edge in block.interior_edges:
                # 计算点到多边形的最短距离 内部为0
                # TODO  这个地方可以可视化出来！！！！！！！！！！！！！！！
                distance = edge.polygon.distance(Point(cur_point))
                if distance < closest_distance:
                    starting_block = block
                    closest_distance = distance

            if np.isclose(closest_distance, 0):
                break

        # Get starting edges
        # NOTE 传入当前代理状态和起始区块，获取附近一定距离内的候选边缘！！！
        edges = get_candidate_edges(ego_state, starting_block)
        # 传入候选边缘、当前代理状态和候选车道边缘 ID，获取所有候选路径。
        # TODO 候选轨迹 也就是附近一定距离内的车道中心线的离散点path
        candidate_paths = get_candidate_paths(edges, ego_state, candidate_lane_edge_ids)
        # 传入候选路径、障碍物列表和当前代理状态，生成自车实际可行的路径。 
        # TODO 采样得到的，类似于Lattice planner 经过三次样条平滑后的轨迹！！！
        paths = generate_paths(candidate_paths, obstacles, ego_state)
        # 用于返回第一个真值
        speed_limit = edges[0].speed_limit_mps or self.max_target_speed

        # Initial tree (root node)
        # traj: x, y, heading, velocity, acceleration, curvature, time
        # 在自车坐标系下，自车的x y heading均是0 ，velocity是当前速度，acceleration是当前加速度，curvature是0，time是0
        state = torch.tensor([[0, 0, 0, ego_state.dynamic_car_state.rear_axle_velocity_2d.x, 
                               ego_state.dynamic_car_state.rear_axle_acceleration_2d.x, 0, 0]])
        # 自车的轨迹树
        tree = TrajTree(state, None, 0)

        # 1st stage expand 时刻是第3s
        tree.expand_children(paths, self.first_stage_horizon, speed_limit, planner)
        leaves = TrajTree.get_children(tree)
        # 切片操作符，1 表示切片的起始索引（从0开始计数），: 表示切片一直到序列的末尾。
        # NOTE 不包括自车当前位置 ，所以从索引1开始
        first_trajs = np.stack([leaf.total_traj[1:].numpy() for leaf in leaves]).astype(np.float32)

        # 2nd stage expand 时刻是第8s
        for leaf in leaves:
            leaf.expand_children(paths, self.future_time_horizon - self.first_stage_horizon, speed_limit, planner)

        # Get all leaves
        leaves = TrajTree.get_children(leaves)
        second_trajs = np.stack([leaf.total_traj[1:].numpy() for leaf in leaves]).astype(np.float32)
        # shape is 1-leafs 2-times 3-traj(: x, y, heading, velocity, acceleration, curvature, time)
        return first_trajs, second_trajs
    # 接受一个参数 data，通常是一个字典，包含了绘图所需的各种数据。
    def plot_scenario(self, data):
        # 用于绘制地图的基础层
        # Create map layers
        print("hailan----------plot_scnario start")
        create_map_raster(data['map_lanes'], data['map_crosswalks'], data['route_lanes'])

        # 提取自我代理过去轨迹的最后一个数据点。
        # Create agent layers
        create_ego_raster(data['ego_agent_past'][-1])
        # 选择二维数组中所有行的最后一个元素。这里的冒号 : 表示选择所有的行，而 -1 表示选择每行的最后一个元素。
        create_agents_raster(data['neighbor_agents_past'][:, -1])

        # Draw past and future trajectories
        # [:1]：所有行的第一个到最后一个，索引从0开始
        draw_trajectory(data['ego_agent_past'], data['neighbor_agents_past'][:1])
        draw_trajectory(data['ego_agent_future'], data['neighbor_agents_future'][:1])

        # Draw candidate trajectories
        draw_plans(data['first_stage_ego_trajectory'], 1)
        draw_plans(data['second_stage_ego_trajectory'], 2)

        plt.legend(loc='upper right')  # 图例位置可以调整，例如 'upper right', 'lower left' 等  
        plt.gca().set_aspect('equal')
        plt.grid(True)  # 添加网格线  
        plt.tight_layout()
        # 当你调用 plt.show() 时，matplotlib 会打开一个图形窗口，显示绘制的图形。
        # 在这个窗口关闭之前，程序会暂停在 plt.show() 这一行，后续的代码不会被执行。
        # 只有当你手动关闭图形窗口后，程序才会继续运行。
        plt.show()
        print("hailan----------plot_scnario finish")
    # token的概念需要理解掌握
    def save_to_disk(self, dir, data):
        # 将 data 字典中的所有数组保存到一个 .npz 文件中
        # token 的值，它可能是一个标识符或某种标记。
        # *表示解包元组 **表示解包字典
        # ** 解包字典 变为元组
        np.savez(f"{dir}/{data['map_name']}_{data['token']}.npz", **data)

    def work(self, save_dir, debug=False):
        for scenario in tqdm(self._scenarios):
            print("------------------------------------------")
            print(f"Processing {scenario}")
            print("******************************************")
            
            map_name = scenario._map_name
            # 获取当前场景的唯一标识符（token）
            token = scenario.token
            self.scenario = scenario
            self.map_api = scenario.map_api
            print(scenario)

            # get agent past tracks
            ego_agent_past, time_stamps_past = self.get_ego_agent()
            # TODO neighbor_agents_past的shape is (time_steps, num_agents, dim) 每个时刻的，agent的顺序不是一样的！！！
            neighbor_agents_past, neighbor_agents_types = self.get_neighbor_agents()
            # 转换为自车坐标系下
            # TODO neighbor_indices 是当前时刻下，按照距离自车从近到远的顺序的其他交通参与者（Agents）的索引index
            ego_agent_past, neighbor_agents_past, neighbor_indices = \
                    agent_past_process(ego_agent_past, time_stamps_past, neighbor_agents_past, neighbor_agents_types, self.num_agents)
            # 获取向量集地图
            # get vector set map
            vector_map = self.get_map()

            # get agent future tracks
            # TODO 这是自车轨迹和他车轨迹的ground truth！
            ego_agent_future = self.get_ego_agent_future()
            neighbor_agents_future = self.get_neighbor_agents_future(neighbor_indices)

            # get candidate trajectories
            try:
                first_stage_trajs, second_stage_trajs = self.get_ego_candidate_trajectories()
            except:
                print(f"Error in {map_name}_{token}")
                continue

            # check if the candidate trajectories are valid
            # :2 表示只取前两个坐标（x 和 y），即位置坐标
            # 表示在第一阶段的最后一个时间点（假设每个时间点有 10 个采样）。
            # None：这是一个 Python 的 None 值，用于增加数组的维度。这里将其增加到第一个维度，使得索引变为二维。
            # NOTE [:, -1, :2] 选择所有叶子节点的最后一个时刻的前两个元素（位置点）
            # axis=-1：这是 np.linalg.norm 函数的一个参数，指定在哪个轴上计算范数。
            # 在这里，-1 表示在最后一个轴上计算，即计算两个向量在 x 和 y 坐标上的差异。
            # TODO 
            expert_error_1 = np.linalg.norm(ego_agent_future[None, self.first_stage_horizon*10-1, :2]
                                            - first_stage_trajs[:, -1, :2], axis=-1)
            expert_error_2 = np.linalg.norm(ego_agent_future[None, self.future_time_horizon*10-1, :2]
                                            - second_stage_trajs[:, -1, :2], axis=-1)       
            # TODO 判断是否构建出一个有效的数据
            if np.min(expert_error_1) > 1.5 and np.min(expert_error_2) > 4:
                continue
            
            # sort the candidate trajectories
            # 元素从小到大的索引数组。
            # 使用从 np.argsort() 返回的索引对 first_stage_trajs 数组进行重新排序。
            # 这将返回一个新的数组，其中轨迹按照对应的误差从小到大排列。

            # NOTE 找出轨迹树中的最后一个位置点 接近 gt轨迹的距离 从近到远进行排序
            # NOTE III-D 模型训练提到了这个地方
            first_stage_trajs = first_stage_trajs[np.argsort(expert_error_1)]
            second_stage_trajs = second_stage_trajs[np.argsort(expert_error_2)]            

            # gather data
            # 创建字典 data
            data = {"map_name": map_name, "token": token, "ego_agent_past": ego_agent_past, "ego_agent_future": ego_agent_future, 
                    "first_stage_ego_trajectory": first_stage_trajs, "second_stage_ego_trajectory": second_stage_trajs,
                    "neighbor_agents_past": neighbor_agents_past, "neighbor_agents_future": neighbor_agents_future}
            # 将地图的字典类型的数据存入
            data.update(vector_map)

            # visualization
            if debug:
                self.plot_scenario(data)

            # save to disk
            # NOTE data是一个字典 每个key对应的value都是numpy数组
            # NOTE 只有 NumPy 数组可以直接被保存为 .npz 格式的文件。PyTorch 张量（tensor）不能直接保存为 .npz 文件，因为 .npz 文件格式不认识 PyTorch 张量。
            # NOTE 如果您想将 PyTorch 张量保存为 .npz 文件，您需要先将张量转换为 NumPy 数组，然后再使用 np.savez
            self.save_to_disk(save_dir, data)


if __name__ == "__main__":
    print("hailan------------data-process!!!")
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--debug', action="store_true", help='if visualize the data output', default=False)
    parser.add_argument('--data_path', type=str, help='path to the data')
    parser.add_argument('--map_path', type=str, help='path to the map')    
    parser.add_argument('--save_path', type=str, help='path to save the processed data')
    parser.add_argument('--total_scenarios', type=int, help='total number of scenarios', default=None)

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    map_version = "nuplan-maps-v1.0"
    # 场景映射
    scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, None, None, map_version, scenario_mapping=scenario_mapping)
    # 在 Python 中，* 用于解包（unpacking）一个可迭代对象（如列表或元组）作为函数的参数。这里，它用于将 get_filter_parameters 函数返回的参数解包并传递给 ScenarioFilter 的构造函数。
    scenario_filter = ScenarioFilter(*get_filter_parameters(num_scenarios_per_type=30000, 
                                                            limit_total_scenarios=args.total_scenarios))
    # 多线程
    worker = SingleMachineParallelExecutor(use_process_pool=True)
    # 得到场景
    # type is List[NuPlanScenario]
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Total number of training scenarios: {len(scenarios)}")
    
    del worker, builder, scenario_filter
    processor = DataProcessor(scenarios)
    processor.work(args.save_path, debug=args.debug)
