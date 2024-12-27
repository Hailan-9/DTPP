import math
import time
import matplotlib.pyplot as plt
from shapely import Point, LineString
from planner_utils import *
from obs_adapter import *
from trajectory_tree_planner import TreePlanner
from scenario_tree_prediction import *

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring

# NOTE 定义了一个名为 Planner 的类，继承自 AbstractPlanner，实现了一个基于树搜索的轨迹规划器（TreePlanner）。
# NOTE 代码的主要功能是初始化规划器，加载模型，处理地图和路由信息，并在每次仿真迭代中生成未来的轨迹。
# TODO 必须继承AbstractPlanner！！！！继承自 AbstractPlanner，是 NuPlan 仿真框架中的一个抽象类，用于实现自定义规划器。
class Planner(AbstractPlanner):
    def __init__(self, model_path, device):
        self._future_horizon = T # [s] 
        self._step_interval = DT # [s]
        self._N_points = int(T/DT)
        self._model_path = model_path
        self._device = device

    def name(self) -> str:
        return "DTPP Planner"
    
    def observation_type(self):
        # NOTE observation_type: 返回感知数据的类型，这里指定为 DetectionsTracks，表示感知数据包含检测的轨迹信息（如其他车辆或行人的位置和速度）。
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization):
        # NOTE initialize: 在仿真开始时调用，用于初始化规划器。
        # initialization.map_api: 地图接口，用于获取地图相关信息（如车道、路口等）。
        # initialization.mission_goal: 当前任务的目标位置。
        # TODO initialization.route_roadblock_ids: route路径中的路段（roadblock）ID 列表。
        # _initialize_route_plan: 调用自定义方法，基于路段 ID 初始化路径规划相关信息。
        # _initialize_model: 调用自定义方法，加载规划器的神经网络模型。
        # TreePlanner: 初始化树搜索规划器，传入设备、编码器和解码器。
        self._map_api = initialization.map_api
        self._goal = initialization.mission_goal
        self._route_roadblock_ids = initialization.route_roadblock_ids
        self._initialize_route_plan(self._route_roadblock_ids)
        self._initialize_model()
        # TODO
        self._trajectory_planner = TreePlanner(self._device, self._encoder, self._decoder)
    # NOTE 初始化模型
    def _initialize_model(self):
        # 加载模型文件。
        model = torch.load(self._model_path, map_location=self._device)
        # NOTE Encoder 和 Decoder: 神经网络的编码器和解码器，用于特征提取和轨迹生成。
        self._encoder = Encoder()
        self._encoder.load_state_dict(model['encoder'])
        self._encoder.to(self._device)
        # eval: 设置模型为推理模式，禁用梯度计算。
        self._encoder.eval()
        self._decoder = Decoder()
        self._decoder.load_state_dict(model['decoder'])
        self._decoder.to(self._device)
        self._decoder.eval()
    # 初始化路径规划
    def _initialize_route_plan(self, route_roadblock_ids):
        self._route_roadblocks = [] # route_roadblock_ids: 全局导航路径中的路段 ID 列表。

        for id_ in route_roadblock_ids:
            # NOTE 从地图中获取路段（ROADBLOCK）或路段连接器（ROADBLOCK_CONNECTOR）。
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            # NOTE or：block为true，等于block，否则等于self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            block = block or self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            # TODO STEP1:
            self._route_roadblocks.append(block)
        # _candidate_lane_edge_ids: 存储所有候选导航（route)车道边缘的 ID。
        self._candidate_lane_edge_ids = [
            # TODO interior_edges: 获取路段的内部车道边缘。
            edge.id for block in self._route_roadblocks if block for edge in block.interior_edges
        ]
    # 计算规划的轨迹
    def compute_planner_trajectory(self, current_input: PlannerInput):
        # Extract iteration, history, and traffic light
        # NOTE iteration: 当前仿真迭代的索引。
        # history: 包含当前和过去的车辆状态。
        # traffic_light_data: 当前交通信号灯的状态。
        # ego_state: 自车的当前状态。
        # observation: 当前感知数据。
        # 在 NuPlan 中，iteration.index 表示当前仿真或规划迭代的索引（Iteration Index）。这是一个整数值，
        # 用于标识当前规划器在整个仿真过程中所处的时间步（或迭代次数）。它通常用于记录或调试，以便知道当前规划器运行到了哪一步。
        # TODO 当前迭代的索引（iteration.index）。
        iteration = current_input.iteration.index
        history = current_input.history
        traffic_light_data = list(current_input.traffic_light_data)
        ego_state, observation = history.current_state

        # Construct input features
        # 记录当前时间，用于计算规划时间。
        start_time = time.perf_counter()
        # TODO 自定义函数，将历史状态、交通信号灯数据等转换为模型的输入特征。
        features = observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)

        # NOTE Get starting block
        starting_block = None
        # NOTE 自车当前的后轴位置
        # TODO _route_roadblocks: 遍历路径中的每个路段。
        # edge.polygon.distance: 计算自车位置到车道边缘的距离。
        # starting_block: 找到与自车最近的路段。
        cur_point = (ego_state.rear_axle.x, ego_state.rear_axle.y)
        closest_distance = math.inf
        # TODO STEP2
        for block in self._route_roadblocks:
            for edge in block.interior_edges:
                distance = edge.polygon.distance(Point(cur_point))
                if distance < closest_distance:
                    starting_block = block
                    closest_distance = distance

            if np.isclose(closest_distance, 0):
                break
        
        # Get traffic light lanes
        # traffic_light_data: 遍历所有交通信号灯信息。
        # TrafficLightStatusType.RED: 检查信号灯是否为红灯。
        # TODO traffic_light_lanes: 存储所有红灯车道。
        traffic_light_lanes = []
        for data in traffic_light_data:
            id_ = str(data.lane_connector_id)
            if data.status == TrafficLightStatusType.RED and id_ in self._candidate_lane_edge_ids:
                lane_conn = self._map_api.get_map_object(id_, SemanticMapLayer.LANE_CONNECTOR)
                traffic_light_lanes.append(lane_conn)

        # Tree policy planner
        # self._trajectory_planner.plan: 调用树搜索规划器计算轨迹。
        # 异常处理: 如果规划失败，返回一个零轨迹。
        try:
            plan = self._trajectory_planner.plan(iteration, ego_state, features, starting_block, self._route_roadblocks, 
                                             self._candidate_lane_edge_ids, traffic_light_lanes, observation)
        except Exception as e:
            print("Error in planning")
            print(e)
            plan = np.zeros((self._N_points, 3))
            
        # Convert relative poses to absolute states and wrap in a trajectory object
        # transform_predictions_to_states: 将规划器的相对轨迹转换为绝对状态。
        # InterpolatedTrajectory: 将状态封装为可插值的轨迹对象。
        # return trajectory: 返回生成的轨迹。
        states = transform_predictions_to_states(plan, history.ego_states, self._future_horizon, DT)
        trajectory = InterpolatedTrajectory(states)
        print(f'Step {iteration+1} Planning time: {time.perf_counter() - start_time:.3f} s')

        return trajectory