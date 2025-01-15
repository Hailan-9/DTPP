import torch
import scipy
import random
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from shapely import Point, LineString
from shapely.geometry.base import CAP_STYLE
from path_planner import calc_spline_course
from bezier_path import calc_4points_bezier_path
from collections import defaultdict
from spline_planner import SplinePlanner
from torch.nn.utils.rnn import pad_sequence
from scenario_tree_prediction import *
from planner_utils import *
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring
from debug_utils import *


class TrajTree:
    # traj: x, y, heading, velocity, acceleration, curvature, time
    def __init__(self, traj, parent, depth):
        self.traj = traj
        self.state = traj[-1, :5]
        self.children = list()
        self.parent = parent
        self.depth = depth
        self.attribute = dict()
        if parent is not None:
            self.total_traj = torch.cat((parent.total_traj, traj), 0)
        else:
            self.total_traj = traj

    def expand(self, child):
        self.children.append(child)

    def expand_set(self, children):
        self.children += children

    def expand_children(self, paths, horizon, speed_limit, planner):
        # NOTE x y yaw v a r t
        # NOTE shape：paths.size timesteps 7
        trajs = planner.gen_trajectories(self.state, horizon, paths, speed_limit, self.isroot())
        children = [TrajTree(traj, self, self.depth + 1) for traj in trajs]
        self.expand_set(children)

    def isroot(self):
        return self.parent is None

    def isleaf(self):
        return len(self.children) == 0

    def get_subseq_trajs(self):
        return [child.traj for child in self.children]
    # TODO 
    def get_all_leaves(self, leaf_set=[]):
        if self.isleaf():
            print_tensor_log(self.state, "self.state", DEBUG)
            print_tensor_shape_log(self.state, "self.state", DEBUG)
            leaf_set.append(self)
        else:
            for child in self.children:
                leaf_set = child.get_all_leaves(leaf_set)

        return leaf_set

    @staticmethod
    def get_children(obj):
        if isinstance(obj, TrajTree):
            return obj.children
        
        elif isinstance(obj, list):
            children = [node.children for node in obj]
            # TODO 
            # itertools.chain.from_iterable是Python的itertools模块中的一个函数，用于将多个可迭代对象（如列表）连接成一个迭代器，这个迭代器依次产生所有可迭代对象的元素。
            # children（在这一步之前）是一个列表，其中包含了多个子节点列表。
            # itertools.chain.from_iterable(children)将这些子节点列表连接起来，形成一个迭代器，这个迭代器会依次产生所有子节点。
            # list(...)将这个迭代器转换成一个列表。
            children = list(itertools.chain.from_iterable(children))
            return children
        
        else:
            raise TypeError("obj must be a TrajTree or a list")
    # 递归绘制轨迹树。
    def plot_tree(self, ax=None, msize=12):
        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 10))
        state = self.state.cpu().detach().numpy()
        
        ax.plot(state[0], state[1], marker="o", color="b", markersize=msize)

        if self.traj.shape[0] > 1:
            if self.parent is not None:
                traj_l = torch.cat((self.parent.traj[-1:],self.traj),0)
                traj = traj_l.cpu().detach().numpy()
            else:
                traj = self.traj.cpu().detach().numpy()

            ax.plot(traj[:, 0], traj[:, 1], color="k")

        for child in self.children:
            child.plot_tree(ax)

        return ax
    # 获取每层节点的子节点索引，用于构建张量形式的树结构。
    @staticmethod
    def get_children_index_torch(nodes):
        indices = dict()
        for depth, nodes_d in nodes.items():
            if depth+1 in nodes:
                childs_d = nodes[depth+1]
                indices_d = list()
                for node in nodes_d:
                    indices_d.append(torch.tensor([childs_d.index(child) for child in node.children]))
                indices[depth] = pad_sequence(indices_d, batch_first=True, padding_value=-1)

        return indices
    
    @staticmethod
    def get_nodes_by_level(obj, depth, nodes=None, trim_short_branch=True):
        assert obj.depth <= depth
        if nodes is None:
            nodes = defaultdict(lambda: list())

        if obj.depth == depth:
            nodes[depth].append(obj)

            return nodes, True
        else:
            if obj.isleaf():
                return nodes, False
            else:
                flag = False
                children_flags = dict()
                for child in obj.children:
                    nodes, child_flag = TrajTree.get_nodes_by_level(child, depth, nodes)
                    children_flags[child] = child_flag
                    flag = flag or child_flag

                if trim_short_branch:
                    obj.children = [child for child in obj.children if children_flags[child]]
                if flag:
                    nodes[obj.depth].append(obj)

                return nodes, flag
            
# TODO 这里面的edge是啥意思 始终没有完全理解
# TODO 定义了一个名为 TreePlanner 的类，用于自动驾驶中的轨迹规划。它采用树搜索的方式生成候选路径，并结合神经网络模型对路径进行预测和评估，最终选择最优路径作为规划结果。
# 路径规划的核心类，负责生成候选路径、扩展轨迹树、评估路径质量并选择最优路径
class TreePlanner:
    def __init__(self, device, encoder, decoder, n_candidates_expand=5, n_candidates_max=30):
        # NOTE encoder 和 decoder: 神经网络模型的编码器和解码器，用于环境特征提取和轨迹预测。
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_path_len = 120 # [m]
        # NOTE 树搜索的目标深度(m作为单位)
        self.target_depth = MAX_LEN # [m]
        self.target_speed = 13 # [m/s]
        self.horizon = 8 # [s] 3 + 5
        self.first_stage_horizon = 3 # [s]
        # NOTE 树搜索扩展阶段的候选轨迹数量。
        self.n_candidates_expand = n_candidates_expand # second stage
        # NOTE 最多允许的候选路径数量。
        self.n_candidates_max = n_candidates_max # max number of candidates
        # NOTE 使用样条曲线规划路径。
        self.planner = SplinePlanner(self.first_stage_horizon, self.horizon)  
    # TODO 得到候选的path，其实就是车道离散的中心线                            
    def get_candidate_paths(self, edges):
        # get all paths
        # NOTE edges: 当前自车所在的候选车道边缘。
        # NOTE depth_first_search: 深度优先搜索，基于车道拓扑生成所有可能的路径。
        # NOTE paths: 存储所有候选路径。
        paths = []
        # TODO 看明白了 edge之间相连的方式也是树的形式 tree-structure
        for edge in edges:
            paths.extend(self.depth_first_search(edge))

        # extract path polyline
        # NOTE path_polyline: 将路径中的每个车道边缘的离散点拼接成完整的路径多段线。
        # NOTE check_path: 检查路径的有效性（例如，是否有重复点或不连续的点）。
        candidate_paths = []
        # TODO step2
        for i, path in enumerate(paths):
            path_polyline = []
            for edge in path:
                # TODO 车道中心线
                path_polyline.extend(edge.baseline_path.discrete_path)

            path_polyline = check_path(np.array(path_to_linestring(path_polyline).coords))
            dist_to_ego = scipy.spatial.distance.cdist([self.ego_point], path_polyline)
            # TODO 从距离自车最近的点开始截取路径。
            path_polyline = path_polyline[dist_to_ego.argmin():]
            if len(path_polyline) < 3:
                continue

            path_len = len(path_polyline) * 0.25
            polyline_heading = calculate_path_heading(path_polyline)
            path_polyline = np.stack([path_polyline[:, 0], path_polyline[:, 1], polyline_heading], axis=1)
            # NOTE 路径的长度 路径到自车的最近距离 x y heading
            candidate_paths.append((path_len, dist_to_ego.min(), path_polyline))

        # trim paths by length
        # max_path_len: 候选路径的最大长度。
        # acceptable_path_len: 如果最大路径长度超过 MAX_LEN/2，则只接受长度大于等于 MAX_LEN/2 的路径。
        # paths: 修剪后的候选路径。
        max_path_len = max([v[0] for v in candidate_paths])
        acceptable_path_len = MAX_LEN/2 if max_path_len > MAX_LEN/2 else max_path_len
        paths = [v for v in candidate_paths if v[0] >= acceptable_path_len]

        return paths
    # 获取候选车道边缘
    def get_candidate_edges(self, starting_block):
        # NOTE starting_block: 自车当前所在的路段。
        edges = []
        edges_distance = []
        self.ego_point = (self.ego_state.rear_axle.x, self.ego_state.rear_axle.y)
        # NOTE 遍历路段中的所有车道边缘。
        # NOTE 计算每条边缘与自车位置的距离。
        for edge in starting_block.interior_edges:
            edges_distance.append(edge.polygon.distance(Point(self.ego_point)))
            # NOTE 如果距离小于 4 米，则将该边缘作为候选边缘。
            if edge.polygon.distance(Point(self.ego_point)) < 4:
                edges.append(edge)
        
        # if no edge is close to ego, use the closest edge
        if len(edges) == 0:
            edges.append(starting_block.interior_edges[np.argmin(edges_distance)])
        # 返回候选车道边缘。
        return edges
    # TODO 采样得到轨迹                 
    # NOTE routes是车道的车道中心线 有很多条，都是自车附近的车道的车道中心线
    def generate_paths(self, routes):
        ego_state = self.ego_state.rear_axle.x, self.ego_state.rear_axle.y, self.ego_state.rear_axle.heading
        
        # TODO generate paths
        new_paths = []
        path_distance = []
        # TODO sampled_index: 根据路径长度选择采样点的索引。
        for (path_len, dist, path_polyline) in routes:
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
                # NOTE 三阶贝塞尔曲线 path
                first_stage_path = calc_4points_bezier_path(ego_state[0], ego_state[1], ego_state[2], 
                                                            state[0], state[1], state[2], 3, sampled_index[j])[0]
                # NOTE 第一阶段路径：使用贝塞尔曲线生成从自车到目标点的路径。
                # NOTE 第二阶段路径：从目标点到路径终点的剩余部分。
                # NOTE path_polyline: 将第一阶段路径和第二阶段路径拼接成完整路径。
                # TODO 第一阶段lc or lk 第二阶段就是lk                                                   
                second_stage_path = path_polyline[sampled_index[j]+1:, :2]
                path_polyline = np.concatenate([first_stage_path, second_stage_path], axis=0)
                new_paths.append(path_polyline)  
                path_distance.append(dist)   

        # evaluate paths
        candiate_paths = {}
        for path, dist in zip(new_paths, path_distance):
            cost = self.calculate_cost(path, dist)
            candiate_paths[cost] = path

        # sort paths by cost
        candidate_paths = []
        # TODO 选出代价最小的前三个
        for cost in sorted(candiate_paths.keys())[:3]:
            path = candiate_paths[cost]
            # 样条插值 坐标转换 Ego frame
            # TODO 后处理
            path = self.post_process(path)
            candidate_paths.append(path)

        return candidate_paths
    
    def calculate_cost(self, path, dist):
        # NOTE 曲率：路径的最大曲率。
        # NOTE 车道变换：路径与自车的距离（表示车道变换代价）。
        # NOTE 障碍物：路径是否与障碍物相交。
        # NOTE 代价函数：综合考虑障碍物、车道变换和曲率。
        # path curvature
        curvature = self.calculate_path_curvature(path[0:100])
        curvature = np.max(curvature)

        # TODO lane change
        lane_change = dist

        # check obstacles
        # NOTE self.obstacles是周边环境中的静态障碍物
        obstacles = self.check_obstacles(path[0:100:10], self.obstacles)
        
        # final cost
        cost = 10 * obstacles + 1 * lane_change  + 0.1 * curvature

        return cost
    # TODO 经过后处理之后得到最终的参考path
    def post_process(self, path):
        # NOTE 转换到自车坐标系下
        path = self.transform_to_ego_frame(path)
        index = np.arange(0, len(path), 10)
        x = path[:, 0][index]
        y = path[:, 1][index]

        # spline interpolation
        rx, ry, ryaw, rk = calc_spline_course(x, y)
        spline_path = np.stack([rx, ry, ryaw, rk], axis=1)
        ref_path = spline_path[:self.max_path_len*10]

        return ref_path

    def depth_first_search(self, starting_edge, depth=0):
        if depth >= self.target_depth:
            return [[starting_edge]]
        else:
            traversed_edges = []
            child_edges = [edge for edge in starting_edge.outgoing_edges if edge.id in self.candidate_lane_edge_ids]

            if child_edges:
                for child in child_edges:
                    edge_len = len(child.baseline_path.discrete_path) * 0.25
                    traversed_edges.extend(self.depth_first_search(child, depth+edge_len))

            if len(traversed_edges) == 0:
                return [[starting_edge]]

            edges_to_return = []

            for edge_seq in traversed_edges:
                edges_to_return.append([starting_edge] + edge_seq)
                    
            return edges_to_return

    @staticmethod
    def calculate_path_curvature(path):
        dx = np.gradient(path[:, 0])
        dy = np.gradient(path[:, 1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        curvature = np.abs(dx * d2y - d2x * dy) / (dx**2 + dy**2)**(3/2)

        return curvature
    
    @staticmethod
    def check_obstacles(path, obstacles):
        expanded_path = LineString(path).buffer((WIDTH/2), cap_style=CAP_STYLE.square)

        for obstacle in obstacles:
            obstacle_polygon = obstacle.geometry
            if expanded_path.intersects(obstacle_polygon):
                return 1

        return 0

    def predict(self, encoder_outputs, traj_inputs, agent_states, timesteps):
        ego_trajs = torch.zeros((self.n_candidates_max, self.horizon*10, 6)).to(self.device)
        for i, traj in enumerate(traj_inputs):
            ego_trajs[i, :len(traj)] = traj[..., :6].float()

        ego_trajs = ego_trajs.unsqueeze(0)
        agent_trajs, scores, _, _ = self.decoder(encoder_outputs, ego_trajs, agent_states, timesteps)

        return agent_trajs, scores

    def transform_to_ego_frame(self, path):
        x = path[:, 0] - self.ego_state.rear_axle.x
        y = path[:, 1] - self.ego_state.rear_axle.y
        x_e = x * np.cos(-self.ego_state.rear_axle.heading) - y * np.sin(-self.ego_state.rear_axle.heading)
        y_e = x * np.sin(-self.ego_state.rear_axle.heading) + y * np.cos(-self.ego_state.rear_axle.heading)
        path = np.column_stack([x_e, y_e])

        return path
    # TODO 
    def plan(self, iteration, ego_state, env_inputs, starting_block, route_roadblocks, candidate_lane_edge_ids, traffic_light, observation, debug=False):
        # get environment information
        self.ego_state = ego_state
        self.candidate_lane_edge_ids = candidate_lane_edge_ids
        # NOTE 从地图中获取的路段
        self.route_roadblocks = route_roadblocks
        self.traffic_light = traffic_light
        object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.BARRIER,
                        TrackedObjectType.CZONE_SIGN, TrackedObjectType.TRAFFIC_CONE,
                        TrackedObjectType.GENERIC_OBJECT]
        objects = observation.tracked_objects.get_tracked_objects_of_types(object_types)
        self.obstacles = []
        # TODO
        for obj in objects:
            if obj.tracked_object_type == TrackedObjectType.VEHICLE:
                if obj.velocity.magnitude() < 0.1:
                    self.obstacles.append(obj.box)
            else:
                self.obstacles.append(obj.box)

        # initial tree (root node)
        # x, y, heading, velocity, acceleration, curvature, time
        state = torch.tensor([[0, 0, 0, # x, y, heading 
                               ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
                               ego_state.dynamic_car_state.rear_axle_acceleration_2d.x, 0, 0]], dtype=torch.float32)
        tree = TrajTree(state, None, 0)

        # environment encoding
        encoder_outputs = self.encoder(env_inputs)
        agent_states = env_inputs['neighbor_agents_past']

        # get candidate map lanes
        # TODO 车道的边界
        # TODO STEP1
        edges = self.get_candidate_edges(starting_block)
        # NOTE 自车前方的车道中心线
        candidate_paths = self.get_candidate_paths(edges)
        # NOTE 自车坐标系下，经过样条插值的path 平滑连接到车道中心线的path ds = 0.1
        # TODO 选出前三个cost最小的path
        paths = self.generate_paths(candidate_paths)
        self.speed_limit = edges[0].speed_limit_mps or self.target_speed
        
        # expand tree
        tree.expand_children(paths, self.first_stage_horizon, self.speed_limit, self.planner)
        # TODO 0~3秒的轨迹
        leaves = TrajTree.get_children(tree)

        # query the model
        parent_scores = {}
        # NOTE shape：列表 timesteps dim
        trajs = [leaf.total_traj[1:] for leaf in leaves]
        # trajs：提取轨迹树叶节点的轨迹。
        # self.predict：使用神经网络解码器预测自车和其他交通参与者的未来轨迹。
        # torch.topk(scores, self.n_candidates_expand)：选择得分最高的候选轨迹。
        agent_trajectories, scores = self.predict(encoder_outputs, trajs, agent_states, self.first_stage_horizon*10)
        # NOTE torch.topk返回两个张量：一个是包含选定元素的张量（即values），另一个是包含这些元素原来位置的索引张量（即indices）。
        # NOTE 最后，代码使用[0]来提取索引张量的第一个元素。这意味着即使self.n_candidates_expand指定了多个候选者的数量，我们也只关心其中排在最前面的那个候选者的索引。
        # TODO scores size:(1, 30)
        # NOTE indices 是一个一维张量，包含了前 self.n_candidates_expand 个分数对应的索引。
        indices = torch.topk(scores, self.n_candidates_expand)[1][0]
        pruned_leaves = []
        for i in indices:
            if i.item() < len(leaves):
                pruned_leaves.append(leaves[i])
                parent_scores[leaves[i]] = scores[0, i].item()

        # expand leaves with higher scores
        for leaf in pruned_leaves:
            leaf.expand_children(paths, self.horizon-self.first_stage_horizon, self.speed_limit, self.planner)

        # get all leaves
        # TODO 5~8s 对应的轨迹
        leaves = TrajTree.get_children(leaves)
        if len(leaves) > self.n_candidates_max:
           leaves = random.sample(leaves, self.n_candidates_max)

        # query the model      
        # 去掉0时刻的，其实就使用未来8秒的轨迹
        trajs = [leaf.total_traj[1:] for leaf in leaves]
        agent_trajectories, scores = self.predict(encoder_outputs, trajs, agent_states, self.horizon*10)
        
        # calculate scores
        # NOTE key：叶子节点的父节点 得分
        children_scores = {}
        for i, leaf in enumerate(leaves):
            if leaf.parent in children_scores:
                children_scores[leaf.parent].append(scores[0, i].item())
            else:
                children_scores[leaf.parent] = [scores[0, i].item()]

        # get the best parent
        best_parent = None
        best_child_index = None
        best_score = -np.inf
        for parent in parent_scores.keys():
            score = parent_scores[parent] + np.max(children_scores[parent])
            if score > best_score:
                best_parent = parent
                best_score = score
                best_child_index = np.argmax(children_scores[parent])

        # get the best trajectory
        # NOTE 从自车的轨迹树中，选择一个得分最高的轨迹作为自车的规划轨迹
        best_traj = best_parent.children[best_child_index].total_traj[1:, :3]
    
        # plot 
        if debug:
            for i, traj in enumerate(trajs):
                self.plot(iteration, env_inputs, traj, agent_trajectories[0, i])

        return best_traj
    
    def plot(self, iteration, env_inputs, ego_future, agents_future):
        fig = plt.gcf()
        dpi = 100
        size_inches = 800 / dpi
        fig.set_size_inches([size_inches, size_inches])
        fig.set_dpi(dpi)
        fig.set_tight_layout(True)

        # plot map
        map_lanes = env_inputs['map_lanes'][0]
        for i in range(map_lanes.shape[0]):
            lane = map_lanes[i].cpu().numpy()
            if lane[0, 0] != 0:
                plt.plot(lane[:, 0], lane[:, 1], color="gray", linewidth=20, zorder=1)
                plt.plot(lane[:, 0], lane[:, 1], "k--", linewidth=1, zorder=2)

        map_crosswalks = env_inputs['map_crosswalks'][0]
        for crosswalk in map_crosswalks:
            pts = crosswalk.cpu().numpy()
            plt.plot(pts[:, 0], pts[:, 1], 'b:', linewidth=2)

        # plot ego
        front_length = get_pacifica_parameters().front_length
        rear_length = get_pacifica_parameters().rear_length
        width = get_pacifica_parameters().width
        rect = plt.Rectangle((0 - rear_length, 0 - width/2), front_length + rear_length, width, 
                             linewidth=2, color='r', alpha=0.9, zorder=3)
        plt.gca().add_patch(rect)

        # plot agents
        agents = env_inputs['neighbor_agents_past'][0]
        for agent in agents:
            agent = agent[-1].cpu().numpy()
            if agent[0] != 0:
                rect = plt.Rectangle((agent[0] - agent[6]/2, agent[1] - agent[7]/2), agent[6], agent[7],
                                      linewidth=2, color='m', alpha=0.9, zorder=3,
                                      transform=mpl.transforms.Affine2D().rotate_around(*(agent[0], agent[1]), agent[2]) + plt.gca().transData)
                plt.gca().add_patch(rect)
                                    

        # plot ego and agents future trajectories
        ego = ego_future.cpu().numpy()
        agents = agents_future.cpu().numpy()
        plt.plot(ego[:, 0], ego[:, 1], color="r", linewidth=3)
        plt.gca().add_patch(plt.Circle((ego[29, 0], ego[29, 1]), 0.5, color="r", zorder=4))
        plt.gca().add_patch(plt.Circle((ego[79, 0], ego[79, 1]), 0.5, color="r", zorder=4))

        for agent in agents:
            if np.abs(agent[0, 0]) > 1:
                agent = trajectory_smoothing(agent)
                plt.plot(agent[:, 0], agent[:, 1], color="m", linewidth=3)
                plt.gca().add_patch(plt.Circle((agent[29, 0], agent[29, 1]), 0.5, color="m", zorder=4))
                plt.gca().add_patch(plt.Circle((agent[79, 0], agent[79, 1]), 0.5, color="m", zorder=4))

        # plot
        plt.gca().margins(0)  
        plt.gca().set_aspect('equal')
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axis([-50, 50, -50, 50])
        plt.show()
