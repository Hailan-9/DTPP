import torch
from prediction_modules import *
from debug_utils import *
# 这段代码实现了一个自动驾驶系统中的 Encoder-Decoder 架构，用于生成其他车辆的轨迹预测信息，并对自车轨迹进行解码和评分


# Encoder 的作用是对输入的自车、周围车辆及地图信息进行编码，生成一个统一的特征表示，供后续的轨迹预测模块使用。
class Encoder(nn.Module):
    def __init__(self, dim=256, layers=3, heads=8, dropout=0.1):
        super(Encoder, self).__init__()
        # self._lane_len：每条车道的最大点数为 50。
        # self._lane_feature：每个车道点的特征维度为 7。
        # self._crosswalk_len：每个人行横道的最大点数为 30。
        # self._crosswalk_feature：每个人行横道点的特征维度为 3。
        self._lane_len = 50
        self._lane_feature = 7 # x y heading + one-shot
        self._crosswalk_len = 30
        self._crosswalk_feature = 3
        # AgentEncoder：
        # TODO 注意：是车辆的过去的状态 past trajectory
        # 用于编码自车和周围车辆的历史状态。
        # 自车特征维度为 7，其他车辆特征维度为 11。
        # 维度分别为x y heading vx vy yawrate length width 三位的one-shot
        self.agent_encoder = AgentEncoder(agent_dim=11)
        # 维度分别为 x y heading vx vy ax ay
        self.ego_encoder = AgentEncoder(agent_dim=7)
        # VectorMapEncoder：
        # 用于编码地图中的车道和人行横道特征。
        # 车道 人行横道
        self.lane_encoder = VectorMapEncoder(self._lane_feature, self._lane_len)
        self.crosswalk_encoder = VectorMapEncoder(self._crosswalk_feature, self._crosswalk_len)
        # Transformer 编码器 (fusion_encoder)：
        # 使用多头注意力机制对不同来源的特征（自车、车辆、地图等）进行融合。
        # 参数：
        # d_model=dim：特征维度为 256。
        # nhead=heads：注意力头数为 8。
        # dim_feedforward=dim*4：前馈网络的隐藏层维度为 1024。
        # layers=3：堆叠 3 层 Transformer 编码器。
        # layers 是一个整数，指定了编码器中堆叠的编码器层的数量。如果 layers=3，则意味着有三个 attention_layer 堆叠在一起。
        attention_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4,
                                                     activation="gelu", dropout=dropout, batch_first=True)
        # TODO 编码器部分 3X的encoder block组成了transformer的Encoder
        self.fusion_encoder = nn.TransformerEncoder(attention_layer, layers)
    # 处理自车和周围车辆的历史轨迹以及地图环境信息。
    def forward(self, inputs):
        # agents
        # ego：自车的历史状态，形状为 [batch_size, time_steps, 7]。
        # neighbors：其他车辆的历史状态，形状为 [batch_size, num_neighbors, time_steps, 11]。
        ego = inputs['ego_agent_past']
        neighbors = inputs['neighbor_agents_past']
        # actors：将自车和邻车的前 5 个特征拼接，形状为 [batch_size, 1+num_neighbors, time_steps, 5]。
        # x y heading vx vy
        # ... 代表之前所有的维度
        actors = torch.cat([ego[:, None, :, :5], neighbors[..., :5]], dim=1) # ego增加一个维度

        # agent encoding
        encoded_ego = self.ego_encoder(ego)
        # neighbors.shape[1]周围他车的数量
        encoded_neighbors = [self.agent_encoder(neighbors[:, i]) for i in range(neighbors.shape[1])]
        # 将 encoded_ego 作为一个列表的单个元素与 encoded_neighbors 列表相加，实际上是将它们连接起来形成一个更长的列表。
        encoded_actors = torch.stack([encoded_ego] + encoded_neighbors, dim=1)
        # batch nums_agent 5
        # 和零作对比，是因为无效的位置使用零填充，为了保证输入的尺寸是固定的。所以和0作对比，告知哪些位置是填充的，从而使得这些位置的注意力权重为0
        # TAG shape: (bt, neighbors + 1)
        # TAG 使用当前的状态来求解掩码
        actors_mask = torch.eq(actors[:, :, -1].sum(-1), 0)

        # vector maps
        map_lanes = inputs['map_lanes']
        map_crosswalks = inputs['map_crosswalks'] # 人行横道

        # map encoding
        # TAG shape：(bt, ne * n_p // 10, 256); (bt, ne * n_p // 10)
        encoded_map_lanes, lanes_mask = self.lane_encoder(map_lanes)
        encoded_map_crosswalks, crosswalks_mask = self.crosswalk_encoder(map_crosswalks)

        # attention fusion encoding
        # 输入是actor（包含自车和他车） 地图信息 人行横道的地图信息
        input = torch.cat([encoded_actors, encoded_map_lanes, encoded_map_crosswalks], dim=1)
        # 将多个张量列表cat在一起
        mask = torch.cat([actors_mask, lanes_mask, crosswalks_mask], dim=1)
        print_tensor_shape_log(actors_mask, "actors_mask", DEBUG)
        print_tensor_shape_log(lanes_mask, "lanes_mask", DEBUG)
        print_tensor_shape_log(crosswalks_mask, "crosswalks_mask", DEBUG)


        # (batch_size, sequence_length, feature_dim)，其中：
        #     batch_size 是批次中序列的数量。
        #     sequence_length 是每个序列的长度。
        #     feature_dim 是每个序列元素的特征维度。
        # NOTE src_key_padding_mask 是一个布尔型张量，用于告诉编码器哪些输入元素应该被视为填充（padding）元素，这些元素在计算注意力时将被忽略。 对应位置为True表示填充，也就是无效的。
        encoding = self.fusion_encoder(input, src_key_padding_mask=mask)

        # outputs
        encoder_outputs = {'encoding': encoding, 'mask': mask}

        return encoder_outputs

# Decoder 类的作用是利用 Encoder 的输出特征生成轨迹预测信息，包括其他车辆的预测轨迹、自车的轨迹正则化，以及轨迹评分。
class Decoder(nn.Module):
    def __init__(self, neighbors=10, max_time=8, max_branch=30, n_heads=8, dim=256, variable_cost=False):
        super(Decoder, self).__init__()
        self._neighbors = neighbors
        self._nheads = n_heads
        self._time = max_time
        self._branch = max_branch
        # 融合环境特征。
        self.environment_decoder = CrossAttention(n_heads, dim)
        # TAG 融合自车轨迹特征。自车条件预测!!!
        self.ego_condition_decoder = CrossAttention(n_heads, dim)
        # 用于为每个时间步创建时间特征。
        # max_time 个时间步，每个时间步映射到 dim 维特征  
        # forward输出的shape is （max_time, dim)
        self.time_embed = nn.Embedding(max_time, dim)
        # 自车6维度的候选轨迹
        self.ego_traj_encoder = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 256))
        # TAG agents 轨迹解码器，得到每一个agent的所有分支的轨迹
        self.agent_traj_decoder = AgentDecoder(max_time, max_branch, dim*2)
        # TAG 
        self.ego_traj_decoder = nn.Sequential(nn.Linear(256, 256), nn.ELU(), nn.Linear(256, max_time*10*3))
        self.scorer = ScoreDecoder(variable_cost)

        # self.time_index 是一个固定的时间步索引张量，形状为 [max_branch, max_time]。
        # 它通过时间嵌入层 (self.time_embed) 将时间步索引映射到特定的特征表示，用于轨迹预测中的时间特征建模。
        # 通过 register_buffer 注册，self.time_index 不会参与梯度计算，但在前向传播中会被使用。

        # TAG 通过注册这些缓冲区，模型可以保存和加载它们的状态，同时在训练过程中保持这些值不变。
        # register_buffer 是 PyTorch nn.Module 提供的一个方法，用于注册一个缓冲区。缓冲区类似于模型的参数，但它们不会在训练过程中被优化（即它们不会被梯度下降更新）。
        # TAG 注册因果掩码作为缓冲区，使得它能够被模型保存和加载，同时在训练过程中保持不变。
        self.register_buffer('casual_mask', self.generate_casual_mask())
        # TAG 注册时间索引作为缓冲区，这可能用于模型中的注意力机制或其他需要时间索引的操作。
        # TAG self.time_index 是一个张量，用于生成时间步的索引。它的作用是为每个轨迹分支（max_branch）
        # TAG 生成对应的时间步索引（max_time），以便后续通过时间嵌入层 (self.time_embed) 将时间步索引映射到特定的特征表示。
        # shape is  [max_branch, max_time] 
        self.register_buffer('time_index', torch.arange(max_time).repeat(max_branch, 1))
    # 对轨迹树进行池化，提取每个时间段的最大特征值。
    def pooling_trajectory(self, trajectory_tree):
        # batch_size multi_branch time feature_dim
        B, M, T, D = trajectory_tree.shape
        # 将时间步划分为 10 个一组，形状变为 [B, M, T//10, 10, D]。
        trajectory_tree = torch.reshape(trajectory_tree, (B, M, T//10, 10, D))
        # 维度变成了(B, M, T//10, D)
        # 得到了一个元组，其中第一个元素是最大值，第二个元素是这些最大值的索引。
        trajectory_tree = torch.max(trajectory_tree, dim=-2)[0]

        return trajectory_tree
    # 因果掩码 (casual_mask)：
    # 确保时间步之间的因果关系，即当前时间步只能使用之前时间步的信息。
    # 和论文第四页的图片一样
    def generate_casual_mask(self):
        # 创建一个下三角矩阵，形状为 [max_time, max_time]。
        time_mask = torch.tril(torch.ones(self._time, self._time))
        casual_mask = torch.zeros(self._branch * self._time, self._branch * self._time)
        # 将时间掩码复制到每个轨迹分支中，形状为 [max_branch * max_time, max_branch * max_time]。
        for i in range(self._branch):
            # 一个子矩阵
            casual_mask[i*self._time:(i+1)*self._time, i*self._time:(i+1)*self._time] = time_mask

        return casual_mask
    # TAG 输入是编码器的输出 自车轨迹的输入 他车的状态(其实是past + current的状态) 轨迹总的步数（时间 / 0.1s）
    # TAG ego_traj_inputs:batchsize branch time(80) dim 自车的轨迹树
    def forward(self, encoder_outputs, ego_traj_inputs, agents_states, timesteps):
        print_tensor_shape_log(ego_traj_inputs, "ego_traj_inputs", DEBUG)
        print_tensor_log(timesteps, "timesteps", DEBUG)


        # TAG get inputs (16 10 11) 只取前neighbors个
        current_states = agents_states[:, :self._neighbors, -1]
        # TAG encoding：(bt, seq_total, 256) encoding_mask：(bt, seq_total)
        encoding, encoding_mask = encoder_outputs['encoding'], encoder_outputs['mask']
        # shape is (bt, max_branch, timesteps, dim)
        # TAG 无论是第一阶段还是第二阶段 为了保持输入的尺寸一样，所以timesteps都是80 
        ego_traj_ori_encoding = self.ego_traj_encoder(ego_traj_inputs)
        # TODO 选择节点，所以是每段轨迹的最后一个时刻的轨迹点！！！！
        # shape is (bt, max_branch, dim) 即（16， 30， 256）
        branch_embedding = ego_traj_ori_encoding[:, :, timesteps-1]
        print_tensor_shape_log(branch_embedding, "branch_embedding", DEBUG)
        # TAG batch_size branch time // 10 dim 池化操作！！！！！
        ego_traj_ori_encoding = self.pooling_trajectory(ego_traj_ori_encoding)
        # shape is (max_branch, max_time, dim(256))
        time_embedding = self.time_embed(self.time_index)
        # 增加batch_size维度
        # TAG (1(None), 30, 8, 256) + (bt, 30, 1(None), 256)
        # boardcast 
        # TAG 自车轨迹树的嵌入信息
        tree_embedding = time_embedding[None, :, :, :] + branch_embedding[:, :, None, :]

        print_tensor_shape_log(time_embedding, "time_embedding", DEBUG)
        print_tensor_shape_log(time_embedding[None, :, :, :], "time_embedding[None, :, :, :]", DEBUG)
        print_tensor_shape_log(branch_embedding[:, :, None, :], "branch_embedding[:, :, None, :]", DEBUG)
        print_tensor_shape_log(tree_embedding, "tree_embedding", DEBUG)

        # get mask
        # 被0填充的轨迹 是满足输入尺寸 不需要关注 所以使用了掩码机制
        # TODO 填充位置是false 有效位置是true
        # bt max_branch timesteps(80)
        ego_traj_mask = torch.ne(ego_traj_inputs.sum(-1), 0)
        print_tensor_shape_log(ego_traj_mask, "ego_traj_mask", DEBUG)

        # 以计算出的步长进行切片
        # bt max_branch  _time
        ego_traj_mask = ego_traj_mask[:, :, ::(ego_traj_mask.shape[-1]//self._time)]
        print_tensor_shape_log(ego_traj_mask, "ego_traj_mask", DEBUG)
        # bt max_branch * _time
        ego_traj_mask = torch.reshape(ego_traj_mask, (ego_traj_mask.shape[0], -1))
        # TAG encoding_mask中 无效的是true 有效的是false 需要下面的代码需要逻辑取反
        # 使用 torch.einsum 计算 ego_traj_mask 和 encoding_mask 的逻辑非的外积，生成环境掩码。
        # .logical_not() 是 PyTorch 的逻辑取反操作，将布尔值 True 转为 False，False 转为 True。
        # 我们使用 torch.einsum 的公式 'ij,ik->ijk' 来计算外积。这个公式的意思是，对于 ego_traj_mask 的每个元素 ego_traj_mask[i, j] 和 encoding_mask.logical_not() 的每个元素 encoding_mask.
        # logical_not()[i, k]，我们计算它们的乘积，并将结果放在新张量的第 i 个样本的第 j 个时间步和第 k 个编码元素的位置上。
        # TODO 在计算 env_mask 时，我们需要构造一个掩码，表示 ego 轨迹的填充位置与环境编码的有效位置之间的关系。
        # TAG 如果 ego_traj_mask[i, j] 是 True（有效位置），并且 encoding_mask[i, k].logical_not 是 True（有效位置），那么 env_mask[i, j, k] 应该是 True，表示这个位置不需要屏蔽 是有效的 否则都是无效的！！
        # TAG else，那么 env_mask[i, j, k] 应该是 False，表示这个位置是无效的，需要屏蔽

        # NOTE shape is (bt, max_branch * _time) , (bt, seq_total) -----> (bt, max_branch * _time, seq_total) 
        env_mask = torch.einsum('ij,ik->ijk', ego_traj_mask, encoding_mask.logical_not())
        print_tensor_shape_log(ego_traj_mask, "ego_traj_mask", DEBUG)
        print_tensor_shape_log(encoding_mask, "encoding_mask", DEBUG)
        print_tensor_shape_log(env_mask, "env_mask", DEBUG)



        # TAG 将 env_mask 中值为1的位置设置为0，其他位置设置为一个大的负数（用于注意力机制中的掩码），使得这些位置的注意力权重接近0，起到屏蔽的作用！！！！！！！！！！！！！！！！！！！！
        env_mask = torch.where(env_mask == 1, torch.tensor(0, dtype=torch.long, device=env_mask.device), torch.tensor(-1e9, dtype=torch.long, device=env_mask.device))
        # 使用布尔掩码设置值  
        # shape:(nheads* bt, max_branch * _time, seq_total) 
        env_mask = env_mask.repeat(self._nheads, 1, 1)
        # 生成 ego_condition_mask，结合了因果掩码和 ego_traj_mask。
        # @是矩阵相乘 *是对应元素相乘 也就是按位相乘
        # TODO casual_mask中 1表示不遮挡 0 表示遮挡
        # (max_branch * time, max_branch * time) 即30 * 8， 30 * 8
        print_tensor_shape_log(self.casual_mask, "self.casual_mask", DEBUG)
        print_tensor_shape_log(ego_traj_mask, "ego_traj_mask", DEBUG)
        # (1, 240, 240) * (16, 240, 1) -----------> (16, 240, 240)
        ego_condition_mask = self.casual_mask[None, :, :] * ego_traj_mask[:, :, None]
        # 调整 ego_condition_mask 的值，并重复以匹配多头注意力的头数。
        # torch.where 的三个参数（condition、x 和 y）所在的设备（device）需要一致
        ego_condition_mask = torch.where(ego_condition_mask == 1, torch.tensor(0, dtype=torch.long, device=ego_condition_mask.device), torch.tensor(-1e9, dtype=torch.long, device=ego_condition_mask.device))
        # 使用布尔掩码设置值  
        # TAG (16, 240, 240) repeat(self._nheads, 1, 1)之后变为 (16 * nhead, 240, 240)
        ego_condition_mask = ego_condition_mask.repeat(self._nheads, 1, 1)

        print_tensor_shape_log(ego_condition_mask, "ego_condition_mask", DEBUG)
        # decode
        # 初始化一个空列表，用于存储解码后的代理轨迹。
        agents_trajecotries = []
        for i in range(self._neighbors):
            # learnable query 可学习的query
            # 第一个是ego
            # tree_embedding：bt branch times(8) 256
            # NOTE shape is (16, 1, 1, 256)   (16, 30, 8, 256) 故query的shape is (16, 30, 8, 256)!!!!!!!!!!!!!!!!!!!
            query = encoding[:, i+1, None, None] + tree_embedding
            # 16 30 * 8 256
            query = torch.reshape(query, (query.shape[0], -1, query.shape[-1]))
            print_tensor_shape_log(encoding, "encoding", DEBUG)
            print_tensor_shape_log(encoding[:, i+1, None, None], "encoding[:, i+1, None, None]", DEBUG)
            print_tensor_shape_log(tree_embedding, "tree_embedding", DEBUG)
            print_tensor_shape_log(query, "query", DEBUG)

      
            # decode from environment inputs
            # q k v
            # TAG transformer中 decoder block中的第二个注意力模块的 k v 来自于编码器输出的编码信息矩阵
            # TAG query shape: (16, 30 * 8, 256) --- encoding shape:(bt(16), seq_total(236), 256)
            # TAG env_mask：shape:(nheads* bt, max_branch * _time, seq_total) 
            env_decoding = self.environment_decoder(query, encoding, encoding, env_mask)

            # TAG 输入注意力模块的query key value会先经过线性层处理！！！！！！！！！！！！！！！！！！！！！！！！！！！！

            # decode from ego trajectory inputs
            # shape: (bt, branch, timesteps // 10, dim(256)) ---> (bt, 30 * 8, 256)
            ego_traj_encoding = torch.reshape(ego_traj_ori_encoding, (ego_traj_ori_encoding.shape[0], -1, ego_traj_ori_encoding.shape[-1]))
            ego_condition_decoding = self.ego_condition_decoder(query, ego_traj_encoding, ego_traj_encoding, ego_condition_mask)

            check_tensor(query, "query", DEBUG_CHECK)
            check_tensor(encoding, "encoding", DEBUG_CHECK)
            check_tensor(ego_traj_encoding, "ego_traj_encoding", DEBUG_CHECK)
            check_tensor(env_mask, "env_mask", DEBUG_CHECK)
            check_tensor(ego_condition_mask, "ego_condition_mask", DEBUG_CHECK)

            # trajectory outputs
            # （16， 30*8， 256 + 256）
            decoding = torch.cat([env_decoding, ego_condition_decoding], dim=-1)

            # (bt, max_branch, timesteps(80), 3)
            trajectory = self.agent_traj_decoder(decoding, current_states[:, i])
            print_tensor_shape_log(trajectory, "trajectory", DEBUG)
            # TAG 附近agent的轨迹
            agents_trajecotries.append(trajectory)

        # score outputs
        # TAG agents_trajecotries：bt branch neighbor times（80） 3
        agents_trajecotries = torch.stack(agents_trajecotries, dim=2)
        print_tensor_shape_log(agents_trajecotries, "agents_trajecotries", DEBUG)
        # scores：(bt, branch) weights:(bt, 8)
        scores, weights = self.scorer(ego_traj_inputs, encoding[:, 0], agents_trajecotries, current_states, timesteps)

        # ego regularization
        ego_traj_regularization = self.ego_traj_decoder(encoding[:, 0])
        ego_traj_regularization = torch.reshape(ego_traj_regularization, (ego_traj_regularization.shape[0], 80, 3))

        return agents_trajecotries, scores, ego_traj_regularization, weights