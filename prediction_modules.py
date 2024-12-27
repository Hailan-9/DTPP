# 这段代码实现了自动驾驶任务中涉及的多个模块，包括位置编码（Positional Encoding）、Agent 编码器（AgentEncoder）、地图编码器（VectorMapEncoder）、
# 交叉注意力（CrossAttention）、Agent 解码器（AgentDecoder）和评分解码器（ScoreDecoder）。
# 它们共同组成了一个复杂的深度学习模型，用于生成轨迹预测、交互建模和评分（scoring）
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from common_utils import *
from debug_utils import *

# refer to:https://blog.csdn.net/m0_37605642/article/details/132866365
# 位置编码是 Transformer 模型的核心组件之一，用于在输入中引入时间或空间信息。
# TAG 引入顺序（位置） 信息
# 它继承自 nn.Module，这是所有PyTorch模块的基类。
class PositionalEncoding(nn.Module):
    # d_model（模型的维度,也就是token的维度，在nlp，就是最基本的单词对应的词向量维度！！！），dropout（dropout层的概率），max_len（序列的最大长度）。
    def __init__(self, d_model=256, dropout=0.1, max_len=100):
        # 调用父类 nn.Module 的构造函数，这是初始化模块时的标准做法。
        super(PositionalEncoding, self).__init__()
        # 创建一个从0到max_len-1的整数张量，代表每个位置的索引。
        # 使用 unsqueeze(1) 在第二个维度上增加一个维度，使其形状从 [max_len] 变为 [max_len, 1]，以便于后续的广播操作。！！！
        # 特征的长度！！！
        position = torch.arange(max_len).unsqueeze(1)
        # 创建一个张量，用于计算位置编码的缩放因子。这个缩放因子用于控制不同维度上正弦和余弦函数的频率。
        # 如果d_model是256，生成的序列将是[0, 2, 4, ..., 254]。
        # 这些索引对应于模型维度中的偶数位置，用于位置编码中的正弦函数。
        # math.log(10000.0)计算10000的自然对数，这是一个常数，用于控制缩放的幅度。
        # 这个值除以d_model，得到一个缩放因子，用于调整每个维度的频率。这个缩放因子确保了不同维度的正弦和余弦函数具有不同的频率，从而允许模型在不同尺度上捕捉信息。
        # 将上述生成的索引序列与缩放因子相乘，得到每个偶数维度的频率缩放值
        # torch.exp函数计算其参数的指数值。在这里，它用于计算每个偶数维度的频率缩放值的指数，生成一个包含这些指数值的张量。
        # 张量！！！
        # tensor shape is (d_model // 2)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # position encoder
        pe = torch.zeros(max_len, 1, d_model)
        # TAG 广播机制：在 PyTorch 中，广播机制允许在不同维度的张量之间进行算术运算。
        # TAG 具体来说，如果两个张量的维度不完全相同，但满足一定的条件（某个维度的大小为 1），那么较小维度的张量会被“广播”以匹配较大维度张量的形状。
        '''
        在 PyTorch 中，两个张量进行元素乘法时，如果它们的维度不完全相同，但满足一定的条件，那么较小维度的张量会被“广播”以匹配较大维度张量的形状。具体规则如下：
            TAG 从右向左比较两个张量的维度。
            如果某个维度的大小为 1，那么它会被广播以匹配另一个张量在该维度上的大小。
            如果某个维度的大小不为 1，那么两个张量在该维度上的大小必须相等。
        '''
        # TAG div_term shape is (d_model // 2)
        # TAG position shape is (max_len, 1) -->(广播为) (max_len, d_model // 2)
        # 结果是(max_len, d_model // 2)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # 调用 permute(1, 0, 2) 后，维度的顺序被重新排列为：
        # 1（原索引1）现在成为新的第一维，这通常对应于批次大小。
        # 0（原索引0）现在成为新的第二维，这对应于序列长度。
        # 2（原索引2）现在成为新的第三维，这对应于特征维度。
        pe = pe.permute(1, 0, 2)
        # 将位置编码 pe 注册为模型的常量，不会被优化。
        self.register_buffer('pe', pe)
        # 定义 Dropout，用于在前向传播中随机丢弃一部分神经元，防止过拟合。
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # positional encoding 和 imput embedding两者之间相加

        # shape 分别为(16, 40, 50, 256)  (1, 50, 256)
        # TAG 广播之后 第二个维度为(16, 40, 50, 256)
        # https://kimi.moonshot.cn/share/ctlc663mtof7p7jmd3o0
        x = x + self.pe
        
        return self.dropout(x)

# 一个继承自 nn.Module 的PyTorch模块，用于编码代理（agent）的运动信息。通常，这种编码器用于处理序列数据，如代理的历史位置或速度数据。
class AgentEncoder(nn.Module):
    def __init__(self, agent_dim):
        # 调用父类 nn.Module 的构造函数，这是初始化模块时的标准做法。
        super(AgentEncoder, self).__init__()
        # 定义了一个 LSTM（长短期记忆）层，用于处理输入序列。
        # NOTE agent_dim 是输入特征的维度。
        # NOTE 256 是 LSTM 层输出特征的维度，即每个时间步!!! 每个时间步的隐藏状态的大小。也就是隐藏层维度！
        # NOTE 2 表示 LSTM 层中包含两个独立的 LSTM 单元（或称为层）。堆叠lstm的层数
        # batch_first=True 指定输入和输出张量的第一个维度是批次大小（batch size），这符合 PyTorch 的标准实践！！！！！！！！！！！！！！！
        # NOTE batch_first： 如果是True，则input为(batch, seq, input_size)！！！！！！！！！！！！！！！！！！！！！！！
        
        # 参数依次为：input_size hidden_size num_layers
        self.motion = nn.LSTM(agent_dim, 256, 2, batch_first=True)
    # TAG 输入的最后一个维度是input_size(输入特征的维度) 输出的最后一个维度是hidden_units,前面的维度 输入和输出都是一样的!!!
    def forward(self, inputs):
        # https://blog.csdn.net/mary19831/article/details/129570030
        # https://blog.csdn.net/Cyril_KI/article/details/122557880
        # inputs:[batch_size， seq_length, input_size]
        # 调用其forward函数
        # 因为batch_first = True
        # 所以output(traj)的shape是b s h即batch_size seq_len hiddien_units\
        # 第二个返回值通常包含隐藏状态和细胞状态，但在这种情况下，我们只关心最终的输出，所以使用 _ 来忽略这些状态。
        traj, _ = self.motion(inputs)
        # 使用切片操作选择所有批次中最后一个时间步的输出。这里的 : 表示选择所有批次，-1 表示选择最后一个时间步。
        # TAG 在 LSTM 模型中，最后一个时间步的输出通常包含了整个序列的重要信息，因此在许多序列建模任务中，我们只需要关注这个输出。
        # TAG 返回最后一个时间步的输出，能够简化后续的处理，并保留序列的上下文信息。
        output = traj[:, -1]

        return output
    
# 对地图信息（如车道和人行横道）进行编码。
class VectorMapEncoder(nn.Module):
    # TAG 输入地图特征的维度和地图的最大长度（序列长度）--->也就是车道的点数（地图车道的点数）
    def __init__(self, map_dim, map_len):
        super(VectorMapEncoder, self).__init__()
        # 一个简单的前馈网络（MLP），用于对输入地图特征进行编码：
        self.point_net = nn.Sequential(nn.Linear(map_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256))
        # 使用位置编码器对地图特征添加位置信息。
        self.position_encode = PositionalEncoding(max_len=map_len)
    # 对地图特征进行分段，并生成掩码。
    def segment_map(self, map, map_encoding):
        # 批量大小 地图元素数量(相关车道的个数) 每个地图元素的点数 特征维度
        # TODO D 是特征维度，应该对应于 RGB channels!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        B, N_e, N_p, D = map_encoding.shape # tensor的形状
        # 对地图特征进行池化，将每 10 个点的特征聚合为一个！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ 池化
        # 对地图特征进行池化，将每 10 个点的特征聚合为一个！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ 池化
        # 使用 permute 调整张量维度以适应 F.max_pool2d 的输入要求。
        # permute方法用于重新排列张量的维度 将原来张量的维度0 3 1 2变成现在的维度0 1 2 3
        # kernel_size参数定义了池化窗口的大小。在这里，(1, 10)表示池化窗口在高度方向上为1，在宽度方向上为10。这意味着在每个通道内，最大池化操作会在宽度方向上覆盖10个像素，而在高度方向上覆盖1个像素。
        # TAG 这里的高度对应N_e 宽度对应N_p
        # 沿着宽度方向提取最大值，而高度方向保持不变。这通常用于提取序列数据中的时间特征
        # 最大池化（max pooling） 的窗口在滑动时，默认是按照 步幅（stride） 来移动的，而在代码中，stride 没有显式指定，因此会默认等于 kernel_size。
        map_encoding = F.max_pool2d(map_encoding.permute(0, 3, 1, 2), kernel_size=(1, 10))
        map_encoding = map_encoding.permute(0, 2, 3, 1).reshape(B, -1, D)

        # TAG 根据输入 map 生成掩码，用于标记无效的地图点。
        # torch.eq是PyTorch中的等价比较函数，它返回一个与输入张量形状相同的布尔张量，其中每个元素是True如果输入元素等于0，否则是False。
        # 选择所有批次、所有通道、所有高度和第一个宽度位置的元素。这将张量的维度从[batch_size, channels, height, width]减少到[batch_size, channels, height]。
        # reshape方法用于重新排列张量的维度。这里，张量被重新排列为四个维度：B（批次大小）、N_e（元素数量）、N_p//10和N_p//(N_p//10)。
        # N_p是原始宽度，N_p//10是宽度除以10，N_p//(N_p//10)是宽度除以宽度除以10的结果，这实际上是10
        # 第一个元素是x的位置 在自车坐标系下
        map_mask = torch.eq(map, 0)[:, :, :, 0].reshape(B, N_e, N_p//10, N_p//(N_p//10))
        print_tensor_shape_log(map_mask, "map_mask", DEBUG)
        # 它沿着指定的维度（dim=-1）找到最大值
        # [0]是最大值张量
        # TODO 这里是为了保障最后返回的两个张量除了最后维度的前面几个维度一样！！！！！！！！！！！！！！
        # max和前面的max_pool2d对应
        map_mask = torch.max(map_mask, dim=-1)[0].reshape(B, -1)

        return map_encoding, map_mask

    def forward(self, input):
        # 对输入地图特征进行编码,并加上位置信息
        # input shape is bt, max_elements, max_points, dim
        # TAG 引入位置信息: 使模型能够理解序列中元素的顺序。
        # TAG 增强自注意力机制: 通过提供位置信息，帮助模型更好地处理序列数据，尤其是在自然语言处理等任务中。
        # TAG max_points在这里就表示序列长度 seq_len。 这些车道位置点是有顺序的，所以需要位置编码
        output = self.position_encode(self.point_net(input))
        # 对地图特征进行分段，并生成掩码。
        encoding, mask = self.segment_map(input, output)

        return encoding, mask

# 实现交叉注意力机制，用于建模不同输入之间的交互。
class CrossAttention(nn.Module):
    # heads（注意力头的数量）多头注意力机制  dim（特征维度），dropout（dropout层的概率）。
    # dim：特征维度（即输入序列中每个时间步的特征向量维度）。
    # heads：注意力头的数量，将特征维度 dim 分成多个子空间，每个子空间独立计算注意力。
    # dropout：在注意力权重上应用的 dropout 概率，防止过拟合。
    def __init__(self, heads=8, dim=256, dropout=0.1):
        super(CrossAttention, self).__init__()
        # 多头注意力机制，用于计算 query 和 key、value 之间的相关性。
        # batch_first=True 表示输入和输出张量的第一个维度是批次大小。
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        # 层归一化（Layer Normalization）层，用于在注意力输出和前馈网络输出后进行归一化。这有助于稳定训练过程。
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        # feedforward neural network 也就是MLP
        # 前馈网络，用于对注意力输出进行进一步处理。
        # 对注意力输出进行进一步的非线性变换，增强特征表达能力。
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim))
        # 定义了一个dropout层，用于在前馈网络的输出后应用dropout，以减少过拟合。
        self.dropout = nn.Dropout(dropout)
    # TAG query（查询向量），key（键向量），value（值向量），以及可选的 mask（注意力掩码）。
    def forward(self, query, key, value, mask=None):
        # 将 query、key 和 value 传递给交叉注意力层，并计算注意力输出。如果提供了 mask，则在注意力计算中使用它来屏蔽某些位置!!!!!!!!!!!!
        # query：查询向量，形状为 (batch_size, seq_len_q, dim)。
        # key：键向量，形状为 (batch_size, seq_len_k, dim)。
        # value：值向量，形状为 (batch_size, seq_len_k, dim)。
        # mask：注意力掩码，形状为 (batch_size, seq_len_q, seq_len_k)，用于屏蔽某些位置（例如填充位置）。
        # mask.to(dtype=torch.bool, device=query.device)：确保掩码的类型和设备与输入一致。

        # attention_output：交叉注意力的输出，形状为 (batch_size, seq_len_q, dim)。
        # _：注意力权重（未使用）。
        # TAG query key value 在注意力模块内部会先线性化，然后再进行计算注意力
        attention_output, _ = self.cross_attention(query, key, value, attn_mask=mask.to(dtype=torch.float, device=query.device))
        check_tensor(attention_output, "attention_output", DEBUG_CHECK)
        check_tensor(query, "query", DEBUG_CHECK)
        check_tensor(key, "key", DEBUG_CHECK)
        check_tensor(value, "value", DEBUG_CHECK)
        check_tensor(attention_output, "attention_output", DEBUG_CHECK)


        attention_output = self.norm_1(attention_output)
        check_tensor(attention_output, "attention_output", DEBUG_CHECK)
        # 对注意力输出进行非线性变换。
        # 将注意力输出传入前馈网络，进一步提取特征。
        linear_output = self.ffn(attention_output)
        # 残差连接，防止梯度消失！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        output = attention_output + self.dropout(linear_output)
        # 层归一化
        output = self.norm_2(output)

        return output

# TAG 将编码后的agents的特征解码为轨迹预测结果。
class AgentDecoder(nn.Module):
    # 预测的时间步数（即轨迹的长度）。 每个时间步可能的轨迹分支数（多模态预测）。 输入特征的维度。
    def __init__(self, max_time, max_branch, dim):
        super(AgentDecoder, self).__init__()
        self._max_time = max_time
        self._max_branch = max_branch
        # 一个前馈神经网络，用于将编码后的特征解码为轨迹点：
        # 输出维度为 3*10，表示每秒的 10个timesteps对应的轨迹点，每个点包含 3 个特征（x, y, yaw）。
        self.traj_decoder = nn.Sequential(nn.Linear(dim, 128), nn.ELU(), nn.Linear(128, 3*10))

    def forward(self, encoding, current_state):
        # 将输入的编码特征 encoding 重塑为 [batch_size, max_branch, max_time, 512] 的形状。
        encoding = torch.reshape(encoding, (encoding.shape[0], self._max_branch, self._max_time, 512))
        # 将编码特征输入到轨迹解码器中，生成轨迹点。
        # TAG 对自车轨迹树的每个分支都预测了对应的agents的预测轨迹！！！！！！！！！！！！！！！！！
        # 将解码后的轨迹点重塑为 [batch_size, max_branch, max_time*10, 3] 的形状
        # max_time*10 表示每个时间步的 10 个轨迹点。3 表示每个点的特征（x, y, yaw）。
        agent_traj = self.traj_decoder(encoding).reshape(encoding.shape[0], self._max_branch, self._max_time*10, 3)
        # Tensor中利用None来增加维度，可以简单的理解为在None的位置上增加一维，新增维度大小为1，同时有几个None就会增加几个维度。
        # 将当前状态（位置和方向）加到预测的轨迹上，确保轨迹是基于当前状态的相对预测。
        # TAG 相当于预测出来的他车轨迹都是相对于当前位置的一个增量！！！！！
        agent_traj += current_state[:, None, None, :3]

        return agent_traj
# TODO ???????????????????????????????????????????????????????????????????????????????
# 用于对生成的轨迹进行评分，评估其质量和安全性。
# 有可学习的权重参数！！！！
class ScoreDecoder(nn.Module):
    # 是否使用可变的成本权重。如果为 True，权重会根据输入动态调整。
    def __init__(self, variable_cost=False):
        super(ScoreDecoder, self).__init__()
        self._n_latent_features = 4
        self._variable_cost = variable_cost
        # 一个前馈网络，用于对交互特征进行编码。
        self.interaction_feature_encoder = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 256))
        # 一个前馈网络，用于将编码后的交互特征解码为潜在交互特征。
        self.interaction_feature_decoder = nn.Sequential(nn.Linear(256, 64), nn.ELU(), nn.Linear(64, self._n_latent_features), nn.Sigmoid())
        # 一个前馈网络，用于生成评分的权重。
        self.weights_decoder = nn.Sequential(nn.Linear(256, 64), nn.ELU(), nn.Linear(64, self._n_latent_features+4), nn.Softplus())
    # 对一些特征 比如速度 加速度 jerk 横向加速度进行处理和归一化
    def get_hardcoded_features(self, ego_traj, max_time):
        # ego_traj: B, M, T, 6
        # x, y, yaw, v, a, r

        # shape is (B, M, max_time)
        speed = ego_traj[:, :, :max_time, 3]
        acceleration = ego_traj[:, :, :max_time, 4]
        # 时间间隔0.1s
        jerk = torch.diff(acceleration, dim=-1) / 0.1
        # torch.cat 是一个函数，用于连接两个或多个张量
        # TODO 使得尺寸对齐!!!
        jerk = torch.cat((jerk[:, :, :1], jerk), dim=-1)
        curvature = ego_traj[:, :, :max_time, 5]
        # 横向加速度公式 v * w
        lateral_acceleration = speed ** 2 * curvature
        # 0~15mps

        # 归一化操作!!!!
        # TODO 为啥速度取负值？ 有一点不理解
        # 首先计算 speed 张量在最后一个维度上的均值，并取负。
        # 然后将结果限制在 0 到 15 的范围内。
        # 最后将结果除以 15，进行归一化。
        # shape is (B, M)
        speed = -speed.mean(-1).clip(0, 15) / 15
        acceleration = acceleration.abs().mean(-1).clip(0, 4) / 4
        jerk = jerk.abs().mean(-1).clip(0, 6) / 6
        lateral_acceleration = lateral_acceleration.abs().mean(-1).clip(0, 5) / 5

        features = torch.stack((speed, acceleration, jerk, lateral_acceleration), dim=-1)

        return features
    
    def calculate_collision(self, ego_traj, agent_traj, agents_states, max_time):
        # ego_traj: B, T, 3
        # agent_traj: B, N, T, 3
        # agents_states: B, N, 11
        # 在深度学习中，掩码（mask）通常用于指示数据中的有效或相关部分。在这个上下文中，通过和零对比，我们可以创建一个掩码来区分 
        # agents_states 中非零（有效或活跃）的代理状态和零（无效或非活跃）的代理状态。这种掩码可以用于后续的计算，比如在计算损失函数时忽略不活跃的代理，或者在处理数据时只考虑有效的代理状态
        # 有效位置设置为true
        agent_mask = torch.ne(agents_states.sum(-1), 0) # B, N

        # Compute the distance between the two agents
        # dist： B N T
        dist = torch.norm(ego_traj[:, None, :max_time, :2] - agent_traj[:, :, :max_time, :2], dim=-1)
    
        # Compute the collision cost
        cost = torch.exp(-0.2 * dist ** 2) * agent_mask[:, :, None]
        # B N T --> B N --> B
        cost = cost.sum(-1).sum(-1)

        return cost
    # 输入的均是某个分支的数据 N是neighbor的意思
    def get_latent_interaction_features(self, ego_traj, agent_traj, agents_states, max_time):
        # ego_traj: B, T, 6
        # agent_traj: B, N, T, 3
        # agents_states: B, N, 11

        # Get agent mask
        # 得到他车(agent)掩码 不等于0的 也就是有效的为true 无效的为false
        agent_mask = torch.ne(agents_states.sum(-1), 0) # B, N

        # Get relative attributes of agents
        # shape is B N T
        relative_yaw = agent_traj[:, :, :max_time, 2] - ego_traj[:, None, :max_time, 2] # None 增加维度
        # torch.sin 和 torch.cos 的输入是弧度制 torch.atan2 的输出也是弧度
        # 由于 sin 和 cos 的值已经消除了多余的周期部分，atan2 会将角度归一化到 [-π, π] 的范围内。
        relative_yaw = torch.atan2(torch.sin(relative_yaw), torch.cos(relative_yaw))
        # bt N T 2
        relative_pos = agent_traj[:, :, :max_time, :2] - ego_traj[:, None, :max_time, :2]
        # 用于将给定的张量序列沿着一个新的维度合并。这个函数接受一个张量序列（可以是列表、元组或其他可迭代对象中的张量）和一个 dim 参数，指定沿着哪个维度进行合并。
        relative_pos = torch.stack([relative_pos[..., 0] * torch.cos(relative_yaw), 
                                    relative_pos[..., 1] * torch.sin(relative_yaw)], dim=-1)
        agent_velocity = torch.diff(agent_traj[:, :, :max_time, :2], dim=-2) / 0.1
        agent_velocity = torch.cat((agent_velocity[:, :, :1, :], agent_velocity), dim=-2)
        ego_velocity_x = ego_traj[:, :max_time, 3] * torch.cos(ego_traj[:, :max_time, 2])
        ego_velocity_y = ego_traj[:, :max_time, 3] * torch.sin(ego_traj[:, :max_time, 2])
        # shape is B N T 2
        relative_velocity = torch.stack([(agent_velocity[..., 0] - ego_velocity_x[:, None]) * torch.cos(relative_yaw),
                                         (agent_velocity[..., 1] - ego_velocity_y[:, None]) * torch.sin(relative_yaw)], dim=-1) 
        # bt neighbor T 5( = 2 + 1 + 2)
        relative_attributes = torch.cat((relative_pos, relative_yaw.unsqueeze(-1), relative_velocity), dim=-1)

        print_tensor_shape_log(relative_pos, "relative_pos", DEBUG)
        print_tensor_shape_log(relative_attributes, "relative_attributes", DEBUG)
        # Get agent attributes
        # .expand() 方法用于扩展张量的维度而不复制数据。它接受一系列整数参数，指定每个维度的新大小。
        # -1 表示保持当前维度的大小不变。在这个例子中，-1 用于保持批次大小和代理数量不变。
        # B N T 5
        agent_attributes = agents_states[:, :, None, 6:].expand(-1, -1, relative_attributes.shape[2], -1)
        # 他车的相对动态属性，位置 速度 yaw等 加上固有的属性
        # shape is B N T 10(5 + 5)
        attributes = torch.cat((relative_attributes, agent_attributes), dim=-1)
        attributes = attributes * agent_mask[:, :, None, None]
        print_tensor_shape_log(agent_mask, "agent_mask", DEBUG)
        print_tensor_shape_log(attributes, "attributes", DEBUG)
        # Encode relative attributes and decode to latent interaction features
        # bt neighbor T 10 = 5 + 5
        features = self.interaction_feature_encoder(attributes)
        # bt neighbor T 256 ----> bt T 256 ---> bt 256
        # TODO  这里的取最大 应该是池化操作。压缩特征
        features = features.max(1).values.mean(1)
        print_tensor_shape_log(features, "features", DEBUG)
        # bt 256 --- > bt 4
        features = self.interaction_feature_decoder(features)
        print_tensor_shape_log(features, "after features", DEBUG)
        return features
    # timesteps:30 or 80
    def forward(self, ego_traj, ego_encoding, agents_traj, agents_states, timesteps):
        # ego_traj：bt branch timesteps(80) 6
        # ego_encoding: bt 256
        # agents_traj：bt branch neighbor(10) timesteps(80) 3 
        # agents_states: bt neighbor(10) 11
        print_tensor_shape_log(agents_traj, "agents_traj", DEBUG)
        # bt branch 4（速度 加速度 jerk 横向加速度）
        ego_traj_features = self.get_hardcoded_features(ego_traj, timesteps)
        print_tensor_shape_log(ego_traj_features, "ego_traj_features", DEBUG)

        if not self._variable_cost:
            print_tensor_log(self._variable_cost, "self._variable_cost")
            ego_encoding = torch.ones_like(ego_encoding)
        # bt 8
        weights = self.weights_decoder(ego_encoding)
        # bt branch
        # 有效：true 无效：false
        ego_mask = torch.ne(ego_traj.sum(-1).sum(-1), 0)
        print_tensor_shape_log(ego_traj.sum(-1).sum(-1), "ego_traj.sum(-1).sum(-1)", DEBUG)
        print_tensor_shape_log(ego_mask, "ego_mask", DEBUG)


        scores = []
        # NOTE 循环---branch
        for i in range(agents_traj.shape[1]):
            print_tensor_log(i, "第 个分支的score计算",DEBUG)
            # bt 4
            hardcoded_features = ego_traj_features[:, i]
            # bt 4
            interaction_features = self.get_latent_interaction_features(ego_traj[:, i], agents_traj[:, i], agents_states, timesteps)
            # bt 8
            features = torch.cat((hardcoded_features, interaction_features), dim=-1)
            print_tensor_shape_log(features, "features", DEBUG)
            # * 是按位相乘 即对应位置的元素相乘
            # bt
            score = -torch.sum(features * weights, dim=-1)
            # bt
            collision_feature = self.calculate_collision(ego_traj[:, i], agents_traj[:, i], agents_states, timesteps)
            score += -10 * collision_feature
            print_tensor_shape_log(collision_feature, "collision_feature", DEBUG)
            print_tensor_shape_log(score, "score", DEBUG)

            scores.append(score)
        # bt branch 
        scores = torch.stack(scores, dim=1)
        print_tensor_shape_log(scores, "scores", DEBUG)
        # 使用 torch.where 函数将 ego_mask 为 False 的位置的得分设置为负无穷大，这通常用于在后续处理中忽略这些得分。
        # 无效的位置是较大的负值 有效的位置就是计算的score
        scores = torch.where(ego_mask, scores, torch.tensor(float('-1e9'), dtype=scores.dtype, device=scores.device))

        return scores, weights