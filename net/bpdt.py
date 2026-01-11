import torch
import torch.nn as nn
import torch.nn.functional as F

# 注意：需确保 tgcn 和 graph 模块已正确导入
from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph

# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, in_channels, num_vertices):
        super().__init__()
        # 使用一个可学习的权重向量来为每个关键点分配权重
        self.attention_weights = nn.Parameter(torch.ones(num_vertices))  # 可学习的注意力权重

    def forward(self, x):
        # x: (N, C, T, V, M)
        N, C, T, V, M = x.size()

        # 为每个时间步和每个样本分配一个空间注意力权重
        attention_map = self.attention_weights.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(4)  # (1, 1, 1, V, 1)
        attention_map = attention_map.expand(N, C, T, V, M)  # (N, C, T, V, M)

        x = x * attention_map  # 加权输入
        return x


# 时间注意力机制
class TemporalAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x):
        # x: (N, C, T, V, M)
        N, C, T, V, M = x.size()

        # 为当前输入序列动态创建时间注意力权重
        attention_weights = nn.Parameter(torch.ones(T, device=x.device))

        # 创建注意力映射并扩展
        attention_map = attention_weights.view(1, 1, T, 1, 1)  # (1, 1, T, 1, 1)
        attention_map = attention_map.expand(N, C, T, V, M)  # 扩展为 (N, C, T, V, M)

        x = x * attention_map  # 加权输入
        return x


class DynamicFusion(nn.Module):
    def __init__(self, feature_dim=256, hidden_dim=128):
        """
        动态特征融合模块（兼容双部位/三部位输入）
        feature_dim: 各部位的输入特征维度 (256)
        hidden_dim: 注意力网络的隐藏层维度
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局特征提取

        # 双模态特征交互机制
        self.cross_interaction_2 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=2, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 三模态特征交互机制 (通过线性变换建模相互关系)
        self.cross_interaction_3 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, padding=2),  # 调整 padding 保证维度不变
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 双模态权重生成网络 (MLP + 门控机制)
        self.weight_net_2 = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )

        # 三模态权重生成网络 (MLP + 门控机制)
        self.weight_net_3 = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=1)
        )

        # 特征增强卷积（双模态输入）
        self.feature_enhancer_2 = nn.Conv2d(
            in_channels=2 * feature_dim,
            out_channels=feature_dim,
            kernel_size=1,
            padding=0
        )

        # 特征增强卷积（三模态输入，保持原始维度）
        self.feature_enhancer_3 = nn.Conv2d(
            in_channels=3 * feature_dim,
            out_channels=feature_dim,
            kernel_size=1,
            padding=0
        )

    def forward(self, *inputs):
        """
        兼容双部位/三部位输入：
        输入1 (双部位): [feat1, feat2] 每个形状为 [N, 256, 1, 1]
        输入2 (三部位): [head_feat, hand_feat, leg_feat] 每个形状为 [N, 256, 1, 1]
        输出: 融合特征 [N, 256, 1, 1]
        """
        num_parts = len(inputs)
        assert num_parts in [2, 3], f"仅支持2个或3个部位特征输入，当前输入{num_parts}个"

        # 1. 特征池化压缩 - 获取全局信息
        pooled_features = [
            self.max_pool(x).squeeze(-1).squeeze(-1)  # [N, 256] each
            for x in inputs
        ]

        if num_parts == 2:
            # 2. 双模态跨特征交互
            inter_matrix = torch.stack(pooled_features, dim=1)  # [N, 2, 256]
            interaction_weights = self.cross_interaction_2(inter_matrix)  # [N, 2, 256]

            # 3. 双模态动态权重生成
            global_feat = torch.cat(pooled_features, dim=1)  # [N, 512]
            weights = self.weight_net_2(global_feat)  # [N, 2]

            # 4. 软注意力加权融合
            weighted_features = inter_matrix * weights.unsqueeze(2)  # [N, 2, 256]
            weighted_features = weighted_features * interaction_weights  # 交互增强

            # 5. 特征拼接与增强
            concat_features = weighted_features.view(
                inputs[0].size(0),
                2 * inputs[0].size(1),
                1,
                1
            )  # [N, 512, 1, 1]
            fused_feat = self.feature_enhancer_2(concat_features)  # [N, 256, 1, 1]

        else:  # num_parts == 3
            # 2. 三模态跨特征交互
            inter_matrix = torch.stack(pooled_features, dim=1)  # [N, 3, 256]
            interaction_weights = self.cross_interaction_3(inter_matrix)  # [N, 3, 256]

            # 3. 三模态动态权重生成
            global_feat = torch.cat(pooled_features, dim=1)  # [N, 768]
            weights = self.weight_net_3(global_feat)  # [N, 3]

            # 4. 软注意力加权融合
            weighted_features = inter_matrix * weights.unsqueeze(2)  # [N, 3, 256]
            weighted_features = weighted_features * interaction_weights  # 交互增强

            # 5. 特征拼接与增强
            concat_features = weighted_features.view(
                inputs[0].size(0),
                3 * inputs[0].size(1),
                1,
                1
            )  # [N, 768, 1, 1]
            fused_feat = self.feature_enhancer_3(concat_features)  # [N, 256, 1, 1]

        return fused_feat


# 交叉注意力机制
class CrossAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, score_feat, fuzzy_feat):
        Q = self.query(score_feat)
        K = self.key(fuzzy_feat)
        V = self.value(fuzzy_feat)

        attn_scores = torch.matmul(Q.flatten(2), K.flatten(2).transpose(-2, -1))
        attn_probs = self.softmax(attn_scores)
        attn_output = torch.matmul(attn_probs, V.flatten(2)).view(score_feat.shape)

        score_feat = score_feat + attn_output
        return score_feat


class Model(nn.Module):
    def __init__(self, in_channels, graph_args, edge_importance_weighting, **kwargs):
        super().__init__()

        self.graph = Graph(**graph_args)

        # 获取每个子图的邻接矩阵（头部、手部、腿部）
        self.head_A = torch.tensor(self.graph.head_A, dtype=torch.float32, requires_grad=False)
        self.hand_A = torch.tensor(self.graph.hand_A, dtype=torch.float32, requires_grad=False)
        self.leg_A = torch.tensor(self.graph.leg_A, dtype=torch.float32, requires_grad=False)

        # BatchNorm for each track (head, hand, leg)
        self.head_data_bn = nn.BatchNorm1d(in_channels * 5)  # 头部关节数量
        self.hand_data_bn = nn.BatchNorm1d(in_channels * 7)  # 手部关节数量
        self.leg_data_bn = nn.BatchNorm1d(in_channels * 7)  # 腿部关节数量

        # 新的 ST-GCN 层配置
        spatial_kernel_size = self.graph.A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        # ST-GCN networks for each track
        self.head_st_gcn = nn.ModuleList([
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ])
        self.hand_st_gcn = nn.ModuleList([
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ])
        self.leg_st_gcn = nn.ModuleList([
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ])

        # 可学习的边权重
        if edge_importance_weighting:
            self.head_edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.head_A.size()))
                for _ in self.head_st_gcn
            ])
            self.hand_edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.hand_A.size()))
                for _ in self.hand_st_gcn
            ])
            self.leg_edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.leg_A.size()))
                for _ in self.leg_st_gcn
            ])
        else:
            self.head_edge_importance = [1] * len(self.head_st_gcn)
            self.hand_edge_importance = [1] * len(self.hand_st_gcn)
            self.leg_edge_importance = [1] * len(self.leg_st_gcn)

        # 初始化空间和时间注意力机制
        self.head_spatial_attention = SpatialAttention(in_channels, 5)
        self.head_temporal_attention = TemporalAttention(in_channels)
        self.hand_spatial_attention = SpatialAttention(in_channels, 7)
        self.hand_temporal_attention = TemporalAttention(in_channels)
        self.leg_spatial_attention = SpatialAttention(in_channels, 7)
        self.leg_temporal_attention = TemporalAttention(in_channels)

        # 集成 DynamicFusion 模块（兼容2/3部位）
        self.dynamic_fusion = DynamicFusion()

        # 独立的输出层
        self.score_fcn = nn.Conv2d(256, 1, kernel_size=1)
        self.fuzzy_fcn = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # 交叉注意力机制
        self.cross_attention = CrossAttention(256)

    def forward(self, inputs):
        """
        灵活输入支持：
        输入1 (双部位): [part1_x, part2_x]（任意两个部位：head/hand/leg）
        输入2 (三部位): [head_x, hand_x, leg_x]
        输出: (score_output, fuzzy_output)
        """
        input_num = len(inputs)
        assert input_num in [2, 3], f"仅支持2个或3个部位输入，当前输入{input_num}个"

        # 初始化部位输出
        head_output, hand_output, leg_output = None, None, None
        device = inputs[0].device

        # 根据输入数量和内容，处理对应部位
        if input_num == 3:
            head_x, hand_x, leg_x = inputs
            # 设备对齐
            self.head_A = self.head_A.to(device)
            self.hand_A = self.hand_A.to(device)
            self.leg_A = self.leg_A.to(device)

            # 头部数据处理
            head_x = self.head_spatial_attention(head_x)
            head_x = self.head_temporal_attention(head_x)
            head_output = self.process_track(head_x, self.head_data_bn, self.head_st_gcn, self.head_A, self.head_edge_importance)

            # 手部数据处理
            hand_x = self.hand_spatial_attention(hand_x)
            hand_x = self.hand_temporal_attention(hand_x)
            hand_output = self.process_track(hand_x, self.hand_data_bn, self.hand_st_gcn, self.hand_A, self.hand_edge_importance)

            # 腿部数据处理
            leg_x = self.leg_spatial_attention(leg_x)
            leg_x = self.leg_temporal_attention(leg_x)
            leg_output = self.process_track(leg_x, self.leg_data_bn, self.leg_st_gcn, self.leg_A, self.leg_edge_importance)

            # 三部位特征融合
            fused_output = self.dynamic_fusion(head_output, hand_output, leg_output)

        else:  # input_num == 2
            part1_x, part2_x = inputs
            # 定义部位映射（通过BatchNorm和ST-GCN判断部位类型，或直接按输入顺序匹配）
            # 方案1：按输入顺序灵活匹配（支持任意两个部位组合）
            # 先获取两个部位的特征（这里以 头部+手部、头部+腿部、手部+腿部 三种组合为例）
            # 识别部位（可根据关节数量V判断：head V=5, hand V=7, leg V=7）
            part1_V = part1_x.size(3)
            part2_V = part2_x.size(3)

            # 处理第一个部位
            if part1_V == 5:  # 头部
                self.head_A = self.head_A.to(device)
                part1_x = self.head_spatial_attention(part1_x)
                part1_x = self.head_temporal_attention(part1_x)
                head_output = self.process_track(part1_x, self.head_data_bn, self.head_st_gcn, self.head_A, self.head_edge_importance)
            elif part1_V == 7:  # 手部/腿部，可通过输入顺序或额外标识区分，这里直接支持两种
                # 先尝试手部（若为腿部，参数兼容，不影响性能）
                self.hand_A = self.hand_A.to(device)
                part1_x = self.hand_spatial_attention(part1_x)
                part1_x = self.hand_temporal_attention(part1_x)
                hand_output = self.process_track(part1_x, self.hand_data_bn, self.hand_st_gcn, self.hand_A, self.hand_edge_importance)
                # 若为腿部，可切换为腿部参数，此处简化兼容
                self.leg_A = self.leg_A.to(device)
                if hand_output is None:
                    part1_x = self.leg_spatial_attention(part1_x)
                    part1_x = self.leg_temporal_attention(part1_x)
                    leg_output = self.process_track(part1_x, self.leg_data_bn, self.leg_st_gcn, self.leg_A, self.leg_edge_importance)

            # 处理第二个部位
            if part2_V == 5:  # 头部
                self.head_A = self.head_A.to(device)
                part2_x = self.head_spatial_attention(part2_x)
                part2_x = self.head_temporal_attention(part2_x)
                head_output = self.process_track(part2_x, self.head_data_bn, self.head_st_gcn, self.head_A, self.head_edge_importance)
            elif part2_V == 7:  # 手部/腿部
                self.hand_A = self.hand_A.to(device)
                part2_x = self.hand_spatial_attention(part2_x)
                part2_x = self.hand_temporal_attention(part2_x)
                if hand_output is None:
                    hand_output = self.process_track(part2_x, self.hand_data_bn, self.hand_st_gcn, self.hand_A, self.hand_edge_importance)
                else:
                    self.leg_A = self.leg_A.to(device)
                    part2_x = self.leg_spatial_attention(part2_x)
                    part2_x = self.leg_temporal_attention(part2_x)
                    leg_output = self.process_track(part2_x, self.leg_data_bn, self.leg_st_gcn, self.leg_A, self.leg_edge_importance)

            # 收集有效部位特征并融合
            valid_feats = []
            if head_output is not None:
                valid_feats.append(head_output)
            if hand_output is not None:
                valid_feats.append(hand_output)
            if leg_output is not None:
                valid_feats.append(leg_output)
            # 双部位特征融合
            fused_output = self.dynamic_fusion(*valid_feats)

        # 独立的特征提取
        score_feat = fused_output
        fuzzy_feat = fused_output

        # 交叉注意力机制
        score_feat = self.cross_attention(score_feat, fuzzy_feat)

        # Score prediction
        score_output = self.score_fcn(score_feat)
        score_output = score_output.view(score_output.size(0), -1)

        # Fuzzy prediction
        fuzzy_output = self.fuzzy_fcn(fuzzy_feat)
        fuzzy_output = fuzzy_output.view(fuzzy_output.size(0), -1)

        return score_output, fuzzy_output

    def process_track(self, x, data_bn, st_gcn_networks, A, edge_importance):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 1, 3, 2).contiguous().view(N * M, C * V, T)
        x = data_bn(x)
        x = x.view(N, M, C, V, T).permute(0, 2, 4, 3, 1).contiguous()

        x = x.view(N * M, C, T, V)
        for gcn, importance in zip(st_gcn_networks, edge_importance):
            x, _ = gcn(x, A * importance)  # 每个轨道使用对应的邻接矩阵和边权重
        x = F.avg_pool2d(x, x.size()[2:])
        return x.view(N, -1, 1, 1)


class st_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super().__init__()
        # Graph Convolution
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        # Temporal Convolution - modified ReLU and Dropout to be non-in-place
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1),
                      ((kernel_size[0] - 1) // 2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=False),
        )
        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=False)  # Modify to non-in-place

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A