import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, num_vertices):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.ones(num_vertices)) 

    def forward(self, x):
        # x: (N, C, T, V, M)
        N, C, T, V, M = x.size()

        attention_map = self.attention_weights.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(4)  # (1, 1, 1, V, 1)
        attention_map = attention_map.expand(N, C, T, V, M)  # (N, C, T, V, M)

        x = x * attention_map  
        return x


class TemporalAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x):
        # x: (N, C, T, V, M)
        N, C, T, V, M = x.size()

        attention_weights = nn.Parameter(torch.ones(T, device=x.device))

        attention_map = attention_weights.view(1, 1, T, 1, 1)  # (1, 1, T, 1, 1)
        attention_map = attention_map.expand(N, C, T, V, M)  

        x = x * attention_map  
        return x


class DynamicFusion(nn.Module):
    def __init__(self, feature_dim=256, hidden_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_pool = nn.AdaptiveMaxPool2d(1)  

        self.cross_interaction_2 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=2, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.cross_interaction_3 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, padding=2),  
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.weight_net_2 = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )

        self.weight_net_3 = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=1)
        )

        self.feature_enhancer_2 = nn.Conv2d(
            in_channels=2 * feature_dim,
            out_channels=feature_dim,
            kernel_size=1,
            padding=0
        )

        self.feature_enhancer_3 = nn.Conv2d(
            in_channels=3 * feature_dim,
            out_channels=feature_dim,
            kernel_size=1,
            padding=0
        )

    def forward(self, *inputs):
        num_parts = len(inputs)
        assert num_parts in [2, 3], f"{num_parts}"

        pooled_features = [
            self.max_pool(x).squeeze(-1).squeeze(-1)  # [N, 256] each
            for x in inputs
        ]

        if num_parts == 2:
            inter_matrix = torch.stack(pooled_features, dim=1)  # [N, 2, 256]
            interaction_weights = self.cross_interaction_2(inter_matrix)  # [N, 2, 256]

            global_feat = torch.cat(pooled_features, dim=1)  # [N, 512]
            weights = self.weight_net_2(global_feat)  # [N, 2]

            weighted_features = inter_matrix * weights.unsqueeze(2)  # [N, 2, 256]
            weighted_features = weighted_features * interaction_weights  

            concat_features = weighted_features.view(
                inputs[0].size(0),
                2 * inputs[0].size(1),
                1,
                1
            )  # [N, 512, 1, 1]
            fused_feat = self.feature_enhancer_2(concat_features)  # [N, 256, 1, 1]

        else:  # num_parts == 3
            inter_matrix = torch.stack(pooled_features, dim=1)  # [N, 3, 256]
            interaction_weights = self.cross_interaction_3(inter_matrix)  # [N, 3, 256]

            global_feat = torch.cat(pooled_features, dim=1)  # [N, 768]
            weights = self.weight_net_3(global_feat)  # [N, 3]

            weighted_features = inter_matrix * weights.unsqueeze(2)  # [N, 3, 256]
            weighted_features = weighted_features * interaction_weights 
            concat_features = weighted_features.view(
                inputs[0].size(0),
                3 * inputs[0].size(1),
                1,
                1
            )  # [N, 768, 1, 1]
            fused_feat = self.feature_enhancer_3(concat_features)  # [N, 256, 1, 1]

        return fused_feat


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

        self.head_A = torch.tensor(self.graph.head_A, dtype=torch.float32, requires_grad=False)
        self.hand_A = torch.tensor(self.graph.hand_A, dtype=torch.float32, requires_grad=False)
        self.leg_A = torch.tensor(self.graph.leg_A, dtype=torch.float32, requires_grad=False)

        # BatchNorm for each track (head, hand, leg)
        self.head_data_bn = nn.BatchNorm1d(in_channels * 5)  
        self.hand_data_bn = nn.BatchNorm1d(in_channels * 7)  
        self.leg_data_bn = nn.BatchNorm1d(in_channels * 7)  

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

        self.head_spatial_attention = SpatialAttention(in_channels, 5)
        self.head_temporal_attention = TemporalAttention(in_channels)
        self.hand_spatial_attention = SpatialAttention(in_channels, 7)
        self.hand_temporal_attention = TemporalAttention(in_channels)
        self.leg_spatial_attention = SpatialAttention(in_channels, 7)
        self.leg_temporal_attention = TemporalAttention(in_channels)

        self.dynamic_fusion = DynamicFusion()

        self.score_fcn = nn.Conv2d(256, 1, kernel_size=1)
        self.fuzzy_fcn = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.cross_attention = CrossAttention(256)

    def forward(self, inputs):
        input_num = len(inputs)
        assert input_num in [2, 3], f"仅支持2个或3个部位输入，当前输入{input_num}个"

        head_output, hand_output, leg_output = None, None, None
        device = inputs[0].device

        if input_num == 3:
            head_x, hand_x, leg_x = inputs
            self.head_A = self.head_A.to(device)
            self.hand_A = self.hand_A.to(device)
            self.leg_A = self.leg_A.to(device)

            head_x = self.head_spatial_attention(head_x)
            head_x = self.head_temporal_attention(head_x)
            head_output = self.process_track(head_x, self.head_data_bn, self.head_st_gcn, self.head_A, self.head_edge_importance)

            hand_x = self.hand_spatial_attention(hand_x)
            hand_x = self.hand_temporal_attention(hand_x)
            hand_output = self.process_track(hand_x, self.hand_data_bn, self.hand_st_gcn, self.hand_A, self.hand_edge_importance)

            leg_x = self.leg_spatial_attention(leg_x)
            leg_x = self.leg_temporal_attention(leg_x)
            leg_output = self.process_track(leg_x, self.leg_data_bn, self.leg_st_gcn, self.leg_A, self.leg_edge_importance)

            fused_output = self.dynamic_fusion(head_output, hand_output, leg_output)

        else:  # input_num == 2
            part1_x, part2_x = inputs
            
            part1_V = part1_x.size(3)
            part2_V = part2_x.size(3)

            if part1_V == 5: 
                self.head_A = self.head_A.to(device)
                part1_x = self.head_spatial_attention(part1_x)
                part1_x = self.head_temporal_attention(part1_x)
                head_output = self.process_track(part1_x, self.head_data_bn, self.head_st_gcn, self.head_A, self.head_edge_importance)
            elif part1_V == 7: 
                self.hand_A = self.hand_A.to(device)
                part1_x = self.hand_spatial_attention(part1_x)
                part1_x = self.hand_temporal_attention(part1_x)
                hand_output = self.process_track(part1_x, self.hand_data_bn, self.hand_st_gcn, self.hand_A, self.hand_edge_importance)
                self.leg_A = self.leg_A.to(device)
                if hand_output is None:
                    part1_x = self.leg_spatial_attention(part1_x)
                    part1_x = self.leg_temporal_attention(part1_x)
                    leg_output = self.process_track(part1_x, self.leg_data_bn, self.leg_st_gcn, self.leg_A, self.leg_edge_importance)

            if part2_V == 5:  
                self.head_A = self.head_A.to(device)
                part2_x = self.head_spatial_attention(part2_x)
                part2_x = self.head_temporal_attention(part2_x)
                head_output = self.process_track(part2_x, self.head_data_bn, self.head_st_gcn, self.head_A, self.head_edge_importance)
            elif part2_V == 7:  
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

            valid_feats = []
            if head_output is not None:
                valid_feats.append(head_output)
            if hand_output is not None:
                valid_feats.append(hand_output)
            if leg_output is not None:
                valid_feats.append(leg_output)
            fused_output = self.dynamic_fusion(*valid_feats)

        score_feat = fused_output
        fuzzy_feat = fused_output

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
            x, _ = gcn(x, A * importance) 
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

