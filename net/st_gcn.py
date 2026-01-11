import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph

class Model(nn.Module):
    def __init__(self, in_channels, graph_args, edge_importance_weighting, **kwargs):
        super().__init__()
        # 1. 构建骨架图 A
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        # print("邻接矩阵 A 的形状：", self.A.shape)   # [3,18,18]
        # print("A[0]（自环子集）：\n", self.A[0])
        # print("A[1]（第1类邻接子集）：\n", self.A[1])
        # print("A[2]（第2类邻接子集）：\n", self.A[2])
        # 2. BatchNorm1d 做 “通道×顶点” 维度上的归一化
        #    num_features = in_channels * num_vertices
        num_vertices = A.size(1)
        self.data_bn = nn.BatchNorm1d(in_channels * num_vertices)

        # 3. ST-GCN 层
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        kwargs0 = {k: v for k,v in kwargs.items() if k!='dropout'}
        self.st_gcn_networks = nn.ModuleList([
            st_gcn(in_channels, 64,  kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64,           64,  kernel_size, 1, **kwargs),
            st_gcn(64,           64,  kernel_size, 1, **kwargs),
            st_gcn(64,           64,  kernel_size, 1, **kwargs),
            st_gcn(64,          128,  kernel_size, 2, **kwargs),
            st_gcn(128,         128,  kernel_size, 1, **kwargs),
            st_gcn(128,         128,  kernel_size, 1, **kwargs),
            st_gcn(128,         256,  kernel_size, 2, **kwargs),
            st_gcn(256,         256,  kernel_size, 1, **kwargs),
            st_gcn(256,         256,  kernel_size, 1, **kwargs),
        ])

        # 4. 可学习的边权重
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # 5. 回归输出
        self.fcn = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        # x: (N, C, T, V, M)
        N, C, T, V, M = x.size()

        # —— 正确的 BatchNorm1d ——
        # 1) 把 (N, C, T, V, M) → (N*M, C, V, T)
        x = x.permute(0,4,1,3,2).contiguous()    # → (N, M, C, V, T)
        x = x.view(N*M, C*V, T)                  # → (N*M, C*V, T)
        # 2) 在 (C*V) 维度上做归一化
        x = self.data_bn(x)                      # BatchNorm1d(num_features=C*V)
        # 3) 恢复到 (N, C, T, V, M)
        x = x.view(N, M, C, V, T).permute(0,2,4,3,1).contiguous()  # → (N, C, T, V, M)
        # —— BatchNorm 完成 ——

        # 准备送入 ST-GCN：合并 N*M
        x = x.view(N*M, C, T, V)                 # → (N*M, C, T, V)

        # 图卷积 + 时序卷积
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # 全局池化
        x = F.avg_pool2d(x, x.size()[2:])        # → (N*M, 256, 1, 1)
        x = x.view(N, M, -1, 1, 1).mean(dim=1)   # → (N, 256, 1, 1)

        # 回归
        x = self.fcn(x)                         # → (N,1,1,1)
        x = x.view(N, -1)                       # → (N,)
        return x

class st_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super().__init__()
        # 图卷积
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        # 时序卷积
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0],1), (stride,1),
                      ((kernel_size[0]-1)//2,0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        # 残差
        if not residual:
            self.residual = lambda x: 0
        elif in_channels==out_channels and stride==1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride,1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A
