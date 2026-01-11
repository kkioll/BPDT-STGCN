import numpy as np
import torch

class Graph():
    """ 用于建模从 OpenPose 提取的骨架数据的图结构

    参数:
        layout (str): 必须为以下之一：
            - openpose: 使用18个关节（详见 https://github.com/CMU-Perceptual-Computing-Lab/openpose#output）
            - ntu-rgb+d: 使用25个关节（详见 https://github.com/shahroudy/NTURGB-D）

        strategy (str): 选择邻接矩阵的构建策略
            - uniform: 均匀标签分配
            - distance: 基于距离的划分
            - spatial: 空间配置

        max_hop (int): 最大跳数（两个节点之间的最大距离）

        dilation (int): 控制核点之间的间隔
    """

    def __init__(self, layout='openpose', strategy='uniform', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout)  # 获取骨架的连接边
        self.hop_dis = self.get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)  # 根据策略构建邻接矩阵
        self.get_subgraph_adjacency()  # 获取头部、手部、腿部的邻接矩阵
        self.full_body_A = self.A  # 添加 full_body_A 属性
        self.full_body_num_vertices = self.num_node  # 添加 full_body_num_vertices 属性

        # 如果 CUDA 可用，则将 A 转移到 GPU
        if torch.cuda.is_available():
            self.A = self.A.cuda()
            self.full_body_A = self.full_body_A.cuda()

    def __str__(self):
        return str(self.A)

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18  # OpenPose 使用 18 个关节
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                (10, 9), (9, 8), (11, 1), (8, 1), (5, 1), (2, 1),
                (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)
            ]
            self.edge = self_link + neighbor_link
            self.center = 1
        else:
            raise ValueError("不支持的布局类型。")

    def get_adjacency(self, strategy):
        """ 获取邻接矩阵 """
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self.normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = torch.tensor(A, dtype=torch.float32)  # 确保 A 是一个张量
        else:
            raise ValueError("不支持的策略类型。")

    def get_subgraph_adjacency(self):
        """ 获取头部、手部和腿部的子图邻接矩阵 """
        head_joints = [16, 14, 0, 15, 17]  # 头部关节
        hand_joints = [4, 3, 2, 1, 5, 6, 7]  # 手部关节
        leg_joints = [10, 9, 8, 1, 11, 12, 13]  # 腿部关节

        # 获取每个子图的邻接矩阵
        self.head_A = self.get_sub_adjacency(head_joints)
        self.hand_A = self.get_sub_adjacency(hand_joints)
        self.leg_A = self.get_sub_adjacency(leg_joints)

    def get_sub_adjacency(self, joints):
        """ 根据给定的关节列表，构建子图的邻接矩阵 """
        sub_num_node = len(joints)
        sub_edge = []
        for i, j in self.edge:
            if i in joints and j in joints:
                sub_i = joints.index(i)
                sub_j = joints.index(j)
                sub_edge.append((sub_i, sub_j))
        sub_hop_dis = self.get_hop_distance(sub_num_node, sub_edge, max_hop=self.max_hop)
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        sub_adjacency = np.zeros((sub_num_node, sub_num_node))
        for hop in valid_hop:
            sub_adjacency[sub_hop_dis == hop] = 1
        sub_normalize_adjacency = self.normalize_digraph(sub_adjacency)
        sub_A = np.zeros((1, sub_num_node, sub_num_node))
        sub_A[0] = sub_normalize_adjacency
        return sub_A

    def get_hop_distance(self, num_node, edge, max_hop=1):
        """ 计算每个节点对之间的跳数距离 """
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1

        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def normalize_digraph(self, A):
        """ 归一化有向图的邻接矩阵 """
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD