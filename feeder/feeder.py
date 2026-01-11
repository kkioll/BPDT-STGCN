import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from feeder.tools import  SkeletonAugmentor

class Feeder(Dataset):
    def __init__(self, data_dir, label_path, random_choose=False, random_move=False,
                 window_size=-1, debug=False, augment=False, augment_train_only=False):
        self.data_dir = data_dir
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.debug = debug
        self.augment = augment
        self.augment_train_only = augment_train_only

        # Load the labels
        with open(self.label_path, 'rb') as f:
            self.labels = pickle.load(f)

        # List the npy files in the data directory
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.sample_names = [os.path.splitext(f)[0] for f in self.file_list]

        if debug:
            self.file_list = self.file_list[:100]
            self.sample_names = self.sample_names[:100]

    def __len__(self):
        return len(self.file_list)

    def normalize_skeleton(self, data):
        """
        归一化骨架：以脖子（索引1）为中心点，平移整个骨架
        data: (3, T, 18) 的骨架数据
        """
        # 获取脖子关节（索引1）的坐标
        neck_x = data[0, :, 1]  # x坐标 (T,)
        neck_y = data[1, :, 1]  # y坐标 (T,)

        # 将每个关节的坐标减去脖子关节的坐标
        # 对于x坐标（通道0）
        data[0, :, :] = data[0, :, :] - neck_x[:, np.newaxis]
        # 对于y坐标（通道1）
        data[1, :, :] = data[1, :, :] - neck_y[:, np.newaxis]

        return data

    def __getitem__(self, index):
        # Load the npy data file
        file_path = os.path.join(self.data_dir, self.file_list[index])
        data = np.load(file_path)  # Shape (T, 18, 3)

        # Convert to (C, T, V) -> (3, T, 18)
        data = np.transpose(data, (2, 0, 1))  # Shape (3, T, 18)

        # 归一化骨架
        data = self.normalize_skeleton(data)

        # 数据增强
        if self.augment and (not self.augment_train_only or 'train' in self.data_dir.lower()):
            data = SkeletonAugmentor.augment(data)


        # Define joint indices for each body part
        head_joints = [16, 14, 0, 15, 17]  # Head joints
        hand_joints = [4, 3, 2, 1, 5, 6, 7]  # Hand joints
        leg_joints = [10, 9, 8, 1, 11, 12, 13]  # Leg joints

        # Split data according to the joint indices
        head_data = data[:, :, head_joints]
        hand_data = data[:, :, hand_joints]
        leg_data = data[:, :, leg_joints]
        full_body_data = data  # 全身数据

        # Add batch dimension (N=1) and M=1 dimension
        head_data = np.expand_dims(head_data, axis=0)  # Shape (N=1, C, T, V_head)
        head_data = np.expand_dims(head_data, axis=-1)  # Shape (N=1, C, T, V_head, M=1)
        hand_data = np.expand_dims(hand_data, axis=0)  # Shape (N=1, C, T, V_hand)
        hand_data = np.expand_dims(hand_data, axis=-1)  # Shape (N=1, C, T, V_hand, M=1)
        leg_data = np.expand_dims(leg_data, axis=0)  # Shape (N=1, C, T, V_leg)
        leg_data = np.expand_dims(leg_data, axis=-1)  # Shape (N=1, C, T, V_leg, M=1)
        full_body_data = np.expand_dims(full_body_data, axis=0)  # Shape (N=1, C, T, V_full_body)
        full_body_data = np.expand_dims(full_body_data, axis=-1)  # Shape (N=1, C, T, V_full_body, M=1)

        # Ensure data is Tensor type
        head_data = torch.FloatTensor(head_data)
        hand_data = torch.FloatTensor(hand_data)
        leg_data = torch.FloatTensor(leg_data)
        full_body_data = torch.FloatTensor(full_body_data)

        # Get the label
        label = torch.FloatTensor([self.labels[self.sample_names[index]]])

        return [head_data, hand_data, leg_data, full_body_data], label