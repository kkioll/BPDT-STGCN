import numpy as np


class SkeletonAugmentor:
    """
    骨骼点数据增强器，专为 Shape (3, T, 18) 格式数据设计
    3个通道分别表示 (x, y, acc)，T是帧数，18是关节点数
    """

    @staticmethod
    def random_rotate(data, angle_range=(-15, 15)):
        """随机旋转整个骨架"""
        angle = np.random.uniform(*angle_range) * np.pi / 180
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

        # 提取x,y坐标并旋转
        xy = data[:2, :, :].reshape(2, -1)
        rotated_xy = np.dot(rotation_matrix, xy).reshape(2, data.shape[1], data.shape[2])
        data[:2, :, :] = rotated_xy
        return data

    @staticmethod
    def random_scale(data, scale_range=(0.9, 1.1)):
        """随机缩放骨架"""
        scale = np.random.uniform(*scale_range)
        data[:2, :, :] *= scale  # 只缩放x,y坐标
        return data

    @staticmethod
    def random_shift(data, shift_range=(-0.1, 0.1)):
        """随机平移骨架"""
        shift = np.random.uniform(*shift_range, size=(2, 1, 1))
        data[:2, :, :] += shift  # 只平移x,y坐标
        return data

    @staticmethod
    def random_time_warp(data, sigma=0.2):
        """随机时间扭曲，改变动作速度"""
        T = data.shape[1]
        # 生成平滑的时间映射
        warp = np.cumsum(np.random.normal(1, sigma, T))
        warp = np.round((warp / warp[-1]) * (T - 1)).astype(int)
        # 应用时间扭曲
        return data[:, warp, :]

    @staticmethod
    def random_drop_joints(data, drop_prob=0.1):
        """随机丢弃关节点，模拟检测失败"""
        mask = np.random.random((1, 1, data.shape[2])) > drop_prob
        data[:2, :, :] *= mask  # 只对x,y坐标应用掩码
        return data

    @staticmethod
    def add_noise(data, sigma=0.01):
        """添加高斯噪声"""
        noise = np.random.normal(0, sigma, data.shape)
        noise[2, :, :] = 0  # 不向acc通道添加噪声
        data += noise
        return data

    @staticmethod
    def temporal_crop_and_resize(data, min_ratio=0.7, max_ratio=1.0):
        """时间维度上的裁剪和缩放"""
        T = data.shape[1]
        crop_len = np.random.randint(int(T * min_ratio), T + 1)
        start = np.random.randint(0, T - crop_len + 1)
        cropped = data[:, start:start + crop_len, :]

        # 使用线性插值进行resize回原长度
        new_timesteps = np.linspace(0, crop_len - 1, T).astype(int)
        return cropped[:, new_timesteps, :]

    @staticmethod
    def horizontal_flip(data, flip_prob=0.5):
        """
        以给定概率进行水平翻转
        翻转时x坐标取反，同时交换左右关节点
        """
        if np.random.random() >= flip_prob:
            return data

        # 对x坐标取反 (第一通道)
        data[0, :, :] *= -1

        # 定义左右关节点的交换映射
        # 基于18点结构: [Nose, Neck, RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle, REye, LEye, REar, LEar]
        left_right_pairs = [
            (2, 5), (3, 6), (4, 7),  # 肩部、肘部、腕部
            (8, 11), (9, 12), (10, 13),  # 髋部、膝部、踝部
            (14, 15), (16, 17)  # 眼睛、耳朵
        ]

        # 交换左右关节点
        for left_idx, right_idx in left_right_pairs:
            data[:, :, [left_idx, right_idx]] = data[:, :, [right_idx, left_idx]]

        return data

    @staticmethod
    def augment(data, rotation=True, scaling=True, shifting=True,
                time_warp=True, joint_drop=True, noise=True, time_crop=True,  horizontal_flip=True):
        """
        组合多种增强方法
        """
        if rotation:
            data = SkeletonAugmentor.random_rotate(data)
        if scaling:
            data = SkeletonAugmentor.random_scale(data)
        if shifting:
            data = SkeletonAugmentor.random_shift(data)
        if time_warp:
            data = SkeletonAugmentor.random_time_warp(data)
        if joint_drop:
            data = SkeletonAugmentor.random_drop_joints(data)
        if noise:
            data = SkeletonAugmentor.add_noise(data)
        if time_crop:
            data = SkeletonAugmentor.temporal_crop_and_resize(data)
        if horizontal_flip:
            data = SkeletonAugmentor.horizontal_flip(data)

        return data