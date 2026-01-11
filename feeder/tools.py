import numpy as np


class SkeletonAugmentor:
    @staticmethod
    def random_rotate(data, angle_range=(-15, 15)):
        angle = np.random.uniform(*angle_range) * np.pi / 180
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        xy = data[:2, :, :].reshape(2, -1)
        rotated_xy = np.dot(rotation_matrix, xy).reshape(2, data.shape[1], data.shape[2])
        data[:2, :, :] = rotated_xy
        return data

    @staticmethod
    def random_scale(data, scale_range=(0.9, 1.1)):
        scale = np.random.uniform(*scale_range)
        data[:2, :, :] *= scale 
        return data

    @staticmethod
    def random_shift(data, shift_range=(-0.1, 0.1)):
        shift = np.random.uniform(*shift_range, size=(2, 1, 1))
        data[:2, :, :] += shift 
        return data

    @staticmethod
    def random_time_warp(data, sigma=0.2):
        T = data.shape[1]
        warp = np.cumsum(np.random.normal(1, sigma, T))
        warp = np.round((warp / warp[-1]) * (T - 1)).astype(int)
        return data[:, warp, :]

    @staticmethod
    def random_drop_joints(data, drop_prob=0.1):
        mask = np.random.random((1, 1, data.shape[2])) > drop_prob
        data[:2, :, :] *= mask 
        return data

    @staticmethod
    def add_noise(data, sigma=0.01):
        noise = np.random.normal(0, sigma, data.shape)
        noise[2, :, :] = 0  
        data += noise
        return data

    @staticmethod
    def temporal_crop_and_resize(data, min_ratio=0.7, max_ratio=1.0):
        T = data.shape[1]
        crop_len = np.random.randint(int(T * min_ratio), T + 1)
        start = np.random.randint(0, T - crop_len + 1)
        cropped = data[:, start:start + crop_len, :]

        new_timesteps = np.linspace(0, crop_len - 1, T).astype(int)
        return cropped[:, new_timesteps, :]

    @staticmethod
    def horizontal_flip(data, flip_prob=0.5):

        if np.random.random() >= flip_prob:
            return data
        data[0, :, :] *= -1
        left_right_pairs = [
            (2, 5), (3, 6), (4, 7),  
            (8, 11), (9, 12), (10, 13),  
            (14, 15), (16, 17) 
        ]
        for left_idx, right_idx in left_right_pairs:
            data[:, :, [left_idx, right_idx]] = data[:, :, [right_idx, left_idx]]

        return data

    @staticmethod
    def augment(data, rotation=True, scaling=True, shifting=True,
                time_warp=True, joint_drop=True, noise=True, time_crop=True,  horizontal_flip=True):

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
