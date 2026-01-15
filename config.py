import os

# ===================== Path Configuration =====================
TRAIN_DATA_DIR = r'D:\BPDT-STGCM\datasets\ball\train'
TRAIN_LABEL_PATH = r'D:\BPDT-STGCM\datasets\ball\train.pkl'
TEST_DATA_DIR = r'D:\BPDT-STGCM\datasets\ball\test'
TEST_LABEL_PATH = r'D:\BPDT-STGCM\datasets\ball\test.pkl'
BASE_SAVE_DIR = './run'

# ===================== Training Hyperparameters =====================
NUM_EPOCH = 280
BATCH_SIZE = 4
LR = 1e-3
LR_DECAY_RATE = 0.1
LR_DECAY_STEP = 70
ACCUMULATION_STEPS = 4
SELECTED_BODY_PARTS = ['head', 'hand', 'leg']

# ===================== Model & Evaluation Config =====================
CLUSTER_NUM = 3
EVAL_EVERY_EPOCH = True
SAVE_BEST_SRC = True
MIN_REWARD = 1.1
MAX_REWARD = 1.35
MIN_PUNISHMENT = 1.1
MAX_PUNISHMENT = 1.35
KMEANS_N_INIT = 10
GRAPH_ARGS = {'layout': 'openpose', 'strategy': 'uniform'}

# ===================== Fixed Config =====================

DEVICE = 'cuda' if __name__ == '__main__' else None


