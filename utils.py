import torch
import torch.nn.functional as F
import os
import time
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import csv

kmeans_model = None
cluster_centers = None
sorted_cluster_indices = None
cluster_counts = None


def pad_data(datas):
    max_T = max([d.size(2) for d in datas])
    padded = []
    for d in datas:
        pad_length = max_T - d.size(2)
        if pad_length > 0:
            padded_d = F.pad(d, (0, 0, 0, 0, 0, pad_length, 0, 0, 0, 0), mode='constant', value=0)
        else:
            padded_d = d
        padded.append(padded_d)
    batch_data = torch.cat(padded, dim=0)
    return batch_data


def pad_collate(batch):
    from config import SELECTED_BODY_PARTS
    datas, labels = zip(*batch)
    part_to_index = {'head': 0, 'hand': 1, 'leg': 2}
    selected_indices = [part_to_index[part] for part in SELECTED_BODY_PARTS]
    selected_datas_list = []
    for idx in selected_indices:
        part_datas = [data[idx] for data in datas]
        selected_datas_list.append(part_datas)
    padded_selected_datas = []
    for part_datas in selected_datas_list:
        padded_data = pad_data(part_datas)
        padded_selected_datas.append(padded_data)
    batch_labels = torch.stack(labels, dim=0)
    return padded_selected_datas, batch_labels


def create_experiment_directory(base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    existing_exps = [d for d in os.listdir(base_dir) if
                     os.path.isdir(os.path.join(base_dir, d)) and d.startswith('exp')]
    if existing_exps:
        exp_nums = [int(exp[3:]) for exp in existing_exps]
        new_exp_num = max(exp_nums) + 1
    else:
        new_exp_num = 1
    exp_dir = os.path.join(base_dir, f'exp{new_exp_num}')
    os.makedirs(exp_dir)
    log_file = os.path.join(exp_dir, 'training.log')
    return exp_dir, log_file


def extract_dataset_name(data_path):
    path_components = os.path.normpath(data_path).split(os.sep)
    dataset_name = None
    for component in reversed(path_components):
        if 'dataset' in component.lower() or 'data' in component.lower():
            dataset_name = component
            break
    if dataset_name is None and path_components:
        dataset_name = path_components[-1]
    return dataset_name or 'unknown'


def log(message, log_file=None, level='INFO'):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] [{level}] {message}"
    print(full_message)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(full_message + '\n')


def log_config(config, log_file):
    log("===== Experiment Configuration =====", log_file)
    for key, value in config.items():
        log(f"{key}: {value}", log_file)
    log("====================================", log_file)


def calculate_spearmanr(preds, labels):
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy().flatten()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy().flatten()
    if np.allclose(preds, preds[0]) or np.allclose(labels, labels[0]):
        log("Warning: Predictions or labels are all identical", level='WARNING')
        return float('nan')
    spearmanr, p_value = stats.spearmanr(preds, labels)
    return spearmanr


def save_prediction_plot(all_preds, all_labels, save_path):

    src_coefficient, p_value = stats.spearmanr(all_labels, all_preds)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 6))

    sample_index = np.arange(1, len(all_labels) + 1)

    plt.plot(sample_index, all_labels, label='Ground Truth', color='blue', linestyle='-', linewidth=1)
    plt.plot(sample_index, all_preds, label=f'Prediction Value (SRC={src_coefficient:.2f})',
             color='red', linestyle='--', linewidth=1)

    plt.scatter(sample_index, all_labels, color='blue', marker='o', s=30, alpha=0.7)
    plt.scatter(sample_index, all_preds, color='red', marker='o', s=30, alpha=0.7)

    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Score', fontsize=12)

    plt.legend(loc='upper left', fontsize=10)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlim(left=0, right=min(50, len(all_labels)))
    xticks = [0, 1] + list(range(5, min(51, len(all_labels) + 1), 5))
    plt.xticks(xticks)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_prediction_to_csv(all_preds, all_labels, save_path):
    with open(save_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['sample_index', 'true_value', 'prediction_value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(len(all_labels)):
            writer.writerow(
                {'sample_index': idx, 'true_value': float(all_labels[idx]), 'prediction_value': float(all_preds[idx])})


def classify_samples(labels):
    global kmeans_model, sorted_cluster_indices
    labels_np = labels.cpu().detach().numpy().reshape(-1, 1)
    clusters = kmeans_model.predict(labels_np)
    bad_cluster = sorted_cluster_indices[0]
    medium_cluster = sorted_cluster_indices[1]
    good_cluster = sorted_cluster_indices[2]
    bad_mask_np = (clusters == bad_cluster)
    medium_mask_np = (clusters == medium_cluster)
    good_mask_np = (clusters == good_cluster)
    bad_mask = torch.tensor(bad_mask_np, dtype=torch.bool, device=labels.device).view_as(labels)
    medium_mask = torch.tensor(medium_mask_np, dtype=torch.bool, device=labels.device).view_as(labels)
    good_mask = torch.tensor(good_mask_np, dtype=torch.bool, device=labels.device).view_as(labels)
    return good_mask, medium_mask, bad_mask


def pretrain_kmeans(train_dataset, log_file=None):
    global kmeans_model, cluster_centers, sorted_cluster_indices, cluster_counts
    from config import CLUSTER_NUM, KMEANS_N_INIT
    log("Pretraining KMeans clustering model...", log_file)
    all_labels = []
    for i in range(len(train_dataset)):
        _, label = train_dataset[i]
        all_labels.append(label.numpy())
    all_labels = np.concatenate(all_labels).reshape(-1, 1)
    all_labels_flat = all_labels.flatten()
    log(f"Extracted {len(all_labels)} training labels for KMeans pretraining", log_file)
    kmeans_model = KMeans(n_clusters=CLUSTER_NUM, random_state=42, n_init=KMEANS_N_INIT)
    clusters = kmeans_model.fit_predict(all_labels)
    cluster_centers = kmeans_model.cluster_centers_.flatten()
    cluster_counts = np.bincount(clusters)
    sorted_cluster_indices = np.argsort(cluster_centers)
    cluster_ranges = []
    for idx in sorted_cluster_indices:
        cluster_labels = all_labels_flat[clusters == idx]
        min_val = np.min(cluster_labels)
        max_val = np.max(cluster_labels)
        cluster_ranges.append((min_val, max_val))
    cluster_labels = ["bad", "medium", "good"]
    log("Clustering centers sorted by value:", log_file)
    for i, cluster_idx in enumerate(sorted_cluster_indices):
        center_val = cluster_centers[cluster_idx]
        count = cluster_counts[cluster_idx]
        min_val, max_val = cluster_ranges[i]
        log(f"  Cluster {i + 1} center: {center_val:.4f} Label: {cluster_labels[i]}, Count: {count}, Range: ({min_val:.4f}, {max_val:.4f})",
            log_file)
    return kmeans_model


def evaluate_on_test(model, test_loader, device, log_file=None):
    model.eval()
    log("Evaluating model on test set...", log_file)
    all_score_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data = [d.to(device, dtype=torch.float) for d in data]
            label = label.to(device, dtype=torch.float)
            score_preds, _ = model(data)
            all_score_preds.append(score_preds.cpu())
            all_labels.append(label.cpu())
    all_score_preds = torch.cat(all_score_preds, dim=0).numpy().flatten()
    all_labels = torch.cat(all_labels, dim=0).numpy().flatten()
    src = calculate_spearmanr(all_score_preds, all_labels)
    log(f"Test Set Spearman Rank Correlation (SRC): {src:.4f}", log_file)
    return src, all_score_preds, all_labels


def calculate_test_loss(model, test_loader, device, score_criterion, log_file=None):
    model.eval()
    log("Calculating test set MSE loss...", log_file)
    total_test_mse_loss = 0.0
    test_dataset_size = len(test_loader.dataset)
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data = [d.to(device, dtype=torch.float) for d in data]
            label = label.to(device, dtype=torch.float)
            score_preds, _ = model(data)
            raw_mse_loss = score_criterion(score_preds, label)
            batch_mse_loss = raw_mse_loss.mean()
            batch_size = data[0].size(0)
            total_test_mse_loss += batch_mse_loss.item() * batch_size
    avg_test_mse_loss = total_test_mse_loss / test_dataset_size
    log(f"Test Set MSE Loss: {avg_test_mse_loss:.6f}", log_file)
    return avg_test_mse_loss, 0.0, avg_test_mse_loss