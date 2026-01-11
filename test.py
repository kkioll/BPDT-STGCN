import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
import os
import torch.utils.data
import csv
import matplotlib.pyplot as plt
from feeder.feeder import Feeder
from net.bpdt import Model

WEIGHTS_PATH = r"D:\Github-BPDT-STGCM\weight\ribbon_best.pth"
TEST_DATA_DIR = r'E:\ACTION-NET-master\datasets\ribbon_datasets\test'
TEST_LABEL_PATH = r'F:\my-st-gcn\datasets\ribbon_datasets\test.pkl'
BATCH_SIZE = 4
SELECTED_BODY_PARTS = ['head', 'hand', 'leg']

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_SAVE_DIR = os.path.join(CURRENT_DIR, 'test_results')
os.makedirs(RESULT_SAVE_DIR, exist_ok=True)
CSV_SAVE_PATH = os.path.join(RESULT_SAVE_DIR, 'prediction_results.csv')
PLOT_SAVE_PATH = os.path.join(RESULT_SAVE_DIR, 'prediction_vs_truth.png')
REPORT_SAVE_PATH = os.path.join(RESULT_SAVE_DIR, 'evaluation_report.txt')


def pad_data(datas):
    max_T = max([d.size(2) for d in datas])
    padded = []
    for d in datas:
        pad_length = max_T - d.size(2)
        padded_d = F.pad(d, (0, 0, 0, 0, 0, pad_length, 0, 0, 0, 0), mode='constant', value=0) if pad_length > 0 else d
        padded.append(padded_d)
    batch_data = torch.cat(padded, dim=0)
    return batch_data


def pad_collate(batch):
    datas, labels = zip(*batch)
    part_to_index = {'head': 0, 'hand': 1, 'leg': 2}
    selected_indices = [part_to_index[part] for part in SELECTED_BODY_PARTS]
    selected_datas_list = [[data[idx] for data in datas] for idx in selected_indices]
    padded_selected_datas = [pad_data(pd) for pd in selected_datas_list]
    batch_labels = torch.stack(labels, dim=0)
    return padded_selected_datas, batch_labels


def calculate_spearmanr(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    if np.allclose(preds, preds[0]) or np.allclose(labels, labels[0]):
        return float('nan')
    spearmanr_coef, _ = spearmanr(preds, labels)
    return spearmanr_coef


def save_to_csv(sample_ids, true_vals, pred_vals, save_path):
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Sample_ID', 'Ground_Truth', 'Prediction'])
        for sid, tv, pv in zip(sample_ids, true_vals, pred_vals):
            writer.writerow([sid, f'{tv:.6f}', f'{pv:.6f}'])


# 完全沿用你指定的可视化绘图样式，无任何修改
def save_plot(true_vals, pred_vals, save_path, src_coef):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 6))
    sample_index = np.arange(1, len(true_vals) + 1)

    plt.plot(sample_index, true_vals, label='Ground Truth', color='blue', linestyle='-', linewidth=1)
    plt.plot(sample_index, pred_vals, label=f'Prediction Value (SRC={src_coef:.2f})', color='red', linestyle='--',
             linewidth=1)
    plt.scatter(sample_index, true_vals, color='blue', marker='o', s=30, alpha=0.7)
    plt.scatter(sample_index, pred_vals, color='red', marker='o', s=30, alpha=0.7)

    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Ribbon', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlim(left=0, right=50)
    plt.xticks([0, 1] + list(range(5, 51, 5)))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_report(mse_loss, src_coef, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('Evaluation Report\n')
        f.write('=' * 50 + '\n')
        f.write(f'Mean MSE Loss: {mse_loss:.8f}\n')
        f.write(f'Spearman Rank Correlation Coefficient: {src_coef:.8f}\n')
        f.write('=' * 50 + '\n')
        f.write(f'Weight Path: {WEIGHTS_PATH}\n')
        f.write(f'Test Data Dir: {TEST_DATA_DIR}\n')


def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    test_dataset = Feeder(
        data_dir=TEST_DATA_DIR,
        label_path=TEST_LABEL_PATH,
        random_choose=False,
        random_move=False,
        window_size=-1,
        debug=False,
        augment=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        collate_fn=pad_collate, pin_memory=True
    )

    print(f'Test dataset loaded, total samples: {len(test_dataset)}')
    print('-' * 70)

    graph_args = {'layout': 'openpose', 'strategy': 'uniform'}
    model = Model(in_channels=3, graph_args=graph_args, edge_importance_weighting=True).to(device)

    assert os.path.exists(WEIGHTS_PATH), f'Weight file not found: {WEIGHTS_PATH}'
    assert os.path.getsize(WEIGHTS_PATH) > 1024, f'Invalid weight file'
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)

    if list(checkpoint.keys())[0].startswith('module.'):
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    print(f'Pretrained weights loaded: {WEIGHTS_PATH}')
    print('-' * 70)

    model.eval()
    total_mse_loss = 0.0
    all_preds, all_labels = [], []
    sample_count = 0
    sample_ids, true_vals, pred_vals = [], [], []

    print('{:>9s} | {:>11s} | {:>10s}'.format('Sample ID', 'Ground-truth', 'Prediction'))
    print('-' * 70)

    with torch.no_grad():
        for data, label in test_loader:
            data = [d.to(device, dtype=torch.float) for d in data]
            label = label.to(device, dtype=torch.float)
            score_preds, _ = model(data)

            total_mse_loss += F.mse_loss(score_preds, label, reduction='sum').item()
            preds_np = score_preds.cpu().numpy()
            labels_np = label.cpu().numpy()

            all_preds.append(preds_np)
            all_labels.append(labels_np)

            for i in range(len(preds_np)):
                sample_count += 1
                true_score = labels_np[i][0]
                pred_score = preds_np[i][0]
                print(f'{sample_count:>9d} | {true_score:>11.6f} | {pred_score:>10.6f}')
                sample_ids.append(sample_count)
                true_vals.append(true_score)
                pred_vals.append(pred_score)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_mse_loss = total_mse_loss / len(test_dataset)
    src = calculate_spearmanr(all_preds, all_labels)

    print('-' * 70)
    print('Evaluation Results')
    print('-' * 70)
    print(f'Mean MSE Loss: {avg_mse_loss:.8f}')
    print(f'Spearman Rank Correlation: {src:.8f}')
    print('-' * 70)

    # 保存所有结果到当前目录
    save_to_csv(sample_ids, true_vals, pred_vals, CSV_SAVE_PATH)
    save_plot(true_vals, pred_vals, PLOT_SAVE_PATH, src)
    save_report(avg_mse_loss, src, REPORT_SAVE_PATH)

    print(f' All results saved to current path: {RESULT_SAVE_DIR}')
    print(f' CSV: {CSV_SAVE_PATH}')
    print(f' Plot: {PLOT_SAVE_PATH}')
    print(f' Report: {REPORT_SAVE_PATH}')


if __name__ == '__main__':
    evaluate()