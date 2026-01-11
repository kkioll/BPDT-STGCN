import numpy as np
import torch
import torch.nn.functional as F
from multiprocessing import freeze_support
from feeder.feeder import Feeder
from net.bpdt import Model
import torch.utils.data
import traceback
import os
import time

import config
import utils

def main():
    try:
        exp_dir, log_file = utils.create_experiment_directory(config.BASE_SAVE_DIR)
        dataset_name = utils.extract_dataset_name(config.TRAIN_DATA_DIR)
        utils.log(f"Experiment Directory: {exp_dir}", log_file)

        cfg = {k:v for k,v in config.__dict__.items() if not k.startswith('__')}
        utils.log_config(cfg, log_file)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        utils.log(f'Using device: {device}', log_file)
        if device.type == 'cuda':
            utils.log(f'CUDA Device Count: {torch.cuda.device_count()}', log_file)
            torch.backends.cudnn.benchmark = True

        utils.log("Loading training dataset...", log_file)
        train_dataset = Feeder(data_dir=config.TRAIN_DATA_DIR,label_path=config.TRAIN_LABEL_PATH,random_choose=False,random_move=False,window_size=-1,debug=False,augment=False,augment_train_only=False)
        utils.log(f"Training dataset loaded with {len(train_dataset)} samples", log_file)

        utils.pretrain_kmeans(train_dataset, log_file)

        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=2,collate_fn=utils.pad_collate)
        test_loader = torch.utils.data.DataLoader(Feeder(data_dir=config.TEST_DATA_DIR,label_path=config.TEST_LABEL_PATH,random_choose=False,random_move=False,window_size=-1,debug=False,augment=False,augment_train_only=False),batch_size=config.BATCH_SIZE,shuffle=False,num_workers=2,collate_fn=utils.pad_collate)
        utils.log(f"Test dataset loaded with {len(test_loader.dataset)} samples", log_file)

        save_dir = os.path.join(exp_dir, 'model')
        os.makedirs(save_dir, exist_ok=True)
        best_src_save_path = os.path.join(save_dir, f'st_gcn_best_src_{dataset_name}.pth')
        best_test_score_loss_save_path = os.path.join(save_dir, f'st_gcn_best_test_score_loss_{dataset_name}.pth')

        utils.log("Initializing model...", log_file)
        model = Model(in_channels=3,graph_args=config.GRAPH_ARGS,edge_importance_weighting=True).to(device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        optimizer = torch.optim.RMSprop(model.parameters(),lr=config.LR,weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=config.LR_DECAY_STEP,gamma=config.LR_DECAY_RATE)
        score_criterion = torch.nn.MSELoss(reduction='none')
        fuzzy_criterion = torch.nn.CrossEntropyLoss()

        best_src = -float('inf')
        best_src_epoch = 0
        best_test_score_loss = float('inf')
        best_test_loss_epoch = 0
        best_test_loss_preds = None
        best_test_loss_labels = None

        utils.log(f"------------------ Start Training ------------------", log_file)
        start_time = time.time()

        for epoch in range(1, config.NUM_EPOCH + 1):
            epoch_start_time = time.time()
            current_lr = optimizer.param_groups[0]['lr']
            utils.log(f"Epoch {epoch}/{config.NUM_EPOCH} | Current LR: {current_lr:.6f}", log_file)

            model.train()
            total_train_score_loss = 0.0
            total_train_fuzzy_loss = 0.0
            optimizer.zero_grad()

            for batch_idx, (data, label) in enumerate(train_loader):
                data = [d.to(device, dtype=torch.float) for d in data]
                label = label.to(device, dtype=torch.float)
                score_preds, fuzzy_preds = model(data)

                good_mask, medium_mask, bad_mask = utils.classify_samples(label)
                fuzzy_probs = F.softmax(fuzzy_preds, dim=1)
                _, fuzzy_classes = torch.max(fuzzy_preds, dim=1)
                good_pred_mask = (fuzzy_classes == 0)
                bad_pred_mask = (fuzzy_classes == 2)
                confidence = torch.max(fuzzy_probs, dim=1)[0]

                dynamic_rewards = torch.ones_like(confidence, device=device)
                good_pred_mask_flat = good_pred_mask.view(-1)
                dynamic_rewards[good_pred_mask_flat] = config.MIN_REWARD + (config.MAX_REWARD - config.MIN_REWARD) * confidence[good_pred_mask_flat]

                dynamic_punishments = torch.ones_like(confidence, device=device)
                bad_pred_mask_flat = bad_pred_mask.view(-1)
                dynamic_punishments[bad_pred_mask_flat] = config.MAX_PUNISHMENT - (config.MAX_PUNISHMENT - config.MIN_PUNISHMENT) * confidence[bad_pred_mask_flat]

                dynamic_coefficients = torch.ones_like(confidence, device=device)
                dynamic_coefficients[good_pred_mask_flat] = dynamic_rewards[good_pred_mask_flat]
                dynamic_coefficients[bad_pred_mask_flat] = dynamic_punishments[bad_pred_mask_flat]
                dynamic_coefficients = dynamic_coefficients.view(-1, 1)

                raw_score_loss = score_criterion(score_preds, label)
                weighted_score_loss = raw_score_loss * dynamic_coefficients
                score_loss = weighted_score_loss.mean()

                fuzzy_labels = torch.zeros(label.size(0), dtype=torch.long, device=device)
                fuzzy_labels[good_mask.view(-1)] = 0
                fuzzy_labels[medium_mask.view(-1)] = 1
                fuzzy_labels[bad_mask.view(-1)] = 2
                fuzzy_loss = fuzzy_criterion(fuzzy_preds, fuzzy_labels)

                total_loss = (score_loss + fuzzy_loss) / config.ACCUMULATION_STEPS
                total_loss.backward()

                total_train_score_loss += score_loss.item() * data[0].size(0)
                total_train_fuzzy_loss += fuzzy_loss.item() * data[0].size(0)

                if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            if (len(train_loader) % config.ACCUMULATION_STEPS) != 0:
                optimizer.step()
                optimizer.zero_grad()

            avg_train_score_loss = total_train_score_loss / len(train_dataset)
            avg_train_fuzzy_loss = total_train_fuzzy_loss / len(train_dataset)
            epoch_time = time.time() - epoch_start_time
            utils.log(f'Epoch {epoch:03d} | Train Score Loss: {avg_train_score_loss:.4f} | Train Fuzzy Loss: {avg_train_fuzzy_loss:.4f} | Time Cost: {epoch_time:.2f}s', log_file)

            current_src, current_preds, current_labels = None, None, None
            current_test_score_loss = None
            if config.EVAL_EVERY_EPOCH:
                current_src, current_preds, current_labels = utils.evaluate_on_test(model, test_loader, device, log_file)
                current_test_score_loss, _, _ = utils.calculate_test_loss(model, test_loader, device, score_criterion, log_file)

            if config.EVAL_EVERY_EPOCH and current_src is not None and current_src > best_src:
                best_src = current_src
                best_src_epoch = epoch
                state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save(state_dict, best_src_save_path)
                utils.log(f'  → Save Best SRC Model (SRC: {best_src:.4f})', log_file)
                utils.save_prediction_plot(current_preds, current_labels, os.path.join(exp_dir, f'best_src_predictions_vs_true.png'))
                utils.save_prediction_to_csv(current_preds, current_labels, os.path.join(exp_dir, f'best_src_predictions_vs_true.csv'))

            if config.EVAL_EVERY_EPOCH and current_test_score_loss is not None and current_test_score_loss < best_test_score_loss:
                best_test_score_loss = current_test_score_loss
                best_test_loss_epoch = epoch
                state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save(state_dict, best_test_score_loss_save_path)
                utils.log(f'  → Save Best Test MSE Model (Test MSE: {best_test_score_loss:.6f})', log_file)
                best_test_loss_preds, best_test_loss_labels = current_preds, current_labels
                utils.save_prediction_plot(current_preds, current_labels, os.path.join(exp_dir, f'best_test_score_loss_predictions_vs_true.png'))
                utils.save_prediction_to_csv(current_preds, current_labels, os.path.join(exp_dir, f'best_test_score_loss_predictions_vs_true.csv'))

            scheduler.step()
            utils.log("---------------------------------------------------", log_file)
            torch.cuda.empty_cache()

        total_training_time = time.time() - start_time
        if best_test_loss_preds is not None and best_test_loss_labels is not None:
            preds_mean = np.mean(best_test_loss_preds)
            preds_std = np.std(best_test_loss_preds)
            labels_mean = np.mean(best_test_loss_labels)
            labels_std = np.std(best_test_loss_labels)
            utils.log(f'Best Test MSE Model Prediction Statistics | Mean: {preds_mean:.4f} | Std: {preds_std:.4f}', log_file)
            utils.log(f'Best Test MSE Model Label Statistics | Mean: {labels_mean:.4f} | Std: {labels_std:.4f}', log_file)

        utils.log(f'Training Completed | Best SRC: {best_src:.4f} (Epoch {best_src_epoch}) | Best Test MSE: {best_test_score_loss:.6f} (Epoch {best_test_loss_epoch})', log_file)
        utils.log(f'Total Training Time: {total_training_time:.2f}s', log_file)

        with open(os.path.join(exp_dir, 'training_results.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Training Results Summary\nTotal Training Time: {total_training_time:.2f}s\nExperiment Directory: {exp_dir}\n\n")
            if best_test_loss_preds is not None:
                f.write(f"Best Test MSE Prediction Mean: {np.mean(best_test_loss_preds):.4f} | Std: {np.std(best_test_loss_preds):.4f}\n")
                f.write(f"Best Test MSE Label Mean: {np.mean(best_test_loss_labels):.4f} | Std: {np.std(best_test_loss_labels):.4f}\n\n")
            f.write(f"Best SRC: {best_src:.4f} (Epoch {best_src_epoch})\nBest MSE: {best_test_score_loss:.6f} (Epoch {best_test_loss_epoch})\n")

        utils.log("================== Experiment Completed ==================", log_file)

    except Exception as e:
        error_msg = traceback.format_exc()
        utils.log(f"Program Abnormal Termination: {str(e)}", None, 'ERROR')
        utils.log(error_msg, None, 'ERROR')
        raise

if __name__ == '__main__':
    freeze_support()
    main()