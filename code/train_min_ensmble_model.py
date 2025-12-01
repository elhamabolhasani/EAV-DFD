from tqdm import tqdm
import torch
import shutil
from torch.utils import data as data_utils
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch import nn
from torch import optim
import os
import torch.multiprocessing

from dataset.audio_visual_dataset import TrainDataset
from models.ensemble_model.ensemble_model import MiniEavNet
from models.ensemble_model.mini_model_config import model_config
from models.utils import decision_making, prepare_model_input, calc_f1_and_auc, save_checkpoint, \
    load_checkpoint, logging_epoch_metrics_val, logging_epoch_metrics_train

# set writer
writer = SummaryWriter(log_dir=model_config.tensorboard_dir)
torch.multiprocessing.set_sharing_strategy('file_system')
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))


def calc_c_loss(feature_list, target, hyper_param=0.99):
    batch_size = target.size(0)
    vid_out = feature_list['contrastive_loss']['av_embedding']
    aud_out = feature_list['contrastive_loss']['va_embedding']
    loss = 0
    for batch in range(batch_size):
        dist = torch.dist(vid_out[batch,:].view(-1), aud_out[batch,:].view(-1), 2)
        dist = dist.to(target.device)
        tar = target[batch,:].view(-1)
        loss += (((1-tar)*(dist**2)) + (tar*(max(hyper_param-dist,0)**2)))

    return loss.mul_(1/batch_size)[0]


def train(model, train_data_loader, val_data_loader, logloss_av, logloss, ce_loss, optimizer, scheduler, checkpoint_dir, current_num_epoch,
          nepochs, device, best_auc):

    print("start training from epoch {} to {}".format(current_num_epoch, nepochs))
    for i in range(current_num_epoch, nepochs):
        print('epoch num : ', i)
        f1_scores, auc_scores, preds, labels = [], [], [], []
        preds_v, labels_v = [], []
        preds_a, labels_a = [], []
        preds_av, labels_av = [], []
        running_loss = 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        total_vid_ture_label = 0
        total_vid_ture_pred = 0
        for step, (vid_v, vid_av, aud_av, aud_a, y_av, y_v, y_a) in prog_bar:
            vid_av, aud_a, vid_v, mels = (
                prepare_model_input(aud_av, vid_av, aud_a, vid_v, device, model_config, kind='train'))

            # train
            model.train()
            optimizer.zero_grad()

            v_outputs, av_outputs, a_outputs, _, feature_list, _, _ = model(vid_v, vid_av, mels, aud_a)


            final_labels, final_probs, v_labels, av_labels, a_labels, a_probs, v_probs, av_probs = (
                decision_making(v_outputs, av_outputs, a_outputs, model_config, len(y_av)))

            # calculate loss
            # split loss
            loss = (logloss(v_outputs, y_v.to(device)) +
                    logloss_av(av_outputs, y_av.squeeze(-1).to(device)) +
                    logloss(a_outputs, y_a.to(device)))

            if model_config.add_joint_loss is True:
                loss += ce_loss(final_probs.to(device),y_av.to(device))

            if model_config.contrastive_loss is True:
                loss += calc_c_loss(feature_list, y_av) * model_config.c_loss_var

            # need to remove
            total_vid_ture_label += sum(y_v)
            total_vid_ture_pred += sum(v_labels[:][0])

            loss.backward()
            optimizer.step()

            # calculate metrics
            f1_metric , auc_metric = calc_f1_and_auc(y_av.clone().detach().cpu().numpy(),
                                                     final_probs.clone().detach().cpu().numpy(),
                                                     final_labels)

            f1_scores.append(f1_metric)
            auc_scores.append(auc_metric)
            preds += list(final_labels)
            preds_v += [[v_labels[i][0]] for i in range(len(y_v))]
            preds_av += [[av_labels[i]] for i in range(len(y_av))]
            preds_a += [[a_labels[i][0]] for i in range(len(y_a))]
            labels += list(y_av.clone().detach().cpu().numpy())
            labels_v += list(y_v.clone().detach().cpu().numpy())
            labels_av += list(y_av.clone().detach().cpu().numpy())
            labels_a += list(y_a.clone().detach().cpu().numpy())
            running_loss += loss.item()

            prog_bar.set_description('[TRAINING LOSS]: {}, [TRAINING F1]: {}, [TRAINING AUC]: {}'
                                     .format(running_loss / (step + 1), np.mean(f1_scores), np.nanmean(auc_scores)))

        if scheduler is not None:
            scheduler.step()

        logging_epoch_metrics_train(f1_scores, auc_scores, labels, preds, labels_v, preds_v, labels_a, preds_a,
                      labels_av, preds_av, running_loss, i, len(train_data_loader), total_vid_ture_label,
                      total_vid_ture_pred, writer, mode='train')

        # saving model
        if i % model_config.save_model_interval == 0:
            print('saved epoch : ', i)
            save_checkpoint(model, optimizer, scheduler, checkpoint_dir, i, best_auc, is_best=False)

        # model evaluation
        with torch.no_grad():
            auc_eval = eval_model(model, val_data_loader, logloss_av, logloss, ce_loss, device, i)
            if auc_eval > best_auc:
                best_auc = auc_eval
                save_checkpoint(model, optimizer, scheduler, checkpoint_dir, i, best_auc, is_best=True)


def eval_model(model, val_data_loader, logloss_av, logloss, ce_loss, device, n_epoch):
    losses, f1_scores, auc_scores, preds, labels = [], [], [], [], []
    running_loss = 0
    prog_bar = tqdm(enumerate(val_data_loader))
    total_vid_ture_label = 0
    total_vid_ture_pred = 0
    for step, (vid_v, vid_av, aud_av, aud_a, y_av, y_v, y_a) in prog_bar:

        model.eval()
        with torch.no_grad():
            vid_av, aud_a, vid_v, mels = (
                prepare_model_input(aud_av, vid_av, aud_a, vid_v, device, model_config, kind='train'))

            # eval model
            v_outputs, av_outputs, a_outputs, _, feature_list, _, _ = model(vid_v, vid_av, mels, aud_a)

            final_labels, final_probs, v_labels, av_labels, a_labels, a_probs, v_probs, av_probs = (
                decision_making(v_outputs, av_outputs, a_outputs, model_config, len(y_av)))

            # calculate loss
            # split loss
            loss = (logloss(v_outputs, y_v.to(device)) +
                        logloss_av(av_outputs, y_av.squeeze(-1).to(device)) +
                        logloss(a_outputs, y_a.to(device)))

            # joint loss
            if model_config.add_joint_loss is True:
                loss += ce_loss(final_probs.to(device), y_av.to(device))

            # contrastive loss
            if model_config.contrastive_loss is True:
                loss += calc_c_loss(feature_list, y_av) * model_config.c_loss_var

            # need to remove
            total_vid_ture_label += sum(y_v)
            total_vid_ture_pred += sum(v_labels[:][0])

            losses.append(loss.item())

            # calculate metrics
            f1_metric, auc_metric = calc_f1_and_auc(y_av.clone().detach().cpu().numpy(),
                                                    final_probs.clone().detach().cpu().numpy(),
                                                    final_labels)

            f1_scores.append(f1_metric)
            auc_scores.append(auc_metric)
            running_loss += loss.item()

            prog_bar.set_description('[VAL RUNNING LOSS]: {}, [VAL F1]: {}, [VAL AUC]: {}'
                                     .format(running_loss / (step + 1), np.mean(f1_scores), np.nanmean(auc_scores)))

            preds += list(final_labels)
            labels += list(y_av.clone().detach().cpu().numpy())


    auc_metric = logging_epoch_metrics_val(f1_scores, auc_scores, labels, preds, running_loss, n_epoch, len(val_data_loader), total_vid_ture_label,
                          total_vid_ture_pred, writer, mode='val')

    return auc_metric


if __name__ == "__main__":
    # make checkpoint directory
    if not os.path.exists(model_config.checkpoint_dir):
        os.mkdir(model_config.checkpoint_dir)

    # copy config file to checkpoint dir
    shutil.copy2(model_config.config_file_path,
                 os.path.join(model_config.tensorboard_dir, model_config.config_file_name))

    # dataset and dataloader setup
    train_dataset = TrainDataset('train', model_config)
    val_dataset = TrainDataset('val', model_config)

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=model_config.batch_size, shuffle=True,
        num_workers=model_config.num_workers)

    val_data_loader = data_utils.DataLoader(
        val_dataset, batch_size=model_config.batch_size,
        num_workers=model_config.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # model
    model = MiniEavNet(model_config)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    model = model.to(device)

    if model_config.loss_fn == 'bce_with_logits_weighted':
        logloss_av = nn.BCEWithLogitsLoss(weight=torch.tensor([model_config.loss_weight_av]).to(device))
        logloss = nn.BCEWithLogitsLoss(weight=torch.tensor([model_config.loss_weight_a_v]).to(device))
    else:
        logloss_av = nn.BCEWithLogitsLoss()
        logloss = nn.BCEWithLogitsLoss()

    if model_config.ce_loss == 'bce_weighted':
        ce_loss = nn.BCELoss(weight=torch.tensor([model_config.loss_weight_ce]).to(device))
    else:
        ce_loss = nn.BCELoss()

    if model_config.optimizer == 'adam':
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=model_config.lr,
                               weight_decay=model_config.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=model_config.lr, momentum=0.9)

    if model_config.scheduler is not None:
        scheduler = StepLR(optimizer, step_size=model_config.scheduler_step_size, gamma=model_config.gamma)
    else:
        scheduler = None

    if model_config.model_checkpoint_path is not None:
        model, last_epoch, best_auc = load_checkpoint(model_config.model_checkpoint_path, model, kind='main')
    else:
        last_epoch = 0
        best_auc = 0


    # train model
    current_num_epoch = 0
    print('start training')
    train(model, train_data_loader, val_data_loader, logloss_av, logloss, ce_loss, optimizer, scheduler, model_config.checkpoint_dir,
          current_num_epoch, model_config.n_epochs, device, best_auc)
