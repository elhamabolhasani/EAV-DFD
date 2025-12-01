from tqdm import tqdm
import importlib
import torch
import shutil
from torch.utils import data as data_utils
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch import nn
from torch import optim
import os, random
import torch.multiprocessing

from dataset.audio_visual_dataset import TrainDataset
from models.teacher_student_model.teacher_student_config import ts_config
from models.teacher_student_model.model_config import model_config
from models.teacher_student_model.teacher_student_network import TeacherStudentNetwork
from models.utils import decision_making, prepare_model_input, calc_f1_and_auc, save_checkpoint, \
    logging_epoch_metrics_val, logging_epoch_metrics_train, load_checkpoint

random.seed(1234)

# set writer
writer = SummaryWriter(log_dir=ts_config.tensorboard_dir)
torch.multiprocessing.set_sharing_strategy('file_system')


def ensemble_label(y_av, y_a, y_v):
    y_e = []
    for x, y, z in zip(y_av.squeeze(-1), y_a, y_v):
        # real
        if sum([x, y, z]) == 3:
            y_e.append([1])
        # fake
        else:
            y_e.append([0])

    return torch.Tensor(y_e)


def calculate_loss(loss_array, t_v_outputs, t_av_outputs, t_a_outputs, t_v_hidden, t_feature_list, t_a_hidden,
                   t_av_hidden, s_v_outputs, s_av_outputs, s_a_outputs, s_v_hidden, s_feature_list, s_a_hidden,
                   s_av_hidden, y_av, y_v, y_a, classifier_type, model, ts_config):
    logloss = loss_array['logloss']
    logloss_av = loss_array['logloss_av']
    ce_loss_fn = loss_array['ce_loss']
    sigmoid_layer = nn.Sigmoid()

    final_labels, final_probs, v_labels, av_labels, a_labels, a_probs, v_probs, av_probs = \
        (decision_making(s_v_outputs, s_av_outputs, s_a_outputs, model_config, len(y_av)))
    
    if  hasattr(ts_config, 'classifier_type') and ts_config.classifier_type == 'joint':
        ce_loss = logloss_av(s_av_outputs, y_av.squeeze(-1).to(device))
    
    elif hasattr(ts_config, 'classifier_type') and ts_config.classifier_type == 'attention':
        ce_loss = logloss_av(s_av_outputs, y_av.squeeze(-1).to(device))
    
    else:
        ce_loss = (logloss(s_v_outputs.squeeze(-1), y_v.to(device).squeeze(-1)) +
                logloss_av(s_av_outputs, y_av.squeeze(-1).to(device)) +
                logloss(s_a_outputs.squeeze(-1), y_a.to(device).squeeze(-1)))

    if ts_config.joint_loss_flag:
        y_e = ensemble_label(y_av, y_a, y_v)
        ce_loss += ce_loss_fn(final_probs.to(device), y_e.to(device))

    out_dict = {
        "tea_fea": t_feature_list,
        "stu_fea": s_feature_list,
        "y_t": t_av_outputs,
        "y_s": s_av_outputs,
        "label": y_av,
    }

    adapt_loss_av_loss_arr = []
    for key in ts_config.loss_subconfig.keys():
        train_disloss = loss_array['adapt_loss' + key]
        adapt_loss_av_loss_arr.append(train_disloss(out_dict))

    distill_av_loss = sum(adapt_loss_av_loss_arr)

    mse_loss = loss_array['mse_loss']
    kl_loss = loss_array['kl_loss']

    distill_mse_loss = mse_loss(t_v_hidden, s_v_hidden) + mse_loss(t_a_hidden, s_a_hidden)

    if ts_config.dynamic_weight:
        y_av_weight = abs(2 * sigmoid_layer(t_av_outputs) - 1)
        y_a_weight = abs(2 * sigmoid_layer(t_a_outputs) - 1)
        y_v_weight = abs(2 * sigmoid_layer(t_v_outputs) - 1)
        kl_output_loss = (kl_loss(out_dict={"y_t": y_v_weight * sigmoid_layer(t_v_outputs),
                                            "y_s": y_v_weight * sigmoid_layer(s_v_outputs)})
                          + kl_loss(out_dict={"y_t": y_a_weight * sigmoid_layer(t_a_outputs),
                                              "y_s": y_a_weight * sigmoid_layer(s_a_outputs)})
                          + kl_loss(out_dict={"y_t": y_av_weight * sigmoid_layer(t_av_outputs),
                                              "y_s": y_av_weight * sigmoid_layer(s_av_outputs)}))
    else:
        if ts_config.kl_output_loss_flag is True:
            kl_output_loss = (kl_loss(out_dict={"y_t": sigmoid_layer(t_v_outputs), "y_s": sigmoid_layer(s_v_outputs)})
                            + kl_loss(out_dict={"y_t": sigmoid_layer(t_a_outputs), "y_s": sigmoid_layer(s_a_outputs)})
                            + kl_loss(out_dict={"y_t": sigmoid_layer(t_av_outputs), "y_s": sigmoid_layer(s_av_outputs)}))

    total_loss = None
    if ts_config.ce_loss_flag is True:
        print('ce_loss : ', ce_loss.item())
        total_loss = ts_config.ce_loss_var * ce_loss
    if ts_config.adapt_mse_loss_flag is True:
        print('distill_mse_loss : ', distill_mse_loss.item())
        total_loss += distill_mse_loss
    if ts_config.adapt_av_loss_flag is True:
        if hasattr(ts_config, 'adapt_av_loss_var'):
            print('distill_av_loss : ', ts_config.adapt_av_loss_var * distill_av_loss.item())
            total_loss += distill_av_loss * ts_config.adapt_av_loss_var
        else:
            print('distill_av_loss : ', distill_av_loss.item())
            total_loss += distill_av_loss
    if ts_config.kl_output_loss_flag is True:
        print('kl_output_loss : ', kl_output_loss.item())
        total_loss += kl_output_loss

    return total_loss, [final_labels, final_probs, v_labels, av_labels, a_labels, a_probs, v_probs, av_probs]


def train(model, train_data_loader, val_data_loader, loss_array, optimizer, scheduler, checkpoint_dir,
          current_num_epoch, nepochs, device, best_auc):
    print("start training from epoch {} to {}".format(current_num_epoch, nepochs))

    for i in range(current_num_epoch, nepochs):
        print('epoch num : ', i)
        f1_scores, auc_scores, preds, labels = [], [], [], []
        preds_v, labels_v = [], []
        preds_a, labels_a = [], []
        preds_av, labels_av = [], []
        total_vid_ture_label = 0
        total_vid_ture_pred = 0
        running_loss = 0.
        prog_bar = tqdm(enumerate(train_data_loader))

        for step, (vid_v, vid_av, aud_av, aud_a, y_av, y_v, y_a) in prog_bar:
            vid_av, aud_a, vid_v, mels = (
                prepare_model_input(aud_av, vid_av, aud_a, vid_v, device, model_config, kind='train'))

            # train
            model.student.train()

            (t_v_outputs, t_av_outputs, t_a_outputs, t_v_hidden, t_feature_list, t_a_hidden, t_av_hidden, s_v_outputs,
             s_av_outputs, s_a_outputs, s_v_hidden, s_feature_list, s_a_hidden, s_av_hidden) \
                = model(vid_v, vid_av, mels, aud_a)

            loss, decision_making_output = calculate_loss(loss_array, t_v_outputs, t_av_outputs, t_a_outputs,
                                                          t_v_hidden, t_feature_list,
                                                          t_a_hidden, t_av_hidden, s_v_outputs, s_av_outputs,
                                                          s_a_outputs, s_v_hidden,
                                                          s_feature_list, s_a_hidden, s_av_hidden, y_av, y_v, y_a,
                                                          model_config.classifier_type,
                                                          model, ts_config)

            final_labels, final_probs, v_labels, av_labels, a_labels, a_probs, v_probs, av_probs = decision_making_output
            total_vid_ture_label += sum(y_v)
            total_vid_ture_pred += sum(v_labels[:][0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_e = ensemble_label(y_av, y_a, y_v)
            f1_metric, auc_metric = calc_f1_and_auc(y_e,
                                                    final_probs.clone().detach().cpu().numpy(),
                                                    final_labels)

            f1_scores.append(f1_metric)
            auc_scores.append(auc_metric)
            running_loss += loss.item()

            prog_bar.set_description('[TRAINING LOSS]: {}, [TRAINING F1]: {}, [TRAINING AUC]: {}'
                                     .format(running_loss / (step + 1), np.mean(f1_scores), np.nanmean(auc_scores)))

            preds += list(final_labels)
            labels += list(y_e.clone().detach().cpu().numpy())

            preds_v += [[v_labels[i][0]] for i in range(len(y_v))]
            preds_av += [[av_labels[i]] for i in range(len(y_av))]
            preds_a += [[a_labels[i][0]] for i in range(len(y_a))]
            labels_v += list(y_v.clone().detach().cpu().numpy())
            labels_av += list(y_av.clone().detach().cpu().numpy())
            labels_a += list(y_a.clone().detach().cpu().numpy())

        if scheduler is not None:
            scheduler.step()

        logging_epoch_metrics_train(f1_scores, auc_scores, labels, preds, labels_v, preds_v, labels_a, preds_a,
                                    labels_av, preds_av, running_loss, i, len(train_data_loader), total_vid_ture_label,
                                    total_vid_ture_pred, writer, mode='train')

        # saving model
        if i % model_config.save_model_interval == 0 or i <= 5:
            print('saved epoch : ', i)
            save_checkpoint(model, optimizer, scheduler, checkpoint_dir, i, best_auc, is_best=False, kind='student')

        # model evaluation
        with torch.no_grad():
            auc_eval = eval_model(model, val_data_loader, loss_array, device, i)
            if auc_eval > best_auc:
                best_auc = auc_eval
                save_checkpoint(model, optimizer, scheduler, checkpoint_dir, i, best_auc, is_best=True, kind='student')


def eval_model(model, val_data_loader, loss_array, device, n_epoch):
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
            (t_v_outputs, t_av_outputs, t_a_outputs, t_v_hidden, t_feature_list, t_a_hidden, t_av_hidden, s_v_outputs,
             s_av_outputs, s_a_outputs, s_v_hidden, s_feature_list, s_a_hidden, s_av_hidden) \
                = model(vid_v, vid_av, mels, aud_a)

            loss, decision_making_output = calculate_loss(loss_array, t_v_outputs, t_av_outputs, t_a_outputs,
                                                          t_v_hidden, t_feature_list,
                                                          t_a_hidden, t_av_hidden, s_v_outputs, s_av_outputs,
                                                          s_a_outputs, s_v_hidden,
                                                          s_feature_list, s_a_hidden, s_av_hidden, y_av, y_v, y_a,
                                                          model_config.classifier_type,
                                                          model, ts_config)

            final_labels, final_probs, v_labels, av_labels, a_labels, a_probs, v_probs, av_probs = decision_making_output

            total_vid_ture_label += sum(y_v)
            total_vid_ture_pred += sum(v_labels[:][0])

            losses.append(loss.item())

            y_e = ensemble_label(y_av, y_a, y_v)
            f1_metric, auc_metric = calc_f1_and_auc(y_e,
                                                    final_probs.clone().detach().cpu().numpy(),
                                                    final_labels)

            auc_scores.append(auc_metric)
            f1_scores.append(f1_metric)
            running_loss += loss.item()

            prog_bar.set_description('[VAL RUNNING LOSS]: {}, [VAL F1]: {}, [VAL AUC]: {}'
                                     .format(running_loss / (step + 1), np.mean(f1_scores), np.nanmean(auc_scores)))

            preds += list(final_labels)
            labels += list(y_e.clone().detach().cpu().numpy())


    auc_metric = logging_epoch_metrics_val(f1_scores, auc_scores, labels, preds, running_loss, n_epoch, len(val_data_loader),
                              total_vid_ture_label,
                              total_vid_ture_pred, writer, mode='val')
    return auc_metric


if __name__ == "__main__":
    # make checkpoint directory
    if not os.path.exists(ts_config.checkpoint_dir):
        os.mkdir(ts_config.checkpoint_dir)

    # copy config file to checkpoint dir
    shutil.copy2(ts_config.config_file_path,
                 os.path.join(ts_config.tensorboard_dir, ts_config.config_file_name))
    shutil.copy2(ts_config.ts_config_file_path,
                 os.path.join(ts_config.tensorboard_dir, ts_config.ts_config_file_name))

    # dataset and dataloader setup
    print(model_config.run_name)
    train_dataset = TrainDataset('train', model_config)
    val_dataset = TrainDataset('val', model_config)

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=model_config.batch_size, shuffle=True,
        num_workers=model_config.num_workers)

    val_data_loader = data_utils.DataLoader(
        val_dataset, batch_size=model_config.batch_size,
        num_workers=model_config.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    ts_model = TeacherStudentNetwork(ts_config)
    print('total trainable params {}'.format(sum(p.numel() for p in ts_model.parameters() if p.requires_grad)))
    ts_model = ts_model.to(device)

    loss_array = {}
    if ts_config.loss_fn == 'bce_with_logits_weighted':
        logloss_av = nn.BCEWithLogitsLoss(weight=torch.tensor([ts_config.loss_weight_av]).to(device))
        logloss = nn.BCEWithLogitsLoss(weight=torch.tensor([ts_config.loss_weight_a_v]).to(device))
    else:
        logloss_av = nn.BCEWithLogitsLoss()
        logloss = nn.BCEWithLogitsLoss()

    if ts_config.ce_loss == 'bce_weighted':
        ce_loss = nn.BCELoss(weight=torch.tensor([ts_config.loss_weight_ce]).to(device))
    else:
        ce_loss = nn.BCELoss()

    loss_array['logloss_av'] = logloss_av
    loss_array['logloss'] = logloss
    loss_array['ce_loss'] = ce_loss

    # teacher student loss for av part of model
    for key in ts_config.loss_subconfig.keys():
        train_disloss = importlib.import_module(
            ts_config.loss_function_name
        ).Loss(key, ts_config.temperature, ts_config.loss_subconfig[key], ts_config.alpha)
        loss_array['adapt_loss' + key] = train_disloss

    # teacher student loss for a and v part of model
    mse_loss = importlib.import_module(
        ts_config.mse_loss_function
    ).Loss(ts_config.mse_temperature)

    loss_array['mse_loss'] = mse_loss

    # teacher student kl outputs
    kl_loss = importlib.import_module(
        ts_config.kl_loss_function
    ).Loss(ts_config.kl_temperature)
    loss_array['kl_loss'] = kl_loss

    optimizer = optim.Adam([p for p in ts_model.student.parameters() if p.requires_grad], lr=model_config.lr,
                           weight_decay=model_config.weight_decay)

    if model_config.scheduler is not None:
        scheduler = StepLR(optimizer, step_size=model_config.scheduler_step_size, gamma=model_config.gamma)
    else:
        scheduler = None

    if ts_config.model_checkpoint_path is not None:
        model, last_epoch, best_auc = load_checkpoint(ts_model.model_checkpoint_path, ts_model, kind='student')
    else:
        last_epoch = 0
        best_auc = 0

    # train model
    current_num_epoch = 0
    print('start training')
    train(ts_model, train_data_loader, val_data_loader, loss_array, optimizer, scheduler, ts_config.checkpoint_dir,
          current_num_epoch, model_config.n_epochs, device, best_auc)
