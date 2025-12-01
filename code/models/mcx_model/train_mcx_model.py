import argparse
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils import data as data_utils
import numpy as np
from models.mcx_model.visual_models import API_Net
from dataset.mcx_model_dataset import TrainDataset
from sklearn.metrics import f1_score, roc_curve, auc, classification_report
from models.mcx_model.utils import accuracy, AverageMeter, save_checkpoint, my_collate
from torch.utils.tensorboard import SummaryWriter
from models.mcx_model.orthogonalprojectionloss import OrthogonalProjectionLoss
from pathlib import Path
from models.mcx_model.visual_model_config import model_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir=model_config.tensorboard_dir)
torch.multiprocessing.set_sharing_strategy('file_system')


def metric_calculation_epoch(f1_scores, auc_scores, labels, preds, running_loss, epoch, num_batches, mode='train'):
    report = classification_report(labels, preds, target_names=['fake', 'real'], output_dict=True)
    print(classification_report(labels, preds, target_names=['fake', 'real']))
    writer.add_scalars('f1_epoch', {mode: np.mean(f1_scores)}, epoch)
    writer.add_scalars('auc_epoch', {mode: np.nanmean(auc_scores)}, epoch)
    writer.add_scalars('loss_epoch', {mode: running_loss / num_batches}, epoch)
    writer.add_scalars('acc_epoch', {mode: report['accuracy']}, epoch)
    writer.add_scalars('precision_epoch', {mode: report['macro avg']['precision']}, epoch)
    writer.add_scalars('recall_epoch', {mode: report['macro avg']['recall']}, epoch)
    return np.nanmean(auc_scores)


def main():
    # create model
    # model = API_Net(num_classes=model_config.num_classes,
    #                 model_name=model_config.model_name,
    #                 weight_init=model_config.weight_init,
    #                 )
    # model = model.to(device)
    # model.conv = nn.DataParallel(model.conv)

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().to(device)
    # optimizer_conv = torch.optim.SGD(model.conv.parameters(), model_config.lr,
    #                                  momentum=model_config.momentum,
    #                                  weight_decay=model_config.weight_decay)
    #
    # fc_parameters = [value for name, value in model.named_parameters() if 'conv' not in name]
    # optimizer_fc = torch.optim.SGD(fc_parameters, model_config.lr,
    #                                momentum=model_config.momentum,
    #                                weight_decay=model_config.weight_decay)
    #
    # if model_config.resume:
    #     if os.path.isfile(model_config.resume):
    #         print('loading checkpoint {}'.format(model_config.resume))
    #         checkpoint = torch.load(model_config.resume)
    #         model_config.start_epoch = checkpoint['epoch']
    #         best_prec1 = checkpoint['best_prec1']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer_conv.load_state_dict(checkpoint['optimizer_conv'])
    #         optimizer_fc.load_state_dict(checkpoint['optimizer_fc'])
    #         print('loaded checkpoint {}(epoch {})'.format(model_config.resume, checkpoint['epoch']))
    #     else:
    #         print('no checkpoint found at {}'.format(model_config.resume))

    train_dataset = TrainDataset('train', model_config)
    val_dataset = TrainDataset('val', model_config)

    train_loader = data_utils.DataLoader(
        train_dataset, batch_size=model_config.batch_size, shuffle=True,
        num_workers=model_config.num_workers)

    val_loader = data_utils.DataLoader(
        val_dataset, batch_size=model_config.batch_size,
        num_workers=model_config.num_workers)

    test_pre_model(model_config, val_loader)


#     scheduler_conv = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_conv, 100 * len(train_loader))
#     scheduler_fc = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fc, 100 * len(train_loader))
#
#     # train model
#     step = 0
#     best_prec1 = 0
#
#     print('START TIME:', time.asctime(time.localtime(time.time())))
#     for epoch in range(model_config.start_epoch, model_config.epochs):
#         train(train_loader, model, criterion, optimizer_conv, scheduler_conv, optimizer_fc, scheduler_fc, epoch, step,
#               model_config.n_classes_total, writer, model_config.dist_type, model_config.image_loader)
#         prec1_val, loss_val = validate(val_loader, model, criterion, model_config.dist_type, model_config.image_loader)
#
#         writer.add_scalar('val_loss', loss_val, epoch)
#         writer.add_scalar('val_top1', prec1_val, epoch)
#
#         # remember best prec@1 and save checkpoint
#         if not os.path.exists(model_config.checkpoint_dir):
#             os.makedirs(model_config.checkpoint_dir)
#         is_best = prec1_val > best_prec1
#         best_prec1 = max(prec1_val, best_prec1)
#
#         save_checkpoint(save_path=model_config.checkpoint_dir, state={
#             'epoch': epoch + 1,
#             'state_dict': model.state_dict(),
#             'best_prec1': best_prec1,
#             'optimizer_conv': optimizer_conv.state_dict(),
#             'optimizer_fc': optimizer_fc.state_dict(),
#         }, is_best=is_best, saved_file=os.path.join(model_config.checkpoint_dir, 'model_best.pth.tar'))
#         # str(epoch) + '_' + model_name

def test_pre_model(model_config, val_loader):
    v_model = API_Net(num_classes=model_config.num_classes,
                      model_name=model_config.model_name,
                      weight_init=model_config.weight_init,
                      )
    v_model = v_model.to(device)
    v_model.conv = nn.DataParallel(v_model.conv)
    checkpoint = torch.load(model_config.pretrained_v_path)
    v_model.load_state_dict(checkpoint['state_dict'])

    preds = []
    labels = []
    for i, (input, target) in enumerate(val_loader):
        input_val = input.to(device)
        target_val = target.to(device).squeeze()

        # compute output
        logits_val, pool_out = v_model(input_val)

        labels += list(target_val.clone().detach().cpu().numpy())
        batch_size = target.size(0)
        _, ind = torch.topk(logits_val, 1)
        preds_labels = []
        for pred in ind.squeeze():
            if pred != 0:
                preds_labels.append(0)
            else:
                preds_labels.append(1)

        preds += preds_labels
        print(sum(preds_labels))
        # print(labels)

    report = classification_report(labels, preds, target_names=['fake', 'real'], output_dict=True)
    print(classification_report(labels, preds, target_names=['fake', 'real']))


def train(train_loader, model, criterion, optimizer_conv, scheduler_conv, optimizer_fc, scheduler_fc, epoch, step,
          n_classes_total, train_writer, dist_type, image_loader):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    softmax_losses = AverageMeter()
    rank_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to train mode
    end = time.time()
    rank_criterion = nn.MarginRankingLoss(margin=0.05)
    op_loss = OrthogonalProjectionLoss(gamma=0.5)
    op_lambda = 0.4
    softmax_layer = nn.Softmax(dim=1).to(device)

    for i, (input, target) in enumerate(train_loader):
        model.train()

        # measure data loading time
        data_time.update(time.time() - end)
        input_var = input.to(device)
        target_var = target.to(device).squeeze()
        # print(f'input size {input_var.shape}')
        # print(f'target size {target_var.shape}')

        # compute output
        logit1_self, logit1_other, logit2_self, logit2_other, labels1, labels2, features = model(input_var, target_var,
                                                                                                 flag='train',
                                                                                                 dist_type=dist_type,
                                                                                                 loader=image_loader)
        batch_size = logit1_self.shape[0]
        labels1 = labels1.to(device)
        labels2 = labels2.to(device)

        self_logits = torch.zeros(2 * batch_size, n_classes_total).to(device)
        other_logits = torch.zeros(2 * batch_size, n_classes_total).to(device)
        self_logits[:batch_size] = logit1_self
        self_logits[batch_size:] = logit2_self
        other_logits[:batch_size] = logit1_other
        other_logits[batch_size:] = logit2_other
        # print(f'logit1_self {logit1_self}, logit2_self {logit2_self}')
        # print(f'labels1 {labels1}, labels2 {labels2}')

        # compute loss
        logits = torch.cat([self_logits, other_logits], dim=0)
        targets = torch.cat([labels1, labels2, labels1, labels2], dim=0)
        # print(f'train logits, targets: {logits}, {targets}')
        softmax_loss = criterion(logits, targets)

        # margin rank loss
        self_scores = softmax_layer(self_logits)[torch.arange(2 * batch_size).to(device).long(),
        torch.cat([labels1, labels2], dim=0)]
        other_scores = softmax_layer(other_logits)[torch.arange(2 * batch_size).to(device).long(),
        torch.cat([labels1, labels2], dim=0)]
        flag = torch.ones([2 * batch_size, ]).to(device)
        rank_loss = rank_criterion(self_scores, other_scores, flag)

        # orthogonal projection loss
        loss_op = op_loss(features, target_var)

        loss = softmax_loss + rank_loss  # + op_lambda * loss_op

        # measure accuracy and record loss
        prec1 = accuracy(logits, targets, 1)
        # prec5 = accuracy(logits, targets, 5)
        losses.update(loss.item(), 2 * batch_size)
        softmax_losses.update(softmax_loss.item(), 4 * batch_size)
        rank_losses.update(rank_loss.item(), 2 * batch_size)
        top1.update(prec1, 4 * batch_size)
        # top5.update(prec5, 4*batch_size)

        # compute gradient and do SGD step
        optimizer_conv.zero_grad()
        optimizer_fc.zero_grad()
        loss.backward()

        # if epoch >= 8:
        optimizer_conv.step()
        scheduler_conv.step()
        optimizer_fc.step()
        scheduler_fc.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Train results: \t Time: {time}\nStep: {step}\t Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'SoftmaxLoss {softmax_loss.val:.4f} ({softmax_loss.avg:.4f})\t'
                  'RankLoss {rank_loss.val:.4f} ({rank_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, softmax_loss=softmax_losses, rank_loss=rank_losses,
                top1=top1, step=step, time=time.asctime(time.localtime(time.time()))))

        # write in tensorboard
        train_writer.add_scalar('train_loss', losses.avg, epoch)
        train_writer.add_scalar('train_top1', top1.avg, epoch)
        train_writer.add_scalar('learning_rate_conv', optimizer_conv.param_groups[0]['lr'], epoch)
        train_writer.add_scalar('learning_rate_fc', optimizer_fc.param_groups[0]['lr'], epoch)

    return top1.avg, softmax_losses.avg


def validate(val_loader, model, criterion, dist_type, image_loader):
    batch_time = AverageMeter()
    softmax_losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            input_val = input.to(device)
            target_val = target.to(device).squeeze()

            # compute output
            logits_val = model(input_val, targets=None, flag='val', dist_type=dist_type, loader=image_loader)
            # print(f'train logits, targets_val: {logits_val}, {target_val}')

            if target_val.dim() != 0:
                # batch size cannot be 1
                softmax_loss = criterion(logits_val, target_val)
                # print('logits_val :', logits_val.shape)
                # print('target_val :', target_val.shape)
                prec1 = accuracy(logits_val, target_val, 1)
                # prec5 = accuracy(logits, target_var, 5)
                softmax_losses.update(softmax_loss.item(), logits_val.size(0))
                top1.update(prec1, logits_val.size(0))
                # top5.update(prec5, logits.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 10 == 0:
                    print('Validation results: \t Time: {time}\nTest: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'SoftmaxLoss {softmax_loss.val:.4f} ({softmax_loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, softmax_loss=softmax_losses,
                        top1=top1, time=time.asctime(time.localtime(time.time()))))
        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, softmax_losses.avg


if __name__ == '__main__':
    main()
