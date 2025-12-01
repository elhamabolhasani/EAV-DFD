import torch
import matplotlib.pyplot as plt
import pickle
from torch import nn
from torchaudio.transforms import MelScale
import numpy as np
import os
from sklearn.metrics import f1_score, roc_curve, auc, classification_report


def load_checkpoint(checkpoint_path, model, kind='main'):
    print("Load checkpoint from: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    if kind == 'student':
        model.load_state_dict(checkpoint["student_state_dict"])
    else:
        model.load_state_dict(checkpoint["state_dict"])

    last_training_epoch = checkpoint["n_epoch"]
    best_val_auc = checkpoint["best_auc"]
    return model, last_training_epoch, best_val_auc

def save_checkpoint(model, optimizer, scheduler, checkpoint_dir, epoch, best_auc, is_best, kind='main'):
    if is_best:
        checkpoint_path = os.path.join(
            checkpoint_dir, "best_checkpoint.pth")
    else:
        checkpoint_path = os.path.join(
            checkpoint_dir, "checkpoint_step{:09d}.pth".format(epoch))

    optimizer_state = optimizer.state_dict()
    if kind == 'main':
        if scheduler is not None:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "n_epoch": epoch,
                    "best_auc": best_auc
                }, checkpoint_path)
        else:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "n_epoch": epoch,
                    "best_auc": best_auc
                }, checkpoint_path)
    else:
        if scheduler is not None:
            torch.save(
                {
                    "student_state_dict": model.get_student_weights(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "n_epoch": epoch,
                    "best_auc": best_auc
                }, checkpoint_path)
        else:
            torch.save(
                {
                    "student_state_dict": model.get_student_weights(),
                    "optimizer": optimizer.state_dict(),
                    "n_epoch": epoch,
                    "best_auc": best_auc
                }, checkpoint_path)

    print("Saved checkpoint:", checkpoint_path)

def decision_making(v_outputs, av_outputs, a_outputs, model_config, n):
    if model_config.classifier_type == 'joint' or model_config.classifier_type == 'attention':
        av_labels = (av_outputs > 0).float().clone().detach().cpu().numpy()
        sigmoid_layer = nn.Sigmoid()
        final_probs = sigmoid_layer(av_outputs).unsqueeze(1)
        final_labels = (final_probs > 0.5).float().clone().detach().cpu().numpy()
        return final_labels, final_probs, [[0]]*len(av_labels), av_labels, [[0]]*len(av_labels), final_probs, final_probs, final_probs

    if n == 1:
        v_outputs = v_outputs.unsqueeze(0)

    # apply sigmoid on model outputs
    v_labels = (v_outputs > 0).float().clone().detach().cpu().numpy()
    av_labels = (av_outputs > 0).float().clone().detach().cpu().numpy()
    a_labels = (a_outputs > 0).float().clone().detach().cpu().numpy()

    # find final probs
    sigmoid_layer = nn.Sigmoid()
    v_probs = sigmoid_layer(v_outputs)
    av_probs = sigmoid_layer(av_outputs).unsqueeze(1)
    a_probs = sigmoid_layer(a_outputs)
    probs = torch.cat((a_probs, av_probs, v_probs), 1)

    final_probs = torch.mean(probs, 1)
    final_labels = (final_probs > 0.5).float().clone().detach().cpu().numpy()

    if model_config.label_kind == 'and':
        final_labels = [[(v_labels[i][0] and av_labels[i] and a_labels[i][0])] for i in range(n)]
        for i in range(n):
            # fake
            if final_labels[i][0] == 0:
                final_probs[i] = ((1 - v_labels[i][0])*(v_probs[i]) + (1 - av_labels[i])*(av_probs[i]) +
                                  (1 - a_labels[i][0])*(a_probs[i])) / (
                                             3 - sum([v_labels[i][0], av_labels[i], a_labels[i][0]]))

            # real : mean

    if model_config.label_kind == 'majority':
        final_labels = [[((v_labels[i][0] and av_labels[i]) or
                         (a_labels[i][0] and av_labels[i])) or
                        (v_labels[i][0] and a_labels[i])] for i in range(n)]

        for i in range(n):
            # fake
            if final_labels[i][0] == 0:
                final_probs[i] = ((1 - v_labels[i][0])*(v_probs[i]) + (1 - av_labels[i])*(av_probs[i]) +
                                  (1 - a_labels[i][0])*(a_probs[i])) / (
                                         3 - sum([v_labels[i][0], av_labels[i], a_labels[i][0]]))
            # real
            if final_labels[i][0] == 1:
                final_probs[i] = ((v_labels[i][0])(v_probs[i]) + (av_labels[i])(av_probs[i]) +
                                  (a_labels[i][0])(a_probs[i])) / (sum([v_labels[i][0], av_labels[i], a_labels[i][0]]))


    return final_labels, final_probs.unsqueeze(1), v_labels, av_labels, a_labels, a_probs, v_probs, av_probs


def prepare_model_input(aud_av, vid_av, aud_a, vid_v, device, model_config, kind='train', noise_type=None):

    if noise_type is not None:
        
        # Apply noise based on noise_type
        if noise_type == 'gaussian_video_audio':
            # Add Gaussian noise to video frames
            noise = torch.randn_like(vid_av) * 0.05
            vid_av = torch.clamp(vid_av + noise, 0, 1)
            noise = torch.randn_like(vid_v) * 0.05
            vid_v = torch.clamp(vid_v + noise, 0, 1)
            
            # Add Gaussian noise to audio
            noise = torch.randn_like(aud_av) * 0.05
            aud_av = aud_av + noise
            noise = torch.randn_like(aud_a) * 0.05
            aud_a = aud_a + noise
            
        elif noise_type == 'salt_pepper_impulse_audio':
            # Add salt and pepper noise to video frames
            prob = 0.05
            mask = torch.rand_like(vid_av) < prob
            salt_mask = mask & (torch.rand_like(mask) < 0.5)
            pepper_mask = mask & (torch.rand_like(mask) >= 0.5)
            vid_av[salt_mask] = 1.0
            vid_av[pepper_mask] = 0.0
            
            mask = torch.rand_like(vid_v) < prob
            salt_mask = mask & (torch.rand_like(mask) < 0.5)
            pepper_mask = mask & (torch.rand_like(mask) >= 0.5)
            vid_v[salt_mask] = 1.0
            vid_v[pepper_mask] = 0.0
            
            # Add impulse noise to audio
            prob = 0.01
            mask = torch.rand_like(aud_av) < prob
            aud_av[mask] = torch.rand_like(aud_av[mask]) * 2 - 1
            mask = torch.rand_like(aud_a) < prob
            aud_a[mask] = torch.rand_like(aud_a[mask]) * 2 - 1
            
        elif noise_type == 'crop_video':
            # Crop video frames (simulate partial occlusion or cropping)
            crop_ratio = noise_params.get('crop_ratio', 0.2)  # Crop 20% from each side
            batch_size, num_frames, channels, height, width = vid_av.shape
            
            # Calculate crop dimensions
            crop_h = int(height * crop_ratio)
            crop_w = int(width * crop_ratio)
            
            # Crop from center
            start_h = crop_h
            end_h = height - crop_h
            start_w = crop_w
            end_w = width - crop_w
            
            # Apply cropping
            vid_av = vid_av[:, :, :, start_h:end_h, start_w:end_w]
            vid_v = vid_v[:, :, :, start_h:end_h, start_w:end_w]
            
            # Resize back to original size using interpolation
            vid_av = torch.nn.functional.interpolate(
                vid_av.view(-1, channels, vid_av.shape[-2], vid_av.shape[-1]),
                size=(height, width), mode='bilinear', align_corners=False
            ).view(batch_size, num_frames, channels, height, width)
            
            vid_v = torch.nn.functional.interpolate(
                vid_v.view(-1, channels, vid_v.shape[-2], vid_v.shape[-1]),
                size=(height, width), mode='bilinear', align_corners=False
            ).view(batch_size, num_frames, channels, height, width)
            
            # Crop audio (simulate audio clipping or partial audio)
            
        elif noise_type == 'compression_video':
            # Simulate video compression artifacts
            import cv2
            quality = 50
            print(vid_av.shape)
            batch_size, num_frames, channels, height, width = vid_av.shape
            for b in range(batch_size):
                for f in range(num_frames):
                    frame = vid_av[b, f].cpu().numpy().transpose(1, 2, 0)
                    frame_uint8 = (frame * 255).astype(np.uint8)
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                    _, encoded_img = cv2.imencode('.jpg', frame_uint8, encode_param)
                    decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
                    compressed_frame = decoded_img.astype(np.float32) / 255.0
                    vid_av[b, f] = torch.from_numpy(compressed_frame.transpose(2, 0, 1))
            
            for b in range(batch_size):
                for f in range(num_frames):
                    frame = vid_v[b, f].cpu().numpy().transpose(1, 2, 0)
                    frame_uint8 = (frame * 255).astype(np.uint8)
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                    _, encoded_img = cv2.imencode('.jpg', frame_uint8, encode_param)
                    decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
                    compressed_frame = decoded_img.astype(np.float32) / 255.0
                    vid_v[b, f] = torch.from_numpy(compressed_frame.transpose(2, 0, 1))
                    
        elif noise_type == 'combined':
            # Combined noise: gaussian on video and impulse on audio
            video_std = noise_params.get('video_std', 0.1)
            audio_prob = noise_params.get('audio_prob', 0.01)
            
            # Video noise
            noise = torch.randn_like(vid_av) * 0.1
            vid_av = torch.clamp(vid_av + noise, 0, 1)
            noise = torch.randn_like(vid_v) * 0.1
            vid_v = torch.clamp(vid_v + noise, 0, 1)
            
            # Audio noise
            prob = 0.01
            mask = torch.rand_like(aud_av) < prob
            aud_av[mask] = torch.rand_like(aud_av[mask]) * 2 - 1
            mask = torch.rand_like(aud_a) < prob
            aud_a[mask] = torch.rand_like(aud_a[mask]) * 2 - 1
            
        elif noise_type == 'combined_crop':
            # Combined crop noise: crop both video and audio
            video_crop = noise_params.get('video_crop_ratio', 0.15)
            audio_crop = noise_params.get('audio_crop_ratio', 0.15)
            
            # Video cropping
            batch_size, num_frames, channels, height, width = vid_av.shape
            crop_h = int(height * video_crop)
            crop_w = int(width * video_crop)
            start_h = crop_h
            end_h = height - crop_h
            start_w = crop_w
            end_w = width - crop_w
            
            vid_av = vid_av[:, :, :, start_h:end_h, start_w:end_w]
            vid_v = vid_v[:, :, :, start_h:end_h, start_w:end_w]
            
            vid_av = torch.nn.functional.interpolate(
                vid_av.view(-1, channels, vid_av.shape[-2], vid_av.shape[-1]),
                size=(height, width), mode='bilinear', align_corners=False
            ).view(batch_size, num_frames, channels, height, width)
            
            vid_v = torch.nn.functional.interpolate(
                vid_v.view(-1, channels, vid_v.shape[-2], vid_v.shape[-1]),
                size=(height, width), mode='bilinear', align_corners=False
            ).view(batch_size, num_frames, channels, height, width)
            
            # Audio cropping
            audio_length = aud_av.shape[-1]
            crop_samples = int(audio_length * audio_crop)
            start_sample = crop_samples
            end_sample = audio_length - crop_samples
            
            aud_av = aud_av[..., start_sample:end_sample]
            aud_a = aud_a[..., start_sample:end_sample]
            
            pad_left = crop_samples
            pad_right = crop_samples
            
            aud_av = torch.nn.functional.pad(aud_av, (pad_left, pad_right), mode='constant', value=0)
            aud_a = torch.nn.functional.pad(aud_a, (pad_left, pad_right), mode='constant', value=0)

    # make av features
    top_db = -model_config.min_level_db
    min_level = np.exp(top_db / -20 * np.log(10))
    melscale = MelScale(n_mels=model_config.num_mels, sample_rate=model_config.sample_rate, f_min=model_config.fmin,
                        f_max=model_config.fmax,
                        n_stft=model_config.n_stft, norm='slaney', mel_scale='slaney').to(device)

    if kind == 'test':
        gt_aud = aud_av.squeeze(0).to(device)
    else:
        gt_aud = aud_av.to(device)

    spec = torch.stft(gt_aud, n_fft=model_config.n_fft, hop_length=model_config.hop_size,
                      win_length=model_config.win_size,
                      window=torch.hann_window(model_config.win_size).to(gt_aud.device), return_complex=True)


    melspec = melscale(torch.abs(spec.detach().clone()).float())
    melspec_tr1 = (20 * torch.log10(torch.clamp(melspec, min=min_level))) - model_config.ref_level_db

    # normalized mel
    normalized_mel = torch.clip(
        (2 * model_config.max_abs_value) * ((melspec_tr1 + top_db) / top_db) - model_config.max_abs_value,
        -model_config.max_abs_value, model_config.max_abs_value)
    mels = normalized_mel[:, :, :-1].unsqueeze(1)

    if kind == 'train':
        vid_av = vid_av.to(device)
        aud_a = aud_a.to(device)
        vid_v = vid_v.to(device)
    else:
        vid_av = vid_av.squeeze(0).to(device)
        aud_a = aud_a.squeeze(0).to(device)
        vid_v = vid_v.squeeze(0).to(device)

    return vid_av, aud_a, vid_v, mels

def calc_f1_and_auc(labels, probs, preds):
    f1_metric = f1_score(labels,
                         preds,
                         average="weighted")

    fpr, tpr, thresholds = roc_curve(labels,
                                     probs,
                                     pos_label=1)

    auc_metric = auc(fpr, tpr)
    return f1_metric, auc_metric

def print_metrics(labels, probs, preds, kind):
    print('report ', kind)
    report = classification_report(labels, preds, target_names=['fake', 'real'], output_dict=True)
    f1_metric, auc_metric = calc_f1_and_auc(labels, probs, preds)

    print({
        'hparam/f1': f1_metric, 'hparam/auc': auc_metric, 'hparam/recall': report['macro avg']['recall'],
        'hparam/acc': report['accuracy'], 'hparam/precision': report['macro avg']['precision']
    }
    )

def logging_epoch_metrics_train(f1_scores, auc_scores, labels, preds, labels_v, preds_v, labels_a, preds_a,
                              labels_av, preds_av, running_loss, epoch, num_batches, total_vid_ture_label,
                              total_vid_ture_pred, writer,  mode='train'):

    
    # Calculate ensemble report
    try:
        report = classification_report(labels, preds, target_names=['fake', 'real'], zero_division=0, output_dict=True)
        print('ensemble_report')
        print(classification_report(labels, preds, target_names=['fake', 'real']))
    except Exception as e:
        print(f"Error in ensemble classification report: {e}")
        report = {'accuracy': 0.0, 'macro avg': {'precision': 0.0, 'recall': 0.0}}
    
    # Calculate video report
    try:
        report_v = classification_report(labels_v, preds_v, target_names=['fake', 'real'], zero_division=0, output_dict=True)
        print('video_report')
        print(classification_report(labels_v, preds_v, target_names=['fake', 'real']))
    except Exception as e:
        print(f"Error in video classification report: {e}")
        report_v = {'accuracy': 0.0}
    
    # Calculate audio report
    try:
        report_a = classification_report(labels_a, preds_a, target_names=['fake', 'real'], zero_division=0, output_dict=True)
        print('audio_report')
        print(classification_report(labels_a, preds_a, target_names=['fake', 'real']))
    except Exception as e:
        print(f"Error in audio classification report: {e}")
        report_a = {'accuracy': 0.0}
    
    # Calculate audio-visual report
    try:
        report_av = classification_report(labels_av, preds_av, target_names=['fake', 'real'], zero_division=0, output_dict=True)
        print('audio_visual_report')
        print(classification_report(labels_av, preds_av, target_names=['fake', 'real']))
    except Exception as e:
        print(f"Error in audio-visual classification report: {e}")
        report_av = {'accuracy': 0.0}
    
    writer.add_scalars('f1_epoch', {mode: np.mean(f1_scores)}, epoch)
    writer.add_scalars('auc_epoch', {mode: np.nanmean(auc_scores)}, epoch)
    writer.add_scalars('loss_epoch', {mode: running_loss / num_batches}, epoch)
    writer.add_scalars('acc_epoch', {mode: report['accuracy']}, epoch)
    writer.add_scalars('precision_epoch', {mode: report['macro avg']['precision']}, epoch)
    writer.add_scalars('recall_epoch', {mode: report['macro avg']['recall']}, epoch)
    writer.add_scalars('vid_label_true', {mode: total_vid_ture_label}, epoch)
    writer.add_scalars('vid_label_pred', {mode: total_vid_ture_pred}, epoch)
    writer.add_scalars('v_acc', {'train': report_v['accuracy']}, epoch)
    writer.add_scalars('a_acc', {'train': report_a['accuracy']}, epoch)
    writer.add_scalars('av_acc', {'train': report_av['accuracy']}, epoch)
    return np.nanmean(auc_scores)

def logging_epoch_metrics_val(f1_scores, auc_scores, labels, preds, running_loss, epoch, num_batches, total_vid_ture_label,
                              total_vid_ture_pred, writer,  mode='val'):

    try:
        report = classification_report(labels, preds, target_names=['fake', 'real'], output_dict=True)
        print('ensemble_report')
        print(classification_report(labels, preds, target_names=['fake', 'real']))
    except Exception as e:
        print(f"Error in ensemble classification report: {e}")
        report = {'accuracy': 0.0, 'macro avg': {'precision': 0.0, 'recall': 0.0}}

    writer.add_scalars('f1_epoch', {mode: np.mean(f1_scores)}, epoch)
    writer.add_scalars('auc_epoch', {mode: np.nanmean(auc_scores)}, epoch)
    writer.add_scalars('loss_epoch', {mode: running_loss / num_batches}, epoch)
    writer.add_scalars('acc_epoch', {mode: report['accuracy']}, epoch)
    writer.add_scalars('precision_epoch', {mode: report['macro avg']['precision']}, epoch)
    writer.add_scalars('recall_epoch', {mode: report['macro avg']['recall']}, epoch)
    writer.add_scalars('vid_label_true', {mode: total_vid_ture_label}, epoch)
    writer.add_scalars('vid_label_pred', {mode: total_vid_ture_pred}, epoch)

    return np.nanmean(auc_scores)

def find_final_clips_label(final_labels, real_ratio):
    if np.sum(final_labels) >= (len(final_labels) * real_ratio):
        # real
        return 1
    else:
        # fake
        return 0

def find_prob_based_on_label(label, probs, labels, kind='mean_clip_prob'):
    # print('************')
    # print(label)
    # print(probs)
    # print(labels)

    num_clip = len(labels)

    if kind == 'min_max_clip_prob':
        if label == 0:
            negative_ones = [val for i, val in enumerate(probs) if labels[i] == 0]
            assert min(negative_ones) < 0.5
            return min(negative_ones)
        else:
            positive_ones = [val for i, val in enumerate(probs) if labels[i] == 1]
            assert max(positive_ones) > 0.5
            return max(positive_ones)

    elif kind == 'mean_clip_prob':
        return torch.mean(probs)
    elif kind == 'min_clip_prob':
        return torch.min(probs)

def find_final_clips_prob_joint(final_probs, a_probs_clip, v_probs_clip, av_probs_clip, v_labels, av_labels, a_labels,
                          final_labels, real_ratio):

    num_clip = len(final_probs)
    final_est_label = [final_labels[i][0] for i in range(num_clip)]
    kind = 'mean_clip_prob'
    final_prob = find_prob_based_on_label(final_label, final_probs, final_est_label, kind)
    final_label = int(final_prob > 0.5)

    return final_label, final_prob.item(), final_prob.item(), final_prob.item(), final_prob.item(), final_label, final_label, final_label

def find_final_clips_prob(final_probs, a_probs_clip, v_probs_clip, av_probs_clip, v_labels, av_labels, a_labels,
                          final_labels, real_ratio):

    num_clip = len(final_probs)
    v_est_label = [v_labels[i][0] for i in range(num_clip)]
    a_est_label = [a_labels[i][0] for i in range(num_clip)]
    av_est_label = [av_labels[i] for i in range(num_clip)]
    final_est_label = [final_labels[i][0] for i in range(num_clip)]

    v_pred = find_final_clips_label(v_est_label, real_ratio)
    a_pred = find_final_clips_label(a_est_label, real_ratio)
    av_pred = find_final_clips_label(av_est_label, real_ratio)

    kind = 'mean_clip_prob'
    final_prob = find_prob_based_on_label(final_label, final_probs, final_est_label, kind)
    v_prob = find_prob_based_on_label(v_pred, v_probs_clip, v_est_label, kind)
    av_prob = find_prob_based_on_label(av_pred, av_probs_clip, av_est_label, kind)
    a_prob = find_prob_based_on_label(a_pred, a_probs_clip, a_est_label, kind)
    final_label = int(final_prob > 0.5)

    return final_label, final_prob.item(), v_prob.item(), a_prob.item(), av_prob.item(), v_pred, a_pred, av_pred




