from tqdm import tqdm
import torch
from torch.utils import data as data_utils
import numpy as np
from torchaudio.transforms import MelScale
from sklearn.metrics import f1_score, roc_curve, auc, classification_report
import os, random
import torch.multiprocessing
from torch.utils.tensorboard import SummaryWriter
from dataset.test_dataset import TestDataset
from models.utils import *
import shutil
import time
from models.ensemble_model.ensemble_model import MiniEavNet
import pickle
import numpy as np

random.seed(4321)

torch.multiprocessing.set_sharing_strategy('file_system')
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))




def save_embeddings(embeddings, save_path):
    """Save embeddings to a pickle file."""
    with open(save_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings saved to {save_path}")


def eval_model(test_data_loader, model_config, device, model, params, writer, epoch=0,
               in_test=True, store_embeddings=False, embedding_save_path=None):

    preds = []
    probs = []
    labels = []
    v_target_labels, v_preds, v_probs = [], [], []
    a_target_labels, a_preds, a_probs = [], [], []
    av_target_labels, av_preds, av_probs = [], [], []
    est_label_count = []
    prog_bar = tqdm(enumerate(test_data_loader))

    check_result_dict = {}
    for i in range(model_config.num_clip_check + 1):
        check_result_dict[str(i)] = 0

    v_labels_dist = []
    a_labels_dist = []

    # store embeddings if requested
    if store_embeddings:
        stored_embeddings = {
            'a_hidden': [],
            'v_hidden': [],
            'av_hidden': [],
            'labels': [],
            'vidnames': []
        }

    # model.tsne_features = []
    # model.tsne_labels_v = []
    # model.tsne_labels_a = []
    # model.tsne_labels_av = []

    for step, (vid_v, vid_av, aud_av, aud_a, y_av, y_v, y_a, vidname) in prog_bar:
        print('step : ', step)
        print('vidname : ', vidname)
        model.eval()
        with torch.no_grad():
            vid_av, aud_a, vid_v, mels = (
                prepare_model_input(aud_av, vid_av, aud_a, vid_v, device, model_config, kind='test', noise_type=None))

            # eval model
            num_clip = min(model_config.num_clip_check, int(vid_av.shape[0]))
            v_outputs, av_outputs, a_outputs, v_hidden, feature_list, a_hidden, av_hidden = model(vid_v, vid_av, mels, aud_a)
        
            # store embeddings if requested
            if store_embeddings:
                stored_embeddings['a_hidden'].append(a_hidden.detach().cpu().numpy())
                stored_embeddings['v_hidden'].append(v_hidden.detach().cpu().numpy())
                stored_embeddings['av_hidden'].append(av_hidden.detach().cpu().numpy())
                stored_embeddings['labels'].append(y_av.squeeze(0)[0].item())
                stored_embeddings['vidnames'].append(vidname[0])

            final_labels, final_probs, v_labels, av_labels, a_labels, a_probs_clip, v_probs_clip, av_probs_clip = (
                decision_making(v_outputs, av_outputs, a_outputs, model_config, num_clip))

            if num_clip == 1:
                v_labels = np.expand_dims(v_labels, axis=0)

            v_labels_dist.append(np.sum([v_labels[i][0] for i in range(num_clip)]))
            a_labels_dist.append(np.sum([a_labels[i][0] for i in range(num_clip)]))

            final_label, final_prob, v_prob, a_prob, av_prob, v_pred, a_pred, av_pred = find_final_clips_prob(
                final_probs, a_probs_clip, v_probs_clip, av_probs_clip, v_labels, av_labels, a_labels, final_labels,
                model_config.real_ratio)

            preds.append(final_label)
            probs.append(final_prob)
            labels.append(y_av.squeeze(0)[0].item())  # all clips have the same label


            if model_config.test_dataset_name == 'fakeavceleb' or model_config.test_dataset_name == 'test_polyglotfake' :
                v_preds.append(v_pred)
                v_probs.append(v_prob)

                a_preds.append(a_pred)
                a_probs.append(a_prob)

                av_preds.append(av_pred)
                av_probs.append(av_prob)

                v_target_labels.append(y_v[0][0].item())
                a_target_labels.append(y_a[0][0].item())
                av_target_labels.append(y_av[0][0].item())

    
    
    try:
        report = classification_report(labels, preds, target_names=['fake', 'real'], output_dict=True)
        print(classification_report(labels, preds, target_names=['fake', 'real']))
    except Exception as e:
        print(f"Error in ensemble classification report: {e}")
        report = {'accuracy': 0.0, 'macro avg': {'precision': 0.0, 'recall': 0.0}}

    f1_metric, auc_metric = calc_f1_and_auc(labels, probs, preds)

    mode = 'val_timit'

    writer.add_scalars('cross_f1_epoch', {mode: f1_metric}, epoch)
    writer.add_scalars('cross_auc_epoch', {mode: auc_metric}, epoch)
    writer.add_scalars('cross_acc_epoch', {mode: report['accuracy']}, epoch)
    writer.add_scalars('cross_precision_epoch', {mode: report['macro avg']['precision']}, epoch)
    writer.add_scalars('cross_recall_epoch', {mode: report['macro avg']['recall']}, epoch)

    return auc_metric


if __name__ == "__main__":
    """"   check the following params before testing   """

    # ............................. set checkpoint path ..............................
    model_root_path = 'models/teacher_model_fakeavceleb/'
    # model_root_path = 'models/student_model/'

    run_name = (
        'mini_b24_v20_400ne_mean_clip_prob_lr0.0001_optimizer(adam0)_scheduler(step_lr0.8-15)_and_(aug_4)_w3.0_1_c_loss_0.005/'  
        # +
        # 'MM_b24_v20_82ne_mcx_lr0.0001_optimizer(adam0)_scheduler(step_lr0.8-15)_(joint)_(aug)_stu_mode_ce(0.25)_(5)_jl_mse_dav'  
    )
    model_name = 'checkpoint_step000000100.pth'  
    checkpoint_path = os.path.join(model_root_path, run_name, model_name)
    kind = 'main'  # main , student

    from teacher_model_tensorboard.mini_model_config import model_config
    # from student_model_tensorboard.model_config import model_config
    
    # ..........................  import dataset that you want to test.....................................
    train_dataset_name = 'fakeavceleb'
    test_dataset_name = 'fakeavceleb'  
    # .............................. import test params ..................................................
    num_clip_check = 7
    model_config.num_clip_check = num_clip_check
    model_config.real_ratio = 0.5
    # .......................................................................................

    test_tensorboard_dir = os.path.join(model_config.tensorboard_path, 'test_results')
    writer = SummaryWriter(log_dir=test_tensorboard_dir)


    from utils.preprocess_dataset.fakeavceleb_config import config
    if test_dataset_name == 'fakeavceleb':
        model_config.test_data_root = config.ensemble_dataset_test
    elif test_dataset_name == 'vid_df_timit':
        model_config.test_data_root = cross_data_config.preprocess_dataset
    elif test_dataset_name == 'dfdc':
        model_config.test_data_root = dfdc_config.preprocess_dataset

    model_config.test_dataset_name = test_dataset_name

    # dataset and dataloader setup
    test_dataset = TestDataset('test', model_config)
    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=1,
        num_workers=0)

    device = torch.device(model_config.cuda if torch.cuda.is_available() else "cpu")

    # model
    model = MiniEavNet(model_config)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters())))
    print('total audio trainable params {}'.format(sum(p.numel() for p in model.a_model.parameters())))
    print('total audio trainable params {}'.format(sum(p.numel() for p in model.a_classifier.parameters())))
    print('total video trainable params {}'.format(sum(p.numel() for p in model.v_model.parameters())))
    print('total video trainable params {}'.format(sum(p.numel() for p in model.v_classifier.parameters())))
    print('total audio-visual trainable params {}'.format(sum(p.numel() for p in model.av_model.parameters())))
    model = model.to(device)
    model, last_epoch, best_auc = load_checkpoint(checkpoint_path, model, kind=kind)
    model = model.to(device)
    print('last_epoch: ', last_epoch)

    # parameters
    params = {
        'train_dataset': train_dataset_name,
        'test_dataset': test_dataset_name,
        'run_name': run_name,
        'model_name': model_name,
        'n_epochs': model_config.n_epochs,
        'learning_rate': model_config.lr,
        'batch_size': model_config.batch_size,
        'v_context': model_config.v_context,
        'last_epoch': last_epoch,
        'total_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'num_clip_check': num_clip_check,
        'real_ratio': model_config.real_ratio,
        'test_strategy': model_config.test_strategy,
        'kind': kind
    }

    with torch.no_grad():
        start_time = time.time()
        store_embeddings_flag = False  
        embedding_save_path = f'embeddings_run_student.pkl'
        
        auc_metric = eval_model(test_data_loader, model_config, device, model, params, writer, 
                               store_embeddings=store_embeddings_flag, embedding_save_path=embedding_save_path)
        print("--- %s seconds ---" % (time.time() - start_time))
