import os
from models.utils import config
import torchvision.transforms as transforms


class Config:
    #  model dirs
    config_file_path = '/models/ensemble_model/model_config.py'
    config_file_name = 'model_config.py'
    models_root = '/deepfake_models/'
    checkpoint_path = models_root + 'mini_ensemble/'
    tensorboard_path = '/ensemble_model_tensorboard/'
    model_checkpoint_path = None

    # data parameters
    train_data_root = config.ensemble_dataset_train
    val_data_root = config.ensemble_dataset_val
    test_data_root = config.ensemble_dataset_test
    val_dfdc_data_root = config.dfdc_val
    num_workers = 18

    # audio model hyperparameters
    audio_model_checkpoint_path = models_root + 'audio_model/hubert-base-ls960/distill_hubret'
    audio_feature_extractor_checkpoint_path = models_root + 'audio_model/hubert-base-ls960/feature_extractor'
    audio_pretrained_distill_model_path = models_root + 'audio_model/distill_audio_model_v10.pt'

    # visual model hyperparameters
    visual_model_type = 'mcx'  # 'vivit'
    if visual_model_type == 'vivit':
        v_img_size = 224
        image_patch_size = 16  # image patch size
        frame_patch_size = 5  # frame patch size
        num_classes = 2
        dim = 512
        spatial_depth = 5  # depth of the spatial transformer
        temporal_depth = 5  # depth of the temporal transformer
        heads = 4
        mlp_dim = 1024
        channels = 3
        dropout = 0.2
        emb_dropout = 0.2
    elif visual_model_type == 'mcx':
        v_img_size = 224
        selected_idx = 2
        pretrained_v_path = '/weights/mcx_rgb_model/model_mcx_api_rgb.tar'
        dropout = 0
        num_classes = 5  # 0 means real and others are fake
        # dist_type = 'euclidean'
        model_name = 'xception'
        weight_init = 'pretrained'
        transform_picked = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([512, 512]),
            transforms.RandomCrop([448, 448]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )])


    # audio-visual model hyperparameters
    num_mels = 80  # Number of mel-spectrogram channels and local conditioning dimensionality
    min_level_db = -100
    ref_level_db = 20
    sample_rate = 16000
    max_abs_value = 4.
    fmin = 55
    # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To
    # test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    fmax = 7600  # To be increased/reduced depending on data.
    n_stft = 401
    n_fft = 800  # Extra window size is filled with 0 paddings to match this parameter
    hop_size = 200  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
    win_size = 800  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    v_context = 30  # 10  # 5
    num_audio_elements = v_context * 640  # 6400  # 16000/25 * syncnet_T
    tot_num_frames = 25  # buffer
    pretrained_av_path = '/weights/vocalist_lsr2/vocalist_5f_lrs2.pth'

    # training hyperparameters
    add_joint_loss = True
    contrastive_loss = True
    augmentation = True
    fake_base_label = True
    freeze = False
    classifier_type = 'split'  # 'split', 'joint'
    batch_size = 16  # 128
    optimizer = 'adam'  # 'adam', 'adamw'
    weight_decay = 0  # set to 0 for off
    # momentum = # todo
    scheduler = 'step_lr'  # 'step_lr'
    gamma = 0.7
    scheduler_step_size = 15
    loss_fn = 'bce_with_logits_weighted'  # bce_with_logits_weighted
    loss_weight = 3.0
    var_v = 0.2
    var_av = 0.6
    var_joint = 0.75
    var_a = 0.2
    device_id = 0
    cuda = 'cuda:0'
    lr = 0.0001
    n_epochs = 403


    save_model_interval = 20

    if classifier_type == 'joint':
        run_name = "b{}_v{}_{}ne_{}_lr{}_optimizer({})_scheduler({})_(joint)".format(str(batch_size),
                                                                                                 str(v_context)
                                                                                                 , str(n_epochs),
                                                                                                 visual_model_type,
                                                                                                 str(lr),
                                                                                                 optimizer + str(
                                                                                                     weight_decay)
                                                                                                 , scheduler + str(
                gamma) + '-' + str(
                scheduler_step_size))
    else:
        run_name = "b{}_v{}_{}ne_{}_lr{}_optimizer({})_scheduler({})_({}_{}_{})".format(str(batch_size),
                                                                                                    str(v_context),
                                                                                                    str(n_epochs),
                                                                                                    visual_model_type,
                                                                                                    str(lr),
                                                                                                    optimizer + str(
                                                                                                        weight_decay),
                                                                                                    scheduler + str(
                                                                                                        gamma) + '-' + str(
                                                                                                        scheduler_step_size),
                                                                                                    str(var_v),
                                                                                                    str(var_av),
                                                                                                    str(var_a))
    if augmentation:
        run_name = run_name + '_(aug)'

    if freeze:
        run_name += '_f'
    if loss_fn == 'bce_with_logits_weighted':
        run_name += 'w' + str(loss_weight)
    if add_joint_loss:
        run_name += '_j_loss'

    if contrastive_loss:
        run_name += 'nce'

    tensorboard_dir = os.path.join(tensorboard_path, run_name)
    checkpoint_dir = os.path.join(checkpoint_path, run_name)


model_config = Config()
