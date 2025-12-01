import os
from utils.preprocess_dataset.fakeavceleb_config import config
import torchvision.transforms as transforms
import random
random.seed(1234)


class Config:
    #  model dirs
    config_file_path = '/models/ensemble_model/mini_model_config.py'
    config_file_name = 'mini_model_config.py'
    models_root = '/teacher_model_fakeavceleb/'
    checkpoint_path = models_root + 'mini_ensemble/'
    tensorboard_path = '/teacher_model_tensorboard/'
    model_root_path = '/models/teacher_model_fakeavceleb/'
    run_name = 'mini_b24_v20_400ne_mean_clip_prob_lr0.0001_optimizer(adam0)_scheduler(step_lr0.8-15)_and_(aug_4)_w3.0_1_c_loss_0.005/'
    model_name = 'checkpoint_step000000100.pth'
    model_checkpoint_path = None 

    # data parameters
    train_data_root = config.ensemble_dataset_train
    val_data_root = config.ensemble_dataset_val
    test_data_root = config.dfdc_val
    val_dfdc_data_root = config.dfdc_val
    num_workers = 18

    # audio model hyperparameters
    audio_model_checkpoint_path = '/weights/' + 'audio_model/hubert-base-ls960/distill_hubret'
    audio_feature_extractor_checkpoint_path = '/weights/' + 'audio_model/hubert-base-ls960/feature_extractor'
    audio_pretrained_distill_model_path = '/weights/' + 'audio_model/distill_audio_model_v10.pt'

    # visual model hyperparameters
    visual_model_type = 'mcx'
    if visual_model_type == 'vivit':
        v_img_size = 224
        image_patch_size = 8  # image patch size
        frame_patch_size = 5  # frame patch size
        num_classes = 2
        dim = 128
        spatial_depth = 2  # depth of the spatial transformer
        temporal_depth = 2  # depth of the temporal transformer
        heads = 2
        mlp_dim = 64
        channels = 3
        dropout = 0.1
        emb_dropout = 0.1
        selected_idx = 2
    elif visual_model_type == 'mcx':
        v_img_size = 224
        pretrained_v_path = 'weights/mcx_rgb_model/model_mcx_api_rgb.tar'
        dropout = 0
        selected_idx = 2
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
    v_context = 20  # 10  # 5
    num_audio_elements = v_context * 640  # 6400  # 16000/25 * syncnet_T
    tot_num_frames = 25  # buffer
    pretrained_av_path = '/weights/vocalist_lsr2/pure_MTDVocaLiST.pth'

    # training hyperparameters
    add_joint_loss = False
    contrastive_loss = True
    c_loss_var = 0.005
    augmentation = True
    aug_type = 4
    label_kind = 'and' # 'majority', 'and'
    test_strategy = 'mean_clip_prob'  
    freeze = False
    classifier_type = 'joint'  # 'split', 'joint'
    batch_size = 24  # 128
    optimizer = 'adam'  #
    weight_decay = 0  # set to 0 for off
    # momentum = # todo
    scheduler = 'step_lr'  # 'step_lr'
    gamma = 0.8
    scheduler_step_size = 15
    loss_fn = 'bce_with_logits_weighted'  # bce_with_logits_weighted, bce
    loss_weight_av = 3
    loss_weight_a_v = 1

    # joint loss
    ce_loss = 'bce_weighted' # bce_weighted, bce
    loss_weight_ce = 1

    device_id = 0
    cuda = 'cuda:0'
    lr = 0.0005


    n_epochs = 250  # 313 if for max . 314 is for max and min , 315 is for meantm

    save_model_interval = 20
    if classifier_type == 'joint':
        run_name = "mini_b{}_v{}_{}ne_{}_lr{}_optimizer({})_scheduler({})_(joint)_{}".format(str(batch_size),
                                                                                                       str(v_context)
                                                                                                       , str(n_epochs),
                                                                                                       test_strategy,
                                                                                                       str(lr),
                                                                                                       optimizer + str(
                                                                                                           weight_decay)
                                                                                                       ,
                                                                                                       scheduler + str(
                                                                                                           gamma) + '-' +
                                                                                                       str(scheduler_step_size)
                                                                                                      , label_kind)
    
    elif classifier_type == 'attention':
        run_name = "mini_b{}_v{}_{}ne_{}_lr{}_optimizer({})_scheduler({})_(attention)_{}".format(str(batch_size),
                                                                                                       str(v_context)
                                                                                                       , str(n_epochs),
                                                                                                       test_strategy,
                                                                                                       str(lr),
                                                                                                       optimizer + str(
                                                                                                           weight_decay)
                                                                                                       ,
                                                                                                       scheduler + str(
                                                                                                           gamma) + '-' +
                                                                                                       str(scheduler_step_size)
                                                                                                      , label_kind)

    else:
        run_name = "mini_b{}_v{}_{}ne_{}_lr{}_optimizer({})_scheduler({})_{}".format(
            str(batch_size), str(v_context)
            , str(n_epochs), test_strategy, str(lr),
            optimizer + str(weight_decay)
            , scheduler + str(gamma) + '-' + str(
                scheduler_step_size),label_kind)

    if augmentation:
        run_name = run_name + '_(aug_' + str(aug_type) + ')'
    if freeze:
        run_name += '_f'
    if loss_fn == 'bce_with_logits_weighted':
        run_name += '_w' + str(loss_weight_av) + '_' + str(loss_weight_a_v)
    if add_joint_loss:
        if ce_loss == 'bce':
            run_name += '_j_loss'
        else:
            run_name += '_j_loss' + str(loss_weight_ce)
    if contrastive_loss:
        if c_loss_var == '0.05':
            run_name += '_' + 'c_loss'
        else:
            run_name += '_' + 'c_loss_' + str(c_loss_var)

    tensorboard_dir = os.path.join(tensorboard_path, run_name)
    checkpoint_dir = os.path.join(checkpoint_path, run_name)


model_config = Config()