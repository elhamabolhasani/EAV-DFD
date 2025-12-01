import os
from models.utils import config


class Config:
    #  model dirs
    config_file_path = '/models/ensemble_model/mini_model_config.py'
    config_file_name = 'model_config.py'
    models_root = '/deepfake_models/'
    checkpoint_path = models_root + 'ensemble/'
    tensorboard_path = '/deepfake_project/ensemble_model_tensorboard/'
    model_checkpoint_path = None

    # data parameters
    train_data_root = config.vocalist_dataset_train
    val_data_root = config.vocalist_dataset_val
    num_workers = 18

    # audio model hyperparameters
    audio_model_checkpoint_path = models_root + 'audio_model/hubert-base-ls960/distill_hubret'
    audio_feature_extractor_checkpoint_path = models_root + 'audio_model/hubert-base-ls960/feature_extractor'

    # visual model hyperparameters
    visual_model_type = 'mcx'
    v_img_size = 224
    pretrained_v_path = '/weights/mcx_rgb_model/model_mcx_api_rgb.tar'
    dropout = 0
    num_classes = 5
    model_name = 'xception'
    weight_init = 'pretrained'

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
    v_context = 10  # 10  # 5
    num_audio_elements = v_context * 640  # 6400  # 16000/25 * syncnet_T
    tot_num_frames = 25  # buffer
    pretrained_av_path = '/weights/vocalist_lsr2/pure_MTDVocaLiST.pth'

    # training hyperparameters
    classifier_type = 'joint'
    batch_size = 64  # 128
    optimizer = 'adam'  #
    weight_decay = 0.01  # set to 0 for off
    # momentum = # todo
    scheduler = 'step_lr'  # 'step_lr'
    gamma = 0.7
    scheduler_step_size = 15
    loss_fn = 'bce_with_logits_weighted'
    loss_weight = 3.0
    var_v = 0.1
    var_av = 0.8
    var_a = 0.1

    lr = 0.0001
    n_epochs = 500

    save_model_interval = 100
    run_name = "b{}_v{}_{}ne_lr{}_optimizer({})_scheduler({})_dropout({})".format(str(batch_size), str(v_context)
                                                                                  , str(n_epochs), str(lr),
                                                                                  optimizer + str(weight_decay)
                                                                                  , scheduler + str(gamma) + '-' + str(
            scheduler_step_size), str(dropout))

    tensorboard_dir = os.path.join(tensorboard_path, run_name)
    checkpoint_dir = os.path.join(checkpoint_path, run_name)


model_config = Config()
