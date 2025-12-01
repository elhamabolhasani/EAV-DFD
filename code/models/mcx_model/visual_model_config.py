import os
from utils.preprocess_dataset.visual_dataset_ff_plusplus_config import config
import torchvision.transforms as transforms


class Config:
    #  model dirs
    config_file_path = '/models/mcx_model/visual_model_config.py'
    config_file_name = 'vidual_model_config.py'
    models_root = '/deepfake_models/'
    checkpoint_path = models_root + 'mcx_model/'
    tensorboard_path = 'visual_mcx_tensorboard/'
    model_checkpoint_path = None

    # data parameters
    dataset = 'ff++'
    train_data_root = config.ff_dataset_train
    val_data_root = config.ff_dataset_val

    num_workers = 18
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
    pretrained_av_path = 'vocalist_lsr2/vocalist_5f_lrs2.pth'


    # visual model hyperparameters
    start_epoch = 0
    epochs = 100
    num_classes = 5
    n_classes_total = 5
    dist_type = 'euclidean'
    model_name = 'xception'
    weight_init = 'pretrained'
    lr = 0.01
    momentum = 0.9
    weight_decay = 5e-4
    resume = None
    image_loader = 'default_loader'
    v_img_size = 224
    pretrained_v_path = 'weights/mcx_rgb_model/model_mcx_api_rgb.tar'
    if image_loader == 'nine_channels' or image_loader == 'temporal_9':
        transform_picked = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([512, 512]),
            transforms.RandomCrop([448, 448]),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225)
            )])
    else:
        transform_picked = transforms.Compose([
            transforms.Resize([512, 512]),
            transforms.RandomCrop([448, 448]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )])

    # ensemble model hyperparameters
    batch_size = 8  # 128
    optimizer = 'adam'  #
    n_epochs = 300

    save_model_interval = 10
    run_name = "b{}_v{}_{}ne_lr{}_optimizer({})_dataset({})".format(str(batch_size), str(v_context)
            , str(n_epochs), str(lr), optimizer + str(weight_decay), dataset)

    tensorboard_dir = os.path.join(tensorboard_path, run_name)
    checkpoint_dir = os.path.join(checkpoint_path, run_name)


model_config = Config()
