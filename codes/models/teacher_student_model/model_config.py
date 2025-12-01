import os
from utils.preprocess_dataset.fakeavceleb_config import config
import torchvision.transforms as transforms


class Config:
    tensorboard_path = 'student_model_tensorboard/'

    dataset_root = '/datasets/student_datasets/polyglotfake/'
    dataset_name = 'polyglotfake'
    train_data_root = dataset_root + 'ts_dataset_cross3_active/train/'
    val_data_root = dataset_root + 'ts_dataset_cross3_active/val/'
    student_mode = True
    num_workers = 18

    # audio model hyperparameters
    audio_model_checkpoint_path = '/weights/' + 'audio_model/hubert-base-ls960/distill_hubret'
    audio_feature_extractor_checkpoint_path = '/weights/' + 'audio_model/hubert-base-ls960/feature_extractor'
    audio_pretrained_distill_model_path = '/weights/' + 'audio_model/distill_audio_model_v10.pt'

    # visual model hyperparameters

    v_img_size = 224
    pretrained_v_path = '/weights/mcx_rgb_model/model_mcx_api_rgb.tar'
    dropout = 0
    selected_idx = 2
    num_classes = 5  # 0 means real and others are fake
    # dist_type = 'euclidean'
    model_name = 'xception'
    weight_init = 'pretrained'
    visual_model_type = 'mcx'
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
    augmentation = True
    contrastive_loss = True
    aug_type = 4
    label_kind = 'and'
    test_strategy = 'mean_clip_prob'
    freeze = False
    classifier_type = 'joint'
    batch_size = 24  # 128
    optimizer = 'adam'  #
    weight_decay = 0  # set to 0 for off
    scheduler = 'step_lr'  # 'step_lr'
    gamma = 0.8
    scheduler_step_size = 15
    device_id = 0
    cuda = 'cuda:0'
    lr = 0.0001


    n_epochs = 82

    save_model_interval = 10
    if classifier_type == 'joint':
        run_name = "mini_b{}_v{}_{}ne_{}_lr{}_optimizer({})_scheduler({})_(joint)".format(str(batch_size),
                                                                                          str(v_context)
                                                                                          , str(n_epochs),
                                                                                          visual_model_type,
                                                                                          str(lr),
                                                                                          optimizer + str(
                                                                                              weight_decay)
                                                                                          ,
                                                                                          scheduler + str(
                                                                                              gamma) + '-' +
                                                                                          str(scheduler_step_size),
                                                                                          )
    else:
        run_name = "mini_b{}_v{}_{}ne_{}_lr{}_{}".format(
            str(batch_size), str(v_context)
            , str(n_epochs), visual_model_type, str(lr), train_data_root.split('/')[-3])

    if augmentation:
        run_name = run_name + '_(aug)'
    if student_mode:
        run_name += '_stu_mode'


model_config = Config()
model_config.run_name = '_'.join(model_config.run_name.split('_')[1:])
