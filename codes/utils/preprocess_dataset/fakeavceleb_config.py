import os


class Config:
    dataset_root = ''
    fake_av_celeb_meta_data = ''
    fakeavceleb_dataset = dataset_root + 'fakeavceleb_dataset/'
    preprocess_dataset = dataset_root + 'preprocessed_fakeavceleb/'

    vocalist_dataset = dataset_root + ''
    vocalist_dataset_val = vocalist_dataset + 'val/'
    vocalist_dataset_train = vocalist_dataset + 'train/'
    vocalist_dataset_test = vocalist_dataset + 'test/'

    ensemble_dataset = dataset_root + 'ensemble_dataset/'
    ensemble_dataset_val = ensemble_dataset + 'val/'
    ensemble_dataset_train = ensemble_dataset + 'train/'
    ensemble_dataset_test = ensemble_dataset + 'test/'      
    ensemble_test_train_ratio = 0.3
    ensemble_val_ratio = 0.145

    pretrain_av_model_path = 'vocalist_5f_lrs2.pth'
    face_detection_weights = 's3fd.pth'
    test_train_ratio = 0.2
    val_ratio = 0.125 
    fakeavceleb_dataset_mathods = ['real', 'rtvc', 'faceswap', 'faceswap-wav2lip', 'fsgan', 'fsgan-wav2lip', 'wav2lip']

config = Config()
