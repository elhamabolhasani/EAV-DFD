import os


class Config:
    models_root = '/deepfake_models/'
    checkpoint_dir = '/deepfake_models/vocalist_fakeavceleb'
    pretrain_av_model_path = '/weights/vocalist_lsr2/vocalist_5f_lrs2.pth'
    final_data_path = '/deepfake_dataset/cropvideo_fakeavceleb'

config = Config()
