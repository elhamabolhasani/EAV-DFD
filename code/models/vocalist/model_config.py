import os


class Config:
    models_root = '/deepfake_models/'
    checkpoint_dir = models_root + 'vocalist_fakeavceleb'
    tensorboard_dir = '/audio_visual_tensorboard/'


model_config = Config()
