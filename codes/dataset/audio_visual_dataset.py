from os.path import dirname, join, basename, isfile
from natsort import natsorted
from glob import glob
import torch
import soundfile as sf
import os
import random

random.seed(1234)
import cv2
from torchvision.transforms import v2
import tqdm as tqdm
import numpy as np
from random import randrange, randint
from transformers import Wav2Vec2FeatureExtractor
from PIL import Image
import torchvision


def get_video_names(video_dir, kind, v_context, aug_flag):
    video_clip_names = []
    
    if len(os.listdir(video_dir)) <= 1:
        return video_clip_names

    video_frames_count = len(os.listdir(video_dir)) - 1
    if kind == 'all_clip':
        num_clip = video_frames_count // v_context
        for i in range(num_clip):
            if aug_flag:
                video_clip_names.append(video_dir + '_' + 'Aug' + '_' + str(i * v_context))

            video_clip_names.append(video_dir + '_' + 'NoAug' + '_' + str(i * v_context))
    else:
        if video_frames_count - v_context < 0:
            return video_clip_names

        clip_idx = randint(0, video_frames_count - v_context)
        if aug_flag and random.random() > 0.5:
            video_clip_names.append(video_dir + '_' + 'Aug' + '_' + str(clip_idx))

        video_clip_names.append(video_dir + '_' + 'NoAug' + '_' + str(clip_idx))

    return video_clip_names


def get_test_video_list(data_root):
    filelist = []

    for video_dir in os.listdir(data_root):
        filelist.append(os.path.join(data_root, video_dir))
    return filelist


def get_train_clip_list(split, data_root, v_context, student_mode, aug_flag):
    filelist = []
    count_real = 0
    count_fake = 0
    print(len(os.listdir(data_root)))
    for video_dir in os.listdir(data_root):
        if video_dir.startswith("3"):
            count_real += 1
        else:
            count_fake += 1

    print('count_fake: ', count_fake)
    print('count_real:', count_real)

    if split == 'train':
        for video_dir in os.listdir(data_root):
            # it means it's a real video
            if video_dir.startswith("3"):
                filelist += get_video_names(os.path.join(data_root, video_dir), 'all_clip', v_context, aug_flag)
            else:
                # filelist += get_video_names(os.path.join(data_root, video_dir), 'all_clip', v_context, aug_flag)
                if student_mode:
                    filelist += get_video_names(os.path.join(data_root, video_dir), 'all_clip', v_context, aug_flag)
                else:
                    filelist += get_video_names(os.path.join(data_root, video_dir), 'one_clip', v_context, aug_flag)
    else:
        # means validation data
        for video_dir in os.listdir(data_root):
            # it means it's a real video
            if video_dir.startswith("3"):
                filelist += get_video_names(os.path.join(data_root, video_dir), 'one_clip', v_context, aug_flag)
            else:
                filelist += get_video_names(os.path.join(data_root, video_dir), 'one_clip', v_context, aug_flag)

    return filelist


def dataset_analysis(dataloader):
    fake_nums = 0
    real_nums = 0
    prog_bar = tqdm(enumerate(dataloader))
    for step, (vid, aud, y) in prog_bar:
        real_nums += sum(y)
        fake_nums += len(y) - sum(y)
    print('train class num : fake-> ', fake_nums, '  real-> ', real_nums)


class Dataset(object):
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_wav(self, wavpath, vid_frame_id, num_audio_elements):
        aud = sf.SoundFile(wavpath)
        can_seek = aud.seekable()
        pos_aud_chunk_start = vid_frame_id * 640
        try:
            _ = aud.seek(pos_aud_chunk_start)
            wav_vec = aud.read(num_audio_elements)
            return wav_vec
        except sf.LibsndfileError:
            return None

    def rms(self, x):
        val = np.sqrt(np.mean(x ** 2))
        if val == 0:
            val = 1
        return val

    def get_window(self, start_frame, v_context):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + v_context):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames


def default_loader(path):
    try:
        img = Image.open(path).convert('RGB')
    except:
        print('error happens !!!!!!!!!')
        return Image.new('RGB', (224, 224), 'white')
    return img


class TrainDataset(Dataset):
    def __init__(self, split, model_config):
        self.split = split
        self.model_config = model_config
        if split == 'train':
            student_mode = False
            if hasattr(model_config, 'student_mode') and model_config.student_mode is True:
                student_mode = True
            self.all_videos = get_train_clip_list(split, model_config.train_data_root, model_config.v_context,
                                                  student_mode, model_config.augmentation)
        else:
            self.all_videos = get_train_clip_list(split, model_config.val_data_root, model_config.v_context,
                                                  student_mode=False, aug_flag=False)


        self.audio_feature_extractor = (
            Wav2Vec2FeatureExtractor.from_pretrained(model_config.audio_feature_extractor_checkpoint_path))


    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        need_change_video = False
        need_change_video_clip = False
        while 1:
            if need_change_video:
                idx = random.randint(0, len(self.all_videos) - 1)
                need_change_video = False
                need_change_video_clip = False

            vidname = '_'.join(self.all_videos[idx].split('_')[0:-2])
            aug_flag = self.all_videos[idx].split('_')[-2]
            pos_frame_id = int(self.all_videos[idx].split('_')[-1])
            img_names = natsorted(list(glob(join(vidname, '*.jpg'))), key=lambda y: y.lower())
            interval_st, interval_end = 0, len(img_names)
            wavpath = join(vidname, "audio.wav")
            if need_change_video_clip:
                pos_frame_id = random.randint(interval_st, interval_end - self.model_config.v_context)
                need_change_video_clip = False

            if interval_end - interval_st <= self.model_config.tot_num_frames:
                need_change_video = True
                # print('video clip changed video tot_num_frames ...')
                continue

            pos_wav = self.get_wav(wavpath, pos_frame_id, self.model_config.num_audio_elements)
            if pos_wav is None:
                need_change_video_clip = True
                # print('unable to extract audio features ...')
                continue

            img_name = os.path.join(vidname, str(pos_frame_id) + '.jpg')
            window_fnames = self.get_window(img_name, self.model_config.v_context)

            if window_fnames is None:
                need_change_video_clip = True
                # print('video clip changed get window return none ...')
                continue

            # read images (v_context) -> window
            av_window = []
            v_window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    if self.model_config.visual_model_type == 'vivit':
                        v_img = cv2.resize(img, (self.model_config.v_img_size, self.model_config.v_img_size))
                    else:
                        v_img = default_loader(fname)

                    av_img = cv2.resize(img, (96, 96))
                except Exception as e:
                    all_read = False
                    break

                v_window.append(v_img)
                av_window.append(av_img)

            if not all_read:
                need_change_video_clip = True
                # print('video clip changed unable to real all images...')
                continue

            if aug_flag == 'Aug':
                av_window, v_window = self.apply_augmentation(av_window, v_window, self.model_config.aug_type)

            if vidname.split('/')[-1].split('_')[0] == '3':
                y_av = torch.ones(1).float()
                y_v = torch.ones(1).float()
                y_a = torch.ones(1).float()
            else:
                y_av = torch.zeros(1).float()
                if vidname.split('/')[-1].split('_')[0] == '1':
                    y_v = torch.ones(1).float()
                    y_a = torch.zeros(1).float()

                elif vidname.split('/')[-1].split('_')[0] == '2':
                    y_v = torch.zeros(1).float()
                    y_a = torch.ones(1).float()
                    if 'dfdc' in vidname:
                        y_av = torch.ones(1).float()
                elif vidname.split('/')[-1].split('_')[0] == '0':
                    y_v = torch.zeros(1).float()
                    y_a = torch.zeros(1).float()

            # H, W, T, 3 --> T*3
            vid_av = np.concatenate(av_window, axis=2) / 255.
            vid_av = vid_av.transpose(2, 0, 1)
            vid_av = torch.FloatTensor(vid_av[:, 48:])

            wav_av = pos_wav
            aud_av = torch.FloatTensor(wav_av)

            if self.model_config.visual_model_type == 'vivit':
                vid_v = np.stack(v_window, axis=0)
                vid_v = torch.FloatTensor(vid_v)
                vid_v = vid_v.permute(3, 0, 1, 2)
            else:
                vid_v = v_window[self.model_config.selected_idx]
                vid_v = self.model_config.transform_picked(np.asarray(vid_v).astype('uint8'))

            aud_a = self.audio_feature_extractor(pos_wav, return_tensors="pt",
                                                 sampling_rate=self.model_config.sample_rate).input_values[0]

            if torch.any(torch.isnan(vid_av)) or torch.any(torch.isnan(aud_av)):
                need_change_video_clip = True
                continue

            if vid_av is None or aud_av is None or aud_av.shape[0] != self.model_config.v_context * 640:
                need_change_video_clip = True
                continue

            assert aud_av.shape[0] == self.model_config.v_context * 640

            return vid_v, vid_av, aud_av, aud_a, y_av, y_v, y_a

        # print(vid_v.shape)
        # print(vid_av.shape)
        # print(aud_av.shape)
        # print(aud_a.shape)
        # print(y_av.shape)
        # print(y_v.shape)
        # print(y_a.shape)
        # torch.Size([10, 3, 10, 96, 96])
        # torch.Size([10, 30, 48, 96])
        # torch.Size([10, 6400])
        # torch.Size([10, 6400])
        # torch.Size([10, 1])
        # torch.Size([10, 2])
        # torch.Size([10, 1])

    def convert_to_pil(self, img):
        img = np.asarray(img).astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        return im_pil

    def convert_to_imread(self, img):
        numpy_image = np.asarray(img).astype('uint8')
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        return opencv_image

    def random_crop(self, window, th, tw):
        _, h, w = torchvision.transforms.functional.get_dimensions(window[0])
        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()

        window = [torchvision.transforms.functional.crop(img, i, j, h, w) for img in window]
        return window

    def apply_augmentation(self, av_window, v_window, aug_type):
        av_window = [self.convert_to_pil(img) for img in av_window]
        v_window = [self.convert_to_pil(img) for img in v_window]
        
        if random.random() < 0.5:
            if random.random() < 0.5:
                brightness_transform = (lambda img: torchvision.transforms.functional.adjust_brightness(img, 0.5))
                av_window = [brightness_transform(img) for img in av_window]
                v_window = [brightness_transform(img) for img in v_window]
            else:
                contrast_transform = (lambda img: torchvision.transforms.functional.adjust_contrast(img, 0.5))
                av_window = [contrast_transform(img) for img in av_window]
                v_window = [contrast_transform(img) for img in v_window]
        else:
            if aug_type == 5:
                if random.random() < 0.5:
                    hfilp_transform = (lambda img: torchvision.transforms.functional.hflip(img))
                    av_window = [hfilp_transform(img) for img in av_window]
                    v_window = [hfilp_transform(img) for img in v_window]

                if random.random() < 0.5:
                    av_window = self.random_crop(av_window, th=80, tw=80)
                    _, h, w = torchvision.transforms.functional.get_dimensions(v_window[0])
                    v_window = self.random_crop(v_window, th=h - int(h * 0.15), tw=w - int(w * 0.15))

        if random.random() < 0.5:
            blurrer = v2.GaussianBlur(kernel_size=(3, 15))
            av_window = [blurrer(img) for img in av_window]
            v_window = [blurrer(img) for img in v_window]

        if random.random() < 0.3:
            mean = 0
            sigma = randint(0, 50)
            w, h = v_window[self.model_config.selected_idx].size
            gaussian_v = np.random.normal(mean, sigma, (h, w))
            gaussian_av = cv2.resize(gaussian_v, (96, 96))
            rgb_gaussian_av = np.repeat(gaussian_av[:, :, np.newaxis], 3, axis=2)
            rgb_gaussian_v = np.repeat(gaussian_v[:, :, np.newaxis], 3, axis=2)
            av_window = [img + rgb_gaussian_av for img in av_window]
            if self.model_config.visual_model_type == 'mcx':
                v_window[self.model_config.selected_idx] = v_window[self.model_config.selected_idx] + rgb_gaussian_v
            else:
                v_window = [img + rgb_gaussian_v for img in v_window]

            av_window = [self.convert_to_imread(img) for img in av_window]
            v_window = [self.convert_to_imread(img) for img in v_window]
            return av_window, v_window

        if random.random() < 0.5:
            angle = randint(10, 45)
            rotate_transform = (lambda img: torchvision.transforms.functional.rotate(img, angle))
            av_window = [rotate_transform(img) for img in av_window]
            v_window = [rotate_transform(img) for img in v_window]

        if random.random() < 0.1:
            av_window = [v2.Grayscale(num_output_channels=3)(img) for img in av_window]
            v_window = [v2.Grayscale(num_output_channels=3)(img) for img in v_window]

        if random.random() < 0.2:
            if self.model_config.visual_model_type == 'vivit':
                av_window = [v2.Resize(size=96)(v2.Resize(size=75)(img)) for img in av_window]
                v_window = [v2.Resize(size=224)(v2.Resize(size=190)(img)) for img in v_window]
            else:
                av_window = [v2.Resize(size=96)(v2.Resize(size=75)(img)) for img in av_window]
                new_v_window = []
                for img in v_window:
                    shape = img.size
                    new_v_window.append(v2.Resize(size=shape)(
                        v2.Resize(size=(int(shape[0] - (shape[0] * 0.3)), int(shape[1] - (shape[1] * 0.3))))(img))
                    )
                v_window = new_v_window

        av_window = [self.convert_to_imread(img) for img in av_window]
        v_window = [self.convert_to_imread(img) for img in v_window]
        return av_window, v_window


class TestDataset(Dataset):
    def __init__(self, split, model_config):
        self.split = split
        self.model_config = model_config
        self.all_videos = get_test_video_list(model_config.test_data_root)
        self.audio_feature_extractor = (
            Wav2Vec2FeatureExtractor.from_pretrained(model_config.audio_feature_extractor_checkpoint_path))

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        av_clips, v_clips = [], []
        av_audio_elements, a_audio_elements = [], []
        av_labels, a_labels, v_labels = [], [], []
        num_try = 0
        selected_clip = 0
        pos_frame_id = -self.model_config.v_context
        vidname = self.all_videos[idx]
        print('in dataset : ', vidname)

        if self.model_config.test_dataset_name in ['vid_df_timit', 'val_vid_timit', 'test_vid_timit']:
            vidname = join(vidname, vidname.split('_')[-1])

        if 'polyglotfake' in self.model_config.test_dataset_name:
            vidname = join(vidname, vidname.split('/')[-1])

        elif 'dfdc' in self.model_config.test_dataset_name:
            if 'REAL' in vidname:
                vidname = join(vidname, vidname.split('_')[-1])
            else:
                vidname = join(vidname, vidname.split('_')[-2])

        
        wavpath = join(vidname, "audio.wav")
        print(vidname)
        img_names = natsorted(list(glob(join(vidname, '*.jpg'))), key=lambda y: y.lower())
        interval_st, interval_end = 0, len(img_names)

        while selected_clip < self.model_config.num_clip_check:
            num_try += 1

            if num_try > 10 * self.model_config.num_clip_check:
                print(vidname)
                break

            
            pos_frame_id = random.randint(interval_st, interval_end - self.model_config.v_context)
            pos_wav = self.get_wav(wavpath, pos_frame_id, self.model_config.num_audio_elements)
            if pos_wav is None:
                continue

            img_name = os.path.join(vidname, str(pos_frame_id) + '.jpg')
            window_fnames = self.get_window(img_name, self.model_config.v_context)
            if window_fnames is None:
                continue

            av_window = []
            v_window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    v_img = default_loader(fname)
                    img = cv2.resize(img, (96, 96))

                except Exception as e:
                    all_read = False
                    break
                v_window.append(v_img)
                av_window.append(img)

            if not all_read:
                continue

            # set video label ---> real = 1, fake = 0
            if self.model_config.test_dataset_name in ['vid_df_timit', 'val_vid_timit', 'test_vid_timit'] :
                if vidname.split('/')[-2].split('_')[0] == '1':
                    y_av = torch.ones(1).float()
                    y_a = -1
                    y_v = -1
                else:
                    y_av = torch.zeros(1).float()
                    y_a = -1
                    y_v = -1
            elif 'dfdc' in self.model_config.test_dataset_name:
                if vidname.split('/')[-2].split('_')[0] == 'REAL':
                    y_av = torch.ones(1).float()
                    y_a = -1
                    y_v = -1
                else:
                    y_av = torch.zeros(1).float()
                    y_a = -1
                    y_v = -1
            else:
                if vidname.split('/')[-1].split('_')[0] == '3':
                    y_av = torch.ones(1).float()
                    y_v = torch.ones(1).float()
                    y_a = torch.ones(1).float()
                else:
                    y_av = torch.zeros(1).float()
                    if vidname.split('/')[-1].split('_')[0] == '1':
                        y_v = torch.ones(1).float()
                    else:
                        y_v = torch.zeros(1).float()

                    if vidname.split('/')[-1].split('_')[0] == '2':
                        y_a = torch.ones(1).float()
                    else:
                        y_a = torch.zeros(1).float()

            # H, W, T, 3 --> T*3
            vid_av = np.concatenate(av_window, axis=2) / 255.
            vid_av = vid_av.transpose(2, 0, 1)
            vid_av = torch.FloatTensor(vid_av[:, 48:])

            wav_av = pos_wav
            aud_av = torch.FloatTensor(wav_av)
            vid_v = v_window[self.model_config.selected_idx]
            vid_v = self.model_config.transform_picked(np.asarray(vid_v).astype('uint8'))

            aud_a = self.audio_feature_extractor(pos_wav, sampling_rate=self.model_config.sample_rate).input_values[0]
            aud_a = torch.FloatTensor(aud_a)

            if torch.any(torch.isnan(vid_av)) or torch.any(torch.isnan(aud_av)):
                continue

            if vid_av is None or aud_av is None or aud_av.shape[0] != self.model_config.v_context * 640:
                continue

            av_clips.append(vid_av)
            v_clips.append(vid_v)
            av_audio_elements.append(aud_av)
            a_audio_elements.append(aud_a)
            av_labels.append(y_av)
            v_labels.append(y_v)
            a_labels.append(y_a)

            selected_clip += 1


        return (torch.stack(v_clips), torch.stack(av_clips), torch.stack(av_audio_elements),
                torch.stack(a_audio_elements), av_labels[0], v_labels[0], a_labels[0], vidname)