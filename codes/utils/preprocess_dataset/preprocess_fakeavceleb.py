import shutil

import pandas as pd
from sklearn.model_selection import train_test_split
import os
from fakeavceleb_config import config


def label_to_int(label):
    if label == 'RealVideo-RealAudio':
        return 3
    elif label == 'FakeVideo-FakeAudio':
        return 0
    elif label == 'RealVideo-FakeAudio':
        return 1
    else:
        return 2


def make_video_dir_name(video_path):
    names = video_path.split('/')
    return str(label_to_int(names[-5])) + '_' + names[0] + '_' + names[-2] + '_' + names[-1].split('.')[0]


def split_train_test(meta_data):
    source_ids = meta_data['source']
    source_ids = list(set(source_ids))
    train_source_ids, test_source_ids = train_test_split(source_ids, test_size=config.test_train_ratio, shuffle=True)
    train_source_ids, val_source_ids = train_test_split(train_source_ids, test_size=config.val_ratio, shuffle=True)
    meta_data_train = meta_data[meta_data['source'].isin(train_source_ids)]
    meta_data_test = meta_data[meta_data['source'].isin(test_source_ids)]
    meta_data_val = meta_data[meta_data['source'].isin(val_source_ids)]
    assert len(meta_data) == len(meta_data_test) + len(meta_data_train) + len(meta_data_val)
    meta_data_train.to_csv(os.path.join(config.fakeavceleb_dataset, 'train_meta_data.csv'))
    meta_data_test.to_csv(os.path.join(config.fakeavceleb_dataset, 'test_meta_data.csv'))
    meta_data_val.to_csv(os.path.join(config.fakeavceleb_dataset, 'val_meta_data.csv'))
    return meta_data_train, meta_data_test, meta_data_val


def make_dataset_folders(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_label(type):
    if type == 'RealVideo-RealAudio':
        return 'real'
    else:
        return 'fake'


def make_fakeavceleb_dataset(meta_data):
    data_dir = config.fakeavceleb_dataset
    for index, row in meta_data.iterrows():
        video_path = os.path.join(config.dataset_root, row['video_path'])
        label = get_label(row['type'])
        os.makedirs(os.path.join(data_dir, row['video_dir_name']))
        shutil.copyfile(video_path, os.path.join(data_dir, row['video_dir_name'], row['video_dir_name'] + '.mp4'))


def main():
    # First run this part
    meta_data = pd.read_csv(config.fake_av_celeb_meta_data)
    meta_data['video_path'] = meta_data[['Unnamed: 9', 'path']].agg('/'.join, axis=1)
    meta_data['name_columns'] = meta_data[['method', 'Unnamed: 9', 'path']].agg('/'.join, axis=1)
    meta_data['video_dir_name'] = meta_data['name_columns'].apply(make_video_dir_name)
    if not os.path.exists(config.fakeavceleb_dataset):
        os.makedirs(config.fakeavceleb_dataset)

    make_fakeavceleb_dataset(meta_data)

    # Second run preprocess file preprosess.py on fakeavceleb_dataset
    print('split train and test ...')
    meta_data_train, meta_data_test, meta_data_val = split_train_test(meta_data)


if __name__ == "__main__":
    main()