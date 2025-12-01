import torch
from torch import nn
import torchvision
from torchvision import models
import numpy as np
import sys
import torch.nn.functional as F
from models.mcx_model.utils import init_weights_zero, init_weights_xavier_uniform, init_weights_xavier_normal, \
    init_weights_kaiming_uniform, init_weights_kaiming_normal
# from model.vit import vit_b_16
from models.mcx_model.resnet_multichannel import get_arch as Resnet_multi
from models.mcx_model.xception import xception
from models.mcx_model.xception_multichannel import xception_multichannels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


def pw_cosine_distance(vector):
    normalized_vec = F.normalize(vector)
    res = torch.mm(normalized_vec, normalized_vec.T)
    cos_dist = 1 - res
    return cos_dist


class API_Net(nn.Module):
    def __init__(self, num_classes=5, model_name='res101', weight_init='pretrained'):
        super(API_Net, self).__init__()

        # ---------Resnet101---------
        if model_name == 'res101':
            model = models.resnet101(pretrained=True)
            kernel_size = 14
        # layers = list(resnet101.children())[:-2]
        elif model_name == 'res101_9ch':
            resnet101_9_channel = Resnet_multi(101, 9)
            # use resnet34_4_channels(False) to get a non pretrained model
            model = resnet101_9_channel(True)
            kernel_size = 14
        elif model_name == 'res101_6ch':
            resnet101_6_channel = Resnet_multi(101, 6)
            model = resnet101_6_channel(True)
            kernel_size = 14

        # ---------Efficientnet---------
        elif model_name == 'effb0':
            model = models.efficientnet_b0(pretrained=True)
            kernel_size = 14
        elif model_name == 'effb1':
            model = models.efficientnet_b1(pretrained=True)
            kernel_size = 14
        elif model_name == 'effb2':
            model = models.efficientnet_b2(pretrained=True)
            kernel_size = 14
        elif model_name == 'effb3':
            model = models.efficientnet_b3(pretrained=True)
            kernel_size = 14
        elif model_name == 'effb4':
            model = models.efficientnet_b4(pretrained=True)
            kernel_size = 14
        elif model_name == 'effb5':
            model = models.efficientnet_b5(pretrained=True)
            kernel_size = 14
        elif model_name == 'effb6':
            model = models.efficientnet_b6(pretrained=True)
            kernel_size = 14
        elif model_name == 'effb7':
            model = models.efficientnet_b7(pretrained=True)
            kernel_size = 14

        # ---------Xception---------
        elif model_name == 'xception':
            model = xception()
            kernel_size = 14
        elif model_name == 'xception_9channels':
            model = xception_multichannels()
            kernel_size = 14
        elif model_name == 'xception_6channels':
            model = xception_multichannels(channel_num=6)
            kernel_size = 14

        # ---------Vision Transformer---------
        # elif model_name == 'vit_b_16':
        #     model = vit_b_16(pretrained=True)
        #     kernel_size = 28

        else:
            sys.exit('wrong model name baby')

        if weight_init == 'zero':
            model.apply(init_weights_zero)
            print('init weight 0')
        elif weight_init == 'xavier_uniform':
            print('init weight xavier uniform')
            model.apply(init_weights_xavier_uniform)
        elif weight_init == 'xavier_normal':
            print('init weight xavier normal')
            model.apply(init_weights_xavier_normal)
        elif weight_init == 'kaiming_uniform':
            print('init weight kaiming uniform')
            model.apply(init_weights_kaiming_uniform)
        elif weight_init == 'kaiming_normal':
            print('init weight kaiming normal')
            model.apply(init_weights_kaiming_normal)

        else:
            print('you are using pretrained model if you do not load the parameter')

        layers = list(model.children())[:-2]
        if 'res' in model_name:
            fc_size = model.fc.in_features
        elif 'eff' in model_name:
            fc_size = model.classifier[1].in_features
        elif 'vit' in model_name:
            fc_size = model.hidden_dim
        elif 'xception' in model_name:
            fc_size = 2048
        else:
            sys.exit('wrong network name baby')

        self.conv = nn.Sequential(*layers)
        self.avg = nn.AvgPool2d(kernel_size=kernel_size, stride=1)

        self.map1 = nn.Linear(fc_size * 2, 512)
        self.map2 = nn.Linear(512, fc_size)
        self.fc = nn.Linear(fc_size, num_classes)

        self.drop = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

        # # wrong 9-channel
        # self.conv_reduce = nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1)
        # # --- to here

    def forward(self, images):
        conv_out = self.conv(images)
        pool_out_old = self.avg(conv_out)
        pool_out = pool_out_old.squeeze()

        return self.fc(pool_out), pool_out
