# -*- coding:utf8 -*-
# @TIME     : 2021/4/22
# @Author   : LiuHanbing
# @File     : GANomaly_test.py

import os
import sys
sys.path.append('..')

import tqdm
import torch
import numpy as np
import cv2
import argparse
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from dataload.DNdata import load_data
from models.DCGAN_GANomaly import NetG, NetD
from evaluate import Evaluate, draw_heatmap
import torch.nn.functional as F
import math
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default=r"../experiments/0420addDlinear_test", help="path to save experiments results")
parser.add_argument("--n_epoches", type=int, default=1, help="number of epoches of training")
parser.add_argument("--batchSize", type=int, default=1, help="size of the batches")
parser.add_argument("--ngpu", type=int, default=1, help="number of gpu")
parser.add_argument("--nz", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--size", type=int, default=128, help="size of image after scaled")
parser.add_argument("--imageSize", type=int, default=12, help="size of each image dimension")
parser.add_argument("--nc", type=int, default=1, help="number of image channels")
parser.add_argument("--ngf", type=int, default=16, help="channels of middle layers for generator")
parser.add_argument("--n_extra_layers", type=int, default=0, help="extra layers of Encoder and Decoder")
parser.add_argument("--gen_pth", default=r"../experiments/0420addDlinear/gen_1999.pth", help="pretrained model of gen")
parser.add_argument("--disc_pth", default=r"../experiments/0420addDlinear/disc_1999.pth", help="pretrained model of disc")
opt = parser.parse_args()
print(opt)
os.makedirs(opt.experiment, exist_ok=True)

## cudnn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

## device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

## dataset
test_data_list, test_data_cnt, max_test_data, mean_test_data = load_data(batch_size = opt.batchSize, train = False)
opt.dataSize = test_data_cnt

## model
gen = NetG(opt).to(device)
disc = NetD(opt).to(device)
gen.load_state_dict(torch.load(opt.gen_pth))
disc.load_state_dict(torch.load(opt.disc_pth))
print("Pretrained models have been loaded.")

## record results
# import time
# now = time.localtime()
# strnow = time.strftime("%m%d-%H%M",now)
# writer = SummaryWriter("../runs{0}/".format(opt.experiment[1:]) + strnow, comment=opt.experiment[1:])


## loss
L_con = nn.L1Loss(reduction='mean')
L_enc = nn.MSELoss(reduction='mean')

## test
gen.eval()
disc.eval()
con_loss = []
enc_loss = []
labels = []
evaluation = Evaluate(opt.experiment)
tqdm_loader = tqdm.tqdm(test_data_list)
for i, test_input in enumerate(tqdm_loader):
    tqdm_loader.set_description(f"Test Sample {i+1} / {opt.dataSize}")
    test_inputs = test_input.to(device)

    ## inference
    with torch.no_grad():  # 该计算不被track, 不会在反向传播中被记录
        outputs, latent_in, latent_out = gen(test_inputs)
    con_loss.append(L_con(outputs, test_inputs).item())
    enc_loss.append(L_enc(latent_in, latent_out).item())

    residule = torch.abs(test_inputs - outputs)

    residual_loss = torch.sum(residule)
    final_l = residual_loss.detach().cpu().flatten()
    score = 1 / math.exp(0.3*final_l) * 100
    tqdm_loader.set_postfix(total_loss=final_l, score=score)
    print("\nloss{}: {}".format(i+1, final_l))

    single_out = F.interpolate(outputs, (120, 120), mode='bilinear', align_corners = True)
    single_input = F.interpolate(test_inputs, (120, 120), mode='bilinear', align_corners = True)
    single_residule = F.interpolate(residule, (120, 120), mode='bilinear', align_corners = True)     
    vutils.save_image(torch.cat((single_input, single_out), dim=0), '{0}/{1}-0.png'.format(opt.experiment, i), normalize=True)
    vutils.save_image(single_residule,'{0}/{1}-loss{2}.png'.format(opt.experiment, i, final_l), normalize=True)

    residule = single_residule.detach().cpu().numpy()
    residule = draw_heatmap(residule)
    cv2.imwrite('{0}/{1}-2.png'.format(opt.experiment, i), residule)

# enc_loss = np.array(enc_loss)
# enc_loss = (enc_loss - np.min(enc_loss)) / (np.max(enc_loss) - np.min(enc_loss))
# evaluation.labels = labels
# evaluation.scores = enc_loss
# evaluation.run()