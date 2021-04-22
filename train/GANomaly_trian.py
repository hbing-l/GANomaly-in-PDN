# -*- coding:utf8 -*-
# @TIME     : 2021/04/22
# @Author   : LiuHanbing
# @File     : GANomaly_trian.py

from __future__ import print_function
import os
import sys
sys.path.append('..')

import tqdm
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from dataload.DNdata import load_data
from models.DCGAN_GANomaly import NetD, NetG
import torch.nn.functional as F
import pdb


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default=r"../experiments/0420addDlinear", help="path to save experiments results")
parser.add_argument("--n_epoches", type=int, default=2000, help="number of epoches of training")
parser.add_argument("--batchSize", type=int, default=144, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--ngpu", type=int, default=1, help="number of gpu")
parser.add_argument("--nz", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--nc", type=int, default=1, help="number of image channels")
parser.add_argument("--ngf", type=int, default=16, help="channels of middle layers for generator")
parser.add_argument("--imageSize", type=int, default=12, help="size of each image dimension")
parser.add_argument("--n_extra_layers", type=int, default=0, help="extra layers of Encoder and Decoder")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
parser.add_argument("--lambda_adv", type=int, default=1, help="weight of loss_adv")
parser.add_argument("--lambda_con", type=int, default=40, help="weight of loss_con")
parser.add_argument("--lambda_enc", type=int, default=1, help="weight of loss_enc")
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
train_data_list, train_data_cnt, max_train_data, mean_train_data = load_data(batch_size = opt.batchSize, train = True)

## model init
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


## model
gen = NetG(opt).to(device)
disc = NetD(opt).to(device)

gen.apply(weights_init)
disc.apply(weights_init)

## record results
import time
now = time.localtime()
strnow = time.strftime("%m%d-%H%M",now)
writer = SummaryWriter("../runs{0}/".format(opt.experiment[1:]) + strnow, comment=opt.experiment[1:])

f = open("../runs{}/{}.txt".format(opt.experiment[1:], strnow), "w")
f.write("params\n{}\n".format(opt))
f.write("\ngenerator\n")
for params in gen.state_dict():   
    f.write("{}\t{}\n".format(params, gen.state_dict()[params].shape))
f.write("\ndiscriminator\n")
for params in disc.state_dict():   
    f.write("{}\t{}\n".format(params, disc.state_dict()[params].shape))

## Gaussian Distribution
def gen_z_gauss(i_size, nz):
    return torch.randn(i_size, nz, 1, 1).to(device)

## opt.dataSize
opt.dataSize = train_data_cnt

## loss function
gen_optimizer = optim.Adam(gen.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
disc_optimizer = optim.Adam(disc.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
L_adv = nn.MSELoss()        # For gen
L_con = nn.L1Loss()         # For gen
L_enc = nn.MSELoss()        # For gen
L_bce = nn.BCELoss()        # For disc


## Training
record = 0
with tqdm.tqdm(range(opt.n_epoches)) as t:
    for e in t:
        t.set_description(f"Epoch {e+1} /{opt.n_epoches} Per epoch {opt.dataSize}")
        L_adv_epoch_loss = 0.0
        L_con_epoch_loss = 0.0
        L_enc_epoch_loss = 0.0
        L_total_epoch_loss = 0.0
        disc_epoch_loss = 0.0

        for i, inputs in enumerate(train_data_list):
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            label_real = torch.ones(batch_size).to(device)
            label_fake = torch.zeros(batch_size).to(device)

            ## Update "D": max log(D(x)) + log(1-D(G(z))
            disc_optimizer.zero_grad()
            D_real, _ = disc(inputs)
            disc_loss_real = L_bce(D_real, label_real)
            outputs, _, _ = gen(inputs)
            D_fake, _ = disc(outputs.detach())
            disc_loss_fake = L_bce(D_fake, label_fake)
            disc_loss = (disc_loss_fake + disc_loss_real) * 0.5
            disc_loss.backward()
            disc_optimizer.step()
            disc_epoch_loss += disc_loss.item() * batch_size

            if record % 2 == 0:
                ## Update 'G' : max log(D(G(z)))
                gen_optimizer.zero_grad()
                outputs, latent_in, latent_out = gen(inputs)
                _, feature_fake = disc(outputs)
                _, feature_real = disc(inputs)
                adv_loss = L_adv(feature_fake, feature_real.detach())
                con_loss = L_con(outputs, inputs)
                enc_loss = L_enc(latent_out, latent_in)
                total_loss = opt.lambda_adv * adv_loss + \
                            opt.lambda_con * con_loss + \
                            opt.lambda_enc * enc_loss
                total_loss.backward()
                L_adv_epoch_loss += adv_loss.item() * batch_size
                L_con_epoch_loss += con_loss.item() * batch_size
                L_enc_epoch_loss += enc_loss.item() * batch_size
                L_total_epoch_loss += total_loss.item() * batch_size

                gen_optimizer.step()

            ## record results
            if record % opt.sample_interval == 0:
            # outputs.data = outputs.data.mul(0.5).add(0.5)

                single_out = outputs[0].unsqueeze(dim=0)
                single_out = F.interpolate(single_out, (120, 120), mode='bilinear', align_corners = True)
                single_input = inputs[0].unsqueeze(dim=0)
                single_input = F.interpolate(single_input, (120, 120), mode='bilinear', align_corners = True)                
                vutils.save_image(single_out,'{0}/outputs_{1}.png'.format(opt.experiment, record), normalize=True)
                vutils.save_image(single_input,'{0}/inputs_{1}.png'.format(opt.experiment, record), normalize=True)

            record += 1


        ## End of epoch
        L_adv_epoch_loss = L_adv_epoch_loss / opt.dataSize / 2
        L_con_epoch_loss = L_con_epoch_loss / opt.dataSize / 2
        L_enc_epoch_loss = L_enc_epoch_loss / opt.dataSize / 2
        L_total_epoch_loss = L_total_epoch_loss / opt.dataSize / 2
        disc_epoch_loss /= opt.dataSize
        t.set_postfix(L_adv=L_adv_epoch_loss, L_con=L_con_epoch_loss, L_enc=L_enc_epoch_loss,
                      L_total=L_total_epoch_loss, disc_loss=disc_epoch_loss)
        writer.add_scalar("gen_epoch_loss", L_total_epoch_loss, e)
        writer.add_scalar("disc_epoch_loss", disc_epoch_loss, e)

        if (e+1) % 50 == 0:
        # save model parameters
            torch.save(gen.state_dict(), '{0}/gen_{1}.pth'.format(opt.experiment, e))
            torch.save(disc.state_dict(), '{0}/disc_{1}.pth'.format(opt.experiment, e))

writer.close()