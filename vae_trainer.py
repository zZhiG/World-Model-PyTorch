import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.optim as optim

from models.vae import VAE
from utils.vae_dataset import VAE_Dataset
from arguments import get_args

def main():
    args = get_args()

    timeStr = time.strftime('_%Y.%m.%d.%H.%M.%S', time.localtime(time.time()))
    curve_save_name = os.path.join(args.train_vae_path, 'curve_'+timeStr)
    w_save_name = os.path.join(args.train_vae_path, 'vae_' + timeStr)

    prefix = ''
    filenames = os.path.abspath(__file__).split('\\')
    for f in filenames[:-1]:
        prefix += f + '\\'
    dir = os.path.join(prefix, args.data_save_path)
    dir = os.path.join(dir, args.sample_method)
    vae_curve_path = os.path.join(prefix, curve_save_name)
    w_save_name = os.path.join(prefix, w_save_name)

    if not os.path.exists(vae_curve_path):
        os.makedirs(vae_curve_path)
    if not os.path.exists(w_save_name):
        os.makedirs(w_save_name)

    writer = SummaryWriter(logdir=vae_curve_path)

    vae_dataset = VAE_Dataset(dir)
    dl = DataLoader(vae_dataset, batch_size=args.sample_epoch, shuffle=True)

    model = VAE(args.vae_latent, args.device).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    for e in range(args.train_vae_epoch):
        for idx, d in enumerate(dl):
            d = d.reshape(args.sample_epoch * args.sample_num, 3, *args.img_size)
            train_dl = DataLoader(d, batch_size=args.train_vae_batchsize, shuffle=True)

            start = time.time()
            tatal_loss = []
            bce_loss = []
            kl_loss = []
            for _, da in enumerate(train_dl):
                da = da.to(args.device)
                # print(da.shape)

                out, mu, logvar = model(da)
                loss, bce, kl = model.compute_loss(out, da, mu, logvar)

                loss.backward()
                optimizer.step()

                tatal_loss.append(loss.item())
                bce_loss.append(bce.item())
                kl_loss.append(kl.item())

                writer.add_scalar('Total Loss', loss.item(), idx*args.sample_num+_+1)
                writer.add_scalar('BCE Loss', bce.item(), idx*args.sample_num+_+1)
                writer.add_scalar('KL Loss', kl.item(), idx*args.sample_num+_+1)

            end = time.time()
            print('epoch:{}, fps:{:.3}, \n'
                  'total loss:{:.4}, bce loss:{:.4}, kl loss:{:.4}\n'.
                  format(e+1, end-start,
                         np.mean(tatal_loss), np.mean(bce_loss), np.mean(kl_loss)))

        if (e+1) % args.vae_save_interval == 0:
            torch.save(model, os.path.join(w_save_name, 'e+1.pt'))

if __name__ == '__main__':
    main()