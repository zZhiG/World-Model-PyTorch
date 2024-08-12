import os
import time

import matplotlib.pyplot as plt
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

    timeStr = time.strftime('%Y.%m.%d.%H.%M.%S', time.localtime(time.time()))
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
                optimizer.zero_grad()

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
            torch.save(model, os.path.join(w_save_name, f'{e+1}.pt'))

# Visualize the training effect of VAE
def visualize_sample(weight_path):
    args = get_args()

    prefix = ''
    filenames = os.path.abspath(__file__).split('\\')
    for f in filenames[:-1]:
        prefix += f + '\\'
    dir = os.path.join(prefix, args.data_save_path)
    dir = os.path.join(dir, args.sample_method)

    vae_dataset = VAE_Dataset(dir)
    dl = DataLoader(vae_dataset, batch_size=args.sample_epoch, shuffle=True)

    model = torch.load(weight_path, map_location='cuda:0')

    for _, d in enumerate(dl):
        d = d.reshape(args.sample_epoch * args.sample_num, 3, *args.img_size)
        train_dl = DataLoader(d, batch_size=1, shuffle=True)
        for _, da in enumerate(train_dl):
            da = da.to(args.device)
            out, _, _ = model(da)

            _, axs = plt.subplots(1, 2)
            axs[0].imshow(da[0].detach().cpu().permute(1, 2, 0))
            axs[1].imshow(out[0].detach().cpu().permute(1, 2, 0))
            plt.show()



if __name__ == '__main__':
    main()

    weight_path = r'F:\our_code\RL\world_model\weights\vae_train\vae_2024.07.30.16.09.46\10000.pt'
    # visualize_sample(weight_path)
