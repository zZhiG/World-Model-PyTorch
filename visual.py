import os

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from arguments import get_args
from utils.vae_dataset import VAE_Dataset


def visualize():
    args = get_args()

    prefix = ''
    filenames = os.path.abspath(__file__).split('\\')
    for f in filenames[:-1]:
        prefix += f + '\\'
    dir = os.path.join(prefix, args.data_save_path)
    dir = os.path.join(dir, args.sample_method)

    vae_dataset = VAE_Dataset(dir)
    dl = DataLoader(vae_dataset, batch_size=args.sample_epoch, shuffle=False)

    for _, d in enumerate(dl):
        d = d.reshape(args.sample_epoch * args.sample_num, 3, *args.img_size)
        train_dl = DataLoader(d, batch_size=4, shuffle=False)
        for _, da in enumerate(train_dl):
            # print(da.shape)
            _, axs = plt.subplots(2, 2)
            axs[0, 0].imshow(da[0].permute(1, 2, 0))
            axs[0, 1].imshow(da[1].permute(1, 2, 0))
            axs[1, 0].imshow(da[2].permute(1, 2, 0))
            axs[1, 1].imshow(da[3].permute(1, 2, 0))
            plt.show()

if __name__ == '__main__':
    visualize()