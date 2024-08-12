import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(description='CarRacing')

    parser.add_argument(
        '--env-name',
        default='CarRacing-v0'
    )
    parser.add_argument(
        '--img-size',
        type=tuple,
        default=(64, 64)
    )
    parser.add_argument(
        '--data-save-path',
        default='data\\CarRacing'
    )
    parser.add_argument(
        '--sample_method',
        default='random'
    )
    parser.add_argument(
        '--sample_epoch',
        type=int,
        default=200
    )
    parser.add_argument(
        '--sample-num',
        type=int,
        default=30
    )
    parser.add_argument(
        '--render',
        default=False
    )
    parser.add_argument(
        '--train-vae-epoch',
        type=int,
        default=10000
    )
    parser.add_argument(
        '--train-vae-batchsize',
        type=int,
        default=256
    )
    parser.add_argument(
        '--train-vae-path',
        default='weights\\vae_train'
    )
    parser.add_argument(
        '--vae-latent',
        type=int,
        default=32
    )
    parser.add_argument(
        '--vae-save-interval',
        type=int,
        default=1000
    )

    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return args

if __name__ == '__main__':
    args = get_args()
    print(args.img_size)
