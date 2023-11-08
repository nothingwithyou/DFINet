from pathlib import Path
from itertools import chain
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from core.model import build_model
from core.checkpoint import CheckpointIO
from os.path import join as ospj
import torchvision.utils as vutils
import argparse
import torch.nn.functional as F


def make_args():
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--latent_channels', type=int, default=64,
                        help='latent channels for coarse')
    parser.add_argument('--norm', type=str, default='in',
                        help='normalization type for coarse')
    parser.add_argument('--pad_type', type=str, default='zero',
                        help='the padding type for coarse')
    parser.add_argument('--activation', type=str, default='elu',
                        help='the activation type for coarse')  # elu
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')

    # training arguments
    parser.add_argument('--resume_iter', type=int, default=300000,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=14,
                        help='Batch size for training')
    parser.add_argument('--num_outs_per_domain', type=int, default=10,
                        help='Number of generated images per domain during sampling')

    # misc
    parser.add_argument('--num_workers', type=int, default=40,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # directory for testing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory for saving network checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory for saving generated images and videos')
    parser.add_argument('--src_dir', type=str, default='data/val',
                        help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='assets/representative/celeba_hq/ref',
                        help='Directory containing input reference images')
    parser.add_argument('--out_dir', type=str, default='outs',
                        help='output directory when aligning faces')
    parser.add_argument('--margin', type=int, default=10, help='margin of image')
    parser.add_argument('--mask_num', type=int, default=20, help='number of mask')
    parser.add_argument('--bbox_shape', type=int, default=30, help='margin of image for bbox mask')
    parser.add_argument('--max_angle', type=int, default=4, help='parameter of angle for free form mask')
    parser.add_argument('--max_len', type=int, default=40, help='parameter of length for free form mask')
    parser.add_argument('--max_width', type=int, default=10, help='parameter of width for free form mask')
    return parser


def read_data(fn1, fn2, transform):
    img = Image.open(fn1).convert('RGB')
    img2 = Image.open(fn2).convert('RGB')
    img = transform(img)
    img2 = transform(img2)
    return img.unsqueeze(0), img2.unsqueeze(0)


def rectanglemask(shape):
    bbox = (64, 64, 128, 128)
    height = shape
    width = shape
    mask = np.zeros((height, width), np.float32)
    mask[(bbox[0]): (bbox[0] + bbox[2]), (bbox[1]): (bbox[1] + bbox[3])] = 1.
    return mask.reshape((1, ) + mask.shape).astype(np.float32)


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


def save_image(x, ncol, filename):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


def make_dataset(root):
    domains = os.listdir(root)
    fnames, labels = [], []
    for idx, domain in enumerate(sorted(domains)):
        class_dir = os.path.join(root, domain)
        cls_fnames = listdir(class_dir)
        fnames += cls_fnames
        labels += [idx] * len(cls_fnames)
    return fnames, labels


def adv_loss(logits, target):
    targets = torch.full_like(logits, fill_value=target)
    # loss = F.binary_cross_entropy_with_logits(logits, targets)
    loss = F.mse_loss(logits, targets, reduction='none')
    return loss


if __name__ == '__main__':
    parser = make_args()
    device = 'cuda'
    args = parser.parse_args()
    nets = build_model(args)
    ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=True, **nets)]
    for ckptio in ckptios:
        ckptio.load(args.resume_iter)
    transform = transforms.Compose([
        # rand_crop,
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    masks = rectanglemask(shape=256)
    masks = torch.from_numpy(masks)
    masks = masks.unsqueeze(0)
    masks = masks.to('cuda')
    samples, t = make_dataset(args.src_dir)
    N = 15
    M = masks.repeat(N, 1, 1, 1)
    y_trg_list = [torch.tensor(y).repeat(N).to(device)
                  for y in range(min(args.num_domains, 5))]
    z_trg_list = torch.randn(N, args.latent_dim).to(device)
    latent_dim = args.latent_dim
    os.makedirs(args.out_dir, exist_ok=True)
    nets.generator.eval()
    nets.style_encoder.eval()
    nets.mapping_network.eval()
    with torch.no_grad():
        for fn in samples:
            img = Image.open(fn).convert('RGB')
            img = transform(img)
            img = img.unsqueeze(0).to('cuda')
            m_img = img * (1 - masks)
            m_img = m_img.repeat(N, 1, 1, 1)
            z_trg_list = torch.randn(N, args.latent_dim).to(device)
            for i, y_trg in enumerate(y_trg_list):
                s_trg = nets.mapping_network(z_trg_list, y_trg)
                _, _, x_fake = nets.generator(m_img, s_trg, masks=M)
                x_fake = denormalize(x_fake)
                for out in range(15):
                    vutils.save_image(x_fake[out].cpu(), os.path.join(args.out_dir, '{}_{}_{}.png'.format(os.path.splitext(os.path.split(fn)[1])[0], i, out)))
