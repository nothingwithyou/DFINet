import os
import argparse

from munch import Munch
import torch

from core.data_loader import get_train_loader
from core.data_loader import get_test_loader
from core.model import CoarseNet
from core.data_loader import InputFetcher
import torch.nn as nn
from tqdm import *
import numpy as np
import torchvision.utils as vutils


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.criteria = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


def compute_coarse_loss(nets, args, x_real, x_p, masks, LossP):
    x_mask = x_real * (1-masks)
    x_p = torch.squeeze(x_p, 1)
    out = nets(x_mask, masks)
    parsing_Loss = LossP(out, x_p.long())
    return parsing_Loss


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


part_colors = torch.Tensor([[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]])


def test_out_put(x_real, x_p, y_org, mask, step):
    x_mask = x_real * (1-mask)
    x_p = torch.squeeze(x_p, 1)
    out = net(x_mask, mask).cpu().argmax(1)
    N, H, W = out.size()
    out_parsing_color = torch.zeros((N, 3, H, W)) + 255
    x_p_color = torch.zeros((N, 3, H, W)) + 255
    x_p = x_p.cpu()
    x_real = denormalize(x_real.cpu())
    x_mask = denormalize(x_mask.cpu())
    num_of_class = int(out.max()) + 1
    for pi in range(1, num_of_class):
        index = np.where(out == pi)
        out_parsing_color[index[0], :, index[1], index[2]] = part_colors[pi]
        index_src = np.where(x_p == pi)
        x_p_color[index_src[0], :, index_src[1], index_src[2]] = part_colors[pi]
    x_p_color = x_p_color / 255
    out_parsing_color = out_parsing_color / 255
    x_concat = torch.cat([x_real, x_mask, x_p_color, out_parsing_color], dim=0)
    vutils.save_image(x_concat, 'pas/{}.png'.format(step), nrow=N, padding=0)


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

# weight for objective functions
parser.add_argument('--lambda_reg', type=float, default=2,
                    help='Weight for R1 regularization')
parser.add_argument('--lambda_cyc', type=float, default=2,
                    help='Weight for cyclic consistency loss')
parser.add_argument('--lambda_sty', type=float, default=1,
                    help='Weight for style reconstruction loss')
parser.add_argument('--lambda_ds', type=float, default=1,
                    help='Weight for diversity sensitive loss')
parser.add_argument('--lambda_pl1', type=float, default=1, help='the parameter of parsing L1Loss')
parser.add_argument('--lambda_il1', type=float, default=1, help='the parameter of image L1Loss')
parser.add_argument('--lambda_perceptual', type=float, default=2,
                    help='the parameter of FML1Loss (perceptual loss)')
parser.add_argument('--lambda_gan', type=float, default=1,
                    help='the parameter of valid loss of AdaReconL1Loss; 0 is recommended')
parser.add_argument('--s_loss', type=float, default=2.5, help='STYLE_LOSS_WEIGHT')
parser.add_argument('--p_loss', type=float, default=0.1, help='CONTENT_LOSS_WEIGHT')
parser.add_argument('--ds_iter', type=int, default=50000,
                    help='Number of iterations to optimize diversity sensitive loss')
parser.add_argument('--w_hpf', type=float, default=1,
                    help='weight for high-pass filtering')

# training arguments
parser.add_argument('--randcrop_prob', type=float, default=0.5,
                    help='Probabilty of using random-resized cropping')
parser.add_argument('--total_iters', type=int, default=200000,
                    help='Number of total iterations')
parser.add_argument('--resume_iter', type=int, default=0,
                    help='Iterations to resume training/testing')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for training')
parser.add_argument('--val_batch_size', type=int, default=32,
                    help='Batch size for validation')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='Learning rate for D, E and G')
parser.add_argument('--f_lr', type=float, default=1e-4,
                    help='Learning rate for F')
parser.add_argument('--beta1', type=float, default=0.0,
                    help='Decay rate for 1st moment of Adam')
parser.add_argument('--beta2', type=float, default=0.99,
                    help='Decay rate for 2nd moment of Adam')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay for optimizer')
parser.add_argument('--num_outs_per_domain', type=int, default=10,
                    help='Number of generated images per domain during sampling')

# misc
parser.add_argument('--num_workers', type=int, default=40,
                    help='Number of workers used in DataLoader')
parser.add_argument('--seed', type=int, default=777,
                    help='Seed for random number generator')

# directory for training
parser.add_argument('--train_img_dir', type=str, default='data/train',
                    help='Directory containing training images')
parser.add_argument('--mask_dir', type=str, default='data/mask',
                    help='Directory containing mask images')
parser.add_argument('--from_mdir', type=bool, default=False,
                    help='Train from the out dir')
parser.add_argument('--parsing_dir', type=str, default='data/train_par',
                    help='Directory containing mask images')
parser.add_argument('--val_img_dir', type=str, default='data/val',
                    help='Directory containing validation images')
parser.add_argument('--sample_dir', type=str, default='samples',
                    help='Directory for saving generated images')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                    help='Directory for saving network checkpoints')

# directory for calculating metrics
parser.add_argument('--eval_dir', type=str, default='eval',
                    help='Directory for saving metrics, i.e., FID and LPIPS')

# directory for testing
parser.add_argument('--result_dir', type=str, default='results',
                    help='Directory for saving generated images and videos')
parser.add_argument('--src_dir', type=str, default='assets/representative/celeba_hq/src',
                    help='Directory containing input source images')
parser.add_argument('--ref_dir', type=str, default='assets/representative/celeba_hq/ref',
                    help='Directory containing input reference images')
parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female',
                    help='input directory when aligning faces')
parser.add_argument('--out_dir', type=str, default='assets/representative/celeba_hq/src/female',
                    help='output directory when aligning faces')


if __name__ == '__main__':
    args = parser.parse_args()
    pre_loaders = Munch(src=get_train_loader(root=args.train_img_dir,
                                             which='source',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers,
                                             pre=True),
                        val=get_test_loader(root=args.val_img_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers))
    dim_in = 2 ** 14 // args.img_size
    net = nn.DataParallel(CoarseNet(dim_in))
    module_dict = torch.load('pas/parsing.ckpt')
    net.module.load_state_dict(module_dict, strict=False)
    optim = torch.optim.Adam(
        params=net.parameters(),
        lr=args.f_lr if net == 'mapping_network' else args.lr,
        betas=[args.beta1, args.beta2],
        weight_decay=args.weight_decay)
    fetcher = InputFetcher(pre_loaders.src, latent_dim=args.latent_dim, mode='test', pre=True)
    net.train()
    n_min = args.batch_size * args.img_size * args.img_size // 16
    LossP = OhemCELoss(thresh=0.7, n_min=n_min)
    with open('1.txt', 'w') as f:
        with tqdm(total=100000, ncols=100) as _tqdm:
            for i in range(0, 100000):
                inputs = next(fetcher)
                x_real, x_p, y_org, mask = inputs.x, inputs.p, inputs.y, inputs.mask
                c_loss = compute_coarse_loss(net, args, x_real, x_p, mask, LossP)
                optim.zero_grad()
                c_loss.backward()
                optim.step()
                if (i + 1) % 10 == 0:
                    _tqdm.set_postfix(loss='{:.4f}'.format(c_loss))
                    f.write('{:.4f}'.format(c_loss))
                    _tqdm.update(10)
                if (i + 1) % 5000 == 0:
                    with torch.no_grad():
                        inputs = next(fetcher)
                        x_real, x_p, y_org, mask = inputs.x, inputs.p, inputs.y, inputs.mask
                        test_out_put(x_real, x_p, y_org, mask, i + 1)
                    torch.save(net.module.state_dict(), 'pas/parsing_{:06}.ckpt'.format(i + 1))