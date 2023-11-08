from core.model import CoarseNet
import torch.nn as nn
import cv2
import os

from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from glob import glob
import argparse
import torchvision.utils as vutils


def rectanglemask(shape):
    bbox = (64, 64, 128, 128)
    height = shape
    width = shape
    mask = np.zeros((height, width), np.float32)
    mask[(bbox[0]): (bbox[0] + bbox[2]), (bbox[1]): (bbox[1] + bbox[3])] = 1.
    return mask.reshape((1, ) + mask.shape).astype(np.float32)


def random_bbox(shape, margin, bbox_shape):
    """Generate a random tlhw with configuration.
    Args:
        config: Config should have configuration including IMG_SHAPES, VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
    Returns:
        tuple: (top, left, height, width)
    """
    img_height = shape
    img_width = shape
    height = bbox_shape
    width = bbox_shape
    ver_margin = margin
    hor_margin = margin
    maxt = img_height - ver_margin - height
    maxl = img_width - hor_margin - width
    t = np.random.randint(low=ver_margin, high=maxt)
    l = np.random.randint(low=hor_margin, high=maxl)
    h = height
    w = width
    return (t, l, h, w)


def hybrid(shape, max_angle=4, max_len=40, max_width=10, times=15, margin=10, bbox_shape=30):
    height = shape
    width = shape
    mask = np.zeros((height, width), np.float32)
    times = np.random.randint(times - 5, times)
    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)
        for j in range(5 + np.random.randint(10)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = max_len//4 + np.random.randint(max_len)
            brush_w = max_width//2 + np.random.randint(max_width)
            end_x = (start_x + length * np.sin(angle)).astype(np.int32)
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)
            cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
            start_x, start_y = end_x, end_y

    bboxs = []
    for i in range(times - 5):
        bbox = random_bbox(shape, margin, bbox_shape)
        bboxs.append(bbox)

    for bbox in bboxs:
        h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
        w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
        mask[(bbox[0] + h): (bbox[0] + bbox[2] - h), (bbox[1] + w): (bbox[1] + bbox[3] - w)] = 1.
    return mask.reshape((1,) + mask.shape).astype(np.float32)


def bbox2mask(shape, margin, bbox_shape, times):
        """Generate mask tensor from bbox.
        Args:
            bbox: configuration tuple, (top, left, height, width)
            config: Config should have configuration including IMG_SHAPES,
                MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.
        Returns:
            tf.Tensor: output with shape [1, H, W, 1]
        """
        bboxs = []
        for i in range(times):
            bbox = random_bbox(shape, margin, bbox_shape)
            bboxs.append(bbox)
        height = shape
        width = shape
        mask = np.zeros((height, width), np.float32)
        for bbox in bboxs:
            h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
            w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
            mask[(bbox[0] + h): (bbox[0] + bbox[2] - h), (bbox[1] + w): (bbox[1] + bbox[3] - w)] = 1.
        return mask.reshape((1, ) + mask.shape).astype(np.float32)


def radom_mask(shape, thre):
    height = shape
    width = shape
    mask = np.random.rand(height, width)
    mask[mask > thre] = 1
    mask[mask < thre] = 0
    return mask.reshape((1,) + mask.shape).astype(np.float32)


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x, ncol, filename):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


part_colors = torch.Tensor([[255, 0, 0], [255, 85, 0], [255, 170, 0],
                                [255, 0, 85], [255, 0, 170],
                                [0, 255, 0], [85, 255, 0], [170, 255, 0],
                                [0, 255, 85], [0, 255, 170],
                                [0, 0, 255], [85, 0, 255], [170, 0, 255],
                                [0, 85, 255], [0, 170, 255],
                                [255, 255, 0], [255, 255, 85], [255, 255, 170],
                                [255, 0, 255], [255, 85, 255], [255, 170, 255],
                                [0, 255, 255], [85, 255, 255], [170, 255, 255]])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')

    # directory for testing
    parser.add_argument('--checkpoints', type=str, default='pas/parsing_050000.ckpt',
                        help='File path for reload network checkpoints')
    parser.add_argument('--result_dir', type=str, default='out',
                        help='Directory for saving generated images and videos')
    parser.add_argument('--test_dir', type=str, default='data/val',
                        help='Directory containing input test images')
    # my mask
    parser.add_argument('--mask', type=str, default='rectangle',
                        choices=['hybrid', 'rectangle', 'bbox', 'random'],
                        help='mask type')
    parser.add_argument('--mask_ratio', type=float, default='0.5',
                        help='mask type')
    parser.add_argument('--margin', type=int, default=10, help='margin of image')
    parser.add_argument('--mask_num', type=int, default=20, help='number of mask')
    parser.add_argument('--bbox_shape', type=int, default=30, help='margin of image for bbox mask')
    parser.add_argument('--max_angle', type=int, default=4, help='parameter of angle for free form mask')
    parser.add_argument('--max_len', type=int, default=40, help='parameter of length for free form mask')
    parser.add_argument('--max_width', type=int, default=10, help='parameter of width for free form mask')
    args = parser.parse_args()
    transform = transforms.Compose([
        # rand_crop,
        transforms.Resize([args.img_size, args.img_size]), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    p_transform = transforms.Compose([
        # rand_crop,
        transforms.Resize([args.img_size, args.img_size], transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor()
    ])
    if args.mask == 'rectangle':
        masks = rectanglemask(shape=args.img_size)
    elif args.mask == 'hybrid':
        masks = hybrid(shape=args.img_size, max_angle=args.max_angle, max_len=args.max_len, max_width=args.max_width,
                       times=args.mask_num, margin=args.margin, bbox_shape=args.bbox_shape)
    elif args.mask == 'bbox':
        masks = bbox2mask(shape=args.img_size, margin=args.margin, bbox_shape=args.bbox_shape, times=args.mask_num)
    else:
        masks = radom_mask(args.img_size, args.mask_ratio)

    masks = torch.from_numpy(masks)
    masks = masks.unsqueeze(0)
    masks = masks.to('cuda')
    masks1 = torch.cat([masks, masks, masks])
    dim_in = 2 ** 14 // args.img_size
    c_net = nn.DataParallel(CoarseNet(dim_in))
    module_dict = torch.load(args.checkpoints)
    c_net.module.load_state_dict(module_dict, strict=False)
    os.makedirs('{}/input'.format(args.result_dir), exist_ok=True)
    os.makedirs('{}/pas'.format(args.result_dir), exist_ok=True)
    samples = glob('{}/*.jpg'.format(args.test_dir))
    samples.sort()
    c_net.eval()
    with torch.no_grad():
        for i in samples:
            image = Image.open(i).convert('RGB')
            image = image.resize((args.img_size, args.img_size))
            img = transform(image)
            img = img.unsqueeze(0).to('cuda')
            m_img = img * (1 - masks)
            # x_p = Image.open(str(i).replace('val', 'val_parsing', 1)[:-3] + 'png')
            # x_p = x_p.resize((256, 256), Image.NEAREST)
            # x_p = p_transform(x_p)

            out = c_net(m_img, masks).cpu().argmax(1)
            x_concat = [img, masks1, m_img]
            x_concat = torch.cat(x_concat, dim=0)
            save_image(x_concat, '{}/input/{}.png'.format(args.result_dir, str(i).split('/')[-1][:-4]))
            N, H, W = out.size()
            out_parsing_color = torch.zeros((1, 3, H, W)) + 255
            num_of_class = int(out.max()) + 1
            # x_p_color = torch.zeros((16, 3, 256, 256)) + 255
            for pi in range(1, num_of_class):
                index = np.where(out == pi)
                out_parsing_color[index[0], :, index[1], index[2]] = part_colors[pi]
                # index_src = np.where(x_p == pi)
                # x_p_color[index_src[0], :, index_src[1], index_src[2]] = part_colors[pi]
            #         out_parsing_color = out_parsing_color / 255
            #         vutils.save_image(out_parsing_color.cpu(), 'out/me/pas/{}.png'.format(i))
            vis_parsing_anno_color = out_parsing_color[0].numpy().astype(np.uint8)
            # x_p_anno_color = x_p_color[0].numpy().astype(np.uint8)
            vis_im = np.array(image).astype(np.uint8)
            vis_parsing_anno_color = vis_parsing_anno_color.transpose(1, 2, 0)
            # x_p_anno_color = x_p_anno_color.transpose(1, 2, 0)
            vis_i1 = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
            # vis_pas = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, x_p_anno_color, 0.6, 0)
            cv2.imwrite('{}/pas/{}.png'.format(args.result_dir, str(i).split('/')[-1][:-4]), vis_i1,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # cv2.imwrite('out/pas/t-{}.png'.format(str(i).split('/')[-1][:-4]), vis_pas,
            #             [int(cv2.IMWRITE_JPEG_QUALITY), 100])
