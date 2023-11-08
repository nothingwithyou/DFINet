"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join as ospj
import time
import datetime
from munch import Munch


import torch.nn as nn
import torch.nn.functional as F

from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
from metrics.eval import calculate_metrics
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm


class Solver(nn.Module):
    def __init__(self, args, pre=False):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.Perceptual = PerceptualLoss()
        # self.Style = StyleLoss()
        self.nets = build_model(args, pre=pre)
        self.loss_net = build_loss_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)

        if args.mode == 'train' or args.mode == 'pre_train' or args.mode == 'train_1':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'fan':
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]

        else:
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets)]

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def pre_train(self, loaders):

        args = self.args
        nets = self.nets
        optims = self.optims
        loss_net = self.loss_net
        writer = SummaryWriter(args.logs_dir)

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, latent_dim=args.latent_dim, mode='test', pre=True)
        fetcher_val = InputFetcher(loaders.val, None, latent_dim=args.latent_dim, mode='test', pre=True)
        inputs_val = next(fetcher_val)

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)
            # self.ckptios[0].load(args.resume_iter)
        else:
            pre_model = torch.load('./checkpoints/parsing.ckpt')
            state_dict = {'coarse.' + k: v for k, v in pre_model.items()}
            nets.generator.module.load_state_dict(state_dict, strict=False)

        # remember the initial value of ds weight
        # initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            x_real, x_p, y_org, mask = inputs.x, inputs.p, inputs.y, inputs.mask

            # train the discriminator
            d_loss, d_losses_self = compute_d_loss(
                nets, args, x_real, y_org, y_trg=y_org, masks=mask, pre=True)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            # train the generator

            g_loss, g_losses_self = compute_g_loss_pre(nets, loss_net, args, x_real, x_p, y_org,  masks=mask)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()

            # decay weight for diversity sensitive loss
            # if args.lambda_ds > 0:
            #     args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses_self, g_losses_self],
                                        ['D/latent_', 'G/latent_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                # all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)
                try:
                    for key, value in all_losses.items():
                        writer.add_scalar(key, value, i)
                except:
                    pass

            # generate images for debugging
            if (i+1) % args.sample_every == 0:
                with torch.no_grad():
                    os.makedirs(args.sample_dir, exist_ok=True)
                    utils.save_sample_png(nets, args, inputs=inputs_val, step=i + 1)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)

    def train_1(self, loaders):
        args = self.args
        nets = self.nets
        optims = self.optims
        loss_net = self.loss_net

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, latent_dim=args.latent_dim, mode='test', pre=True)
        fetcher_val = InputFetcher(loaders.val, None, latent_dim=args.latent_dim, mode='test', pre=True)
        inputs_val = next(fetcher_val)
        writer = SummaryWriter(args.logs_dir)
        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)
        else:
            state_dict = torch.load('./checkpoints/pre.ckpt')
            state_dict['generator'] = {k: v for k, v in state_dict['generator'].items() if k[:6] != 'decode' and k[:6] != 'to_rgb' and k[:7] != 'out_put'}
            # nets.style_encoder.module.load_state_dict(state_dict['style_encoder'], strict=False)
            nets.generator.module.load_state_dict(state_dict['generator'], strict=False)

        # remember the initial value of ds weight
        # initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            x_real, x_p, y_org, mask = inputs.x, inputs.p, inputs.y, inputs.mask

            # train the discriminator
            d_loss, d_losses_self = compute_d_loss(
                nets, args, x_real, y_org, x_ref=x_real, y_trg=y_org, masks=mask)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()


            # train the generator

            g_loss, g_losses_self = compute_g_loss_pre(nets, loss_net, args, x_real, x_p, y_org, masks=mask)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.style_encoder.step()

            # decay weight for diversity sensitive loss
            # if args.lambda_ds > 0:
            #     args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # print out log info
            if (i + 1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i + 1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses_self, g_losses_self],
                                        ['D/latent_', 'G/latent_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                # all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.5f]' % (key, value) for key, value in all_losses.items()])
                print(log)
                try:
                    for key, value in all_losses.items():
                        writer.add_scalar(key, value, i)
                except:
                    pass

            # generate images for debugging
            if (i + 1) % args.sample_every == 0:
                with torch.no_grad():
                    os.makedirs(args.sample_dir, exist_ok=True)
                    utils.save_sample_png(nets, args, inputs=inputs_val, step=i + 1)

            # save model checkpoints
            if (i + 1) % args.save_every == 0:
                self._save_checkpoint(step=i + 1)

    def train_(self, loaders):
        args = self.args
        nets = self.nets
        optims = self.optims
        loss_net = self.loss_net
        import torch
        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train')
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
        inputs_val = next(fetcher_val)
        writer = SummaryWriter('runs/experiment_3')
        # resume training if necessary
        if args.resume_iter > 0:
            # self.ckptios[0].load(args.resume_iter)
            self._load_checkpoint(args.resume_iter)
            # state_dict = torch.load('./checkpoints/pre.ckpt')
            # state_dict['generator'] = {k: v for k, v in state_dict['generator'].items() if k[:6] != 'decode' and k[:6] != 'to_rgb' and k[:7] != 'out_put'}
            # nets.generator.module.load_state_dict(state_dict['generator'], strict=False)
            # pre_model = torch.load('./checkpoints/parsing.ckpt')
            # state_dict = {'coarse.' + k: v for k, v in pre_model.items()}
            # nets.generator.module.load_state_dict(state_dict, strict=False)
        else:
            state_dict = torch.load('./checkpoints/pre_1.ckpt')
            nets.style_encoder.module.load_state_dict(state_dict['style_encoder'], strict=False)
            nets.generator.module.load_state_dict(state_dict['generator'], strict=False)
        import torch._dynamo
        torch._dynamo.reset()
        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            x_real, p_real, y_org, mask = inputs.x_src, inputs.p_src, inputs.y_src, inputs.mask
            x_ref, p_ref, x_ref2, p_ref2, y_trg = inputs.x_ref, inputs.p_ref, inputs.x_ref2, inputs.p_ref2, inputs.y_ref
            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

            # masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

            # train the discriminator
            d_loss, d_losses_latent = compute_d_loss(
                nets, args, x_real, y_org, y_trg, z_trg=z_trg, masks=mask)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            d_loss, d_losses_ref = compute_d_loss(
                nets, args, x_real, y_org, y_trg, x_ref=x_ref, masks=mask)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            # train the generator
            g_loss, g_losses_latent = compute_g_loss(
                nets, loss_net, args, x_real, p_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=mask)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()
            optims.style_encoder.step()

            g_loss, g_losses_ref = compute_g_loss(
                nets, loss_net, args, x_real, p_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=mask)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            # optims.style_encoder.step()

            # decay weight for diversity sensitive loss
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                        ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.5f]' % (key, value) for key, value in all_losses.items()])
                print(log)
                try:
                    for key, value in all_losses.items():
                        writer.add_scalar(key, value, i)
                except:
                    pass

            # generate images for debugging
            if (i+1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)
                utils.debug_image(nets, args, inputs=inputs_val, step=i+1)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)

            # compute FID and LPIPS if necessary
            if (i+1) % args.eval_every == 0:
                pass
                # calculate_metrics(nets, args, i+1, mode='latent')
                # calculate_metrics(nets, args, i+1, mode='reference')

    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        nets = self.nets
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = InputFetcher(loaders.src, None, args.latent_dim, 'test')
        # ref = InputFetcher(loaders.ref, None, args.latent_dim, 'test')
        os.makedirs(args.sample_dir, exist_ok=True)
        for i in range(len(loaders.src)):
            inputs = next(src)
            utils.lat_image(nets, args, inputs=inputs, step=i + 1)

        #　utils.translate_using_reference(nets, args, src.x, ref.x, ref.y, fname)

        # fname = ospj(args.result_dir, 'video_ref.mp4')
        # print('Working on {}...'.format(fname))
        # utils.video_ref(nets, args, src.x, ref.x, ref.y, fname)

    @torch.no_grad()
    def evaluate(self):
        args = self.args
        nets = self.nets
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets, args, step=resume_iter, mode='latent')
        calculate_metrics(nets, args, step=resume_iter, mode='reference')


def compute_d_loss(nets, args, x_real, y_org, y_trg=None, z_trg=None, x_ref=None, masks=None, pre=False):
    if not pre:
        assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.requires_grad_()
    out = nets.discriminator(x_real, masks, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    # with fake images
    with torch.no_grad():
        x_mask = x_real * (1 - masks)
        if pre:
            _, _, x_fake = nets.generator(x_mask, None, masks)
        else:
            if z_trg is not None:
                s_trg = nets.mapping_network(z_trg, y_trg)
            else:  # x_ref is not None
                s_trg = nets.style_encoder(x_ref, masks, y_trg)
            _,  _, x_fake = nets.generator(x_mask, s_trg, masks=masks)
    out = nets.discriminator(x_fake, masks, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())


def compute_g_loss_pre(nets, loss_net, args, x_real, x_p, y_org, masks=None):
    x_mask = x_real * (1-masks)
    # s_org = None
    s_org = nets.style_encoder(x_mask, masks, y_org)
    first_out, out_pas, x_fake = nets.generator(x_mask, s_org, masks=masks)
    out = nets.discriminator(x_fake, masks, y_org)
    loss_adv = adv_loss(out, 1)

    # parsing loss
    x_p = torch.squeeze(x_p, 1)
    # parsing_Loss = loss_net.LossP(first_out, x_p.long()) + loss_net.LossP(out_pas, x_p.long())
    parsing_Loss = loss_net.LossP(out_pas, x_p.long())
    # L1 Loss
    out_L1Loss = F.l1_loss(x_fake, x_real)

    # perceptual loss and style loss
    perceptual_loss = loss_net.Perceptual(x_fake, x_real)
    style_loss = loss_net.Style(x_fake * masks, x_real * masks)

    loss = loss_adv + args.lambda_pl1 * parsing_Loss + args.lambda_il1 * out_L1Loss +\
           args.p_loss * perceptual_loss + args.s_loss * style_loss

    return loss, Munch(adv=loss_adv.item(),
                       ps=parsing_Loss.item(),
                       l1=out_L1Loss.item(),
                       content=perceptual_loss.item(),
                       style=style_loss.item())


def compute_g_loss_lat(nets, loss_net, args, x_real, x_p, y_org, masks=None):
    x_mask = x_real * (1-masks)
    z_trg = torch.randn(x_mask.size(0), args.latent_dim)
    z_trg2 = torch.randn(x_mask.size(0), args.latent_dim)
    s_org = nets.mapping_network(z_trg, y_org)
    s_org2 = nets.mapping_network(z_trg2, y_org)
    first_out, out_pas, x_fake = nets.generator(x_mask, s_org, masks=masks)
    first_out2, out_pas2, x_fake2 = nets.generator(x_mask, s_org2, masks=masks)
    out = nets.discriminator(x_fake, masks, y_org)
    out2 = nets.discriminator(x_fake2, masks, y_org)
    loss_adv = (adv_loss(out, 1) + adv_loss(out2, 1)) / 2

    # parsing loss
    x_p = torch.squeeze(x_p, 1)
    # parsing_Loss = loss_net.LossP(first_out, x_p.long()) + loss_net.LossP(out_pas, x_p.long())
    parsing_Loss = (loss_net.LossP(out_pas, x_p.long()) + loss_net.LossP(out_pas2, x_p.long())) / 2
    # L1 Loss
    out_L1Loss = (F.l1_loss(x_fake, x_real) + F.l1_loss(x_fake2, x_real)) / 2

    # perceptual loss and style loss
    perceptual_loss = (loss_net.Perceptual(x_fake, x_real) + loss_net.Perceptual(x_fake2, x_real)) / 2
    style_loss = (loss_net.Style(x_fake * masks, x_real * masks) + loss_net.Style(x_fake2 * masks, x_real * masks)) / 2

    loss = loss_adv + args.lambda_pl1 * parsing_Loss + args.lambda_il1 * out_L1Loss +\
           args.p_loss * perceptual_loss + args.s_loss * style_loss

    return loss, Munch(adv=loss_adv.item(),
                       ps=parsing_Loss.item(),
                       l1=out_L1Loss.item(),
                       content=perceptual_loss.item(),
                       style=style_loss.item())


def compute_g_loss(nets, loss_net, args, x_real, x_p, y_org, y_trg, z_trgs=None, x_refs=None, masks=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, masks, y_trg)

    x_mask = x_real * (1 - masks)
    first_out, out_pas, x_fake = nets.generator(x_mask, s_trg, masks=masks)
    out = nets.discriminator(x_fake, masks, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, masks, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, masks, y_trg)
    p2, _, x_fake2 = nets.generator(x_mask, s_trg2, masks=masks)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency loss
    # masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else N  one
    s_org = nets.style_encoder(x_real, masks, y_org)
    x_fake_mask = x_fake * (1 - masks)
    sec_out, _, x_rec = nets.generator(x_fake_mask, s_org, masks=masks)
    loss_cyc = args.lambda_il1 * F.l1_loss(x_rec, x_real) + args.s_loss * loss_net.Style(x_rec*masks, x_real*masks) + \
               args.p_loss * loss_net.Perceptual(x_fake, x_real)

    # parsing loss
    x_p = torch.squeeze(x_p, 1)
    parsing_Loss = (loss_net.LossP(first_out, x_p.long()) + loss_net.LossP(out_pas, x_p.long())) / 2
    # parsing_Loss = loss_net.LossP(out_pas, x_p.long())
    # perceptual loss and style loss
    out_L1Loss = F.l1_loss(x_fake*(1-masks), x_real*(1-masks))
    # style_loss = loss_net.Style(x_fake*masks, x_real*masks)

    loss = loss_adv + args.lambda_sty * loss_sty - args.lambda_ds * loss_ds + args.lambda_il1 * out_L1Loss + \
           args.lambda_cyc * loss_cyc + args.lambda_pl1 * parsing_Loss #  args.s_loss * style_loss
    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item(),
                       cyc=loss_cyc.item(),
                       l1=out_L1Loss.item(),
                       ps=parsing_Loss.item(),
                       # content=perceptual_loss.item(),
                       # style=style_loss.item(),
                       total=loss.item()
                       )


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


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    targets = torch.full_like(logits, fill_value=target)
    # loss = F.binary_cross_entropy_with_logits(logits, targets)
    loss = F.mse_loss(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


class PerceptualLoss(nn.Module):
    """
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=None):
        super(PerceptualLoss, self).__init__()
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return content_loss


class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss


class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(weights='IMAGENET1K_V1').features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


def build_loss_model(args):
    n_min = args.batch_size * args.img_size * args.img_size // 16
    Perceptual = nn.DataParallel(PerceptualLoss())
    Style = nn.DataParallel(StyleLoss())
    nets = Munch(Perceptual=Perceptual,
                 Style=Style,
                 LossP=OhemCELoss(thresh=0.7, n_min=n_min))

    return nets
