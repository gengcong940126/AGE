from itertools import chain
import copy
import functools
import logging
import os

import tqdm
import collections
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np

from detectron2.structures import ImageList
from detectron2.utils import comm
from detectron2.utils.comm import get_world_size
from detectron2.utils.events import get_event_storage
from detectron2.checkpoint import Checkpointer, DetectionCheckpointer

from template_lib.d2.distributions.fairnas_noise_sample import fairnas_repeat_tensor
from template_lib.d2.distributions import build_d2distributions
from template_lib.d2.models import build_d2model
from template_lib.trainer.base_trainer import Trainer
from template_lib.trainer import get_ddp_attr
from template_lib.gans import inception_utils, gan_utils, gan_losses, GANLosses
from template_lib.gans.networks import build_discriminator, build_generator
from template_lib.gans.evaluation import get_sample_imgs_list_ddp
from template_lib.d2.optimizer import build_optimizer
from template_lib.utils import modelarts_utils
from template_lib.gans.evaluation import build_GAN_metric_dict
from template_lib.gans.evaluation.fid_score import FIDScore
from template_lib.gans.models import build_GAN_model
from template_lib.utils import get_eval_attr, print_number_params, get_attr_kwargs, get_attr_eval
from template_lib.trainer.base_trainer import summary_dict2txtfig

from template_lib.d2template.trainer import TRAINER_REGISTRY, BaseTrainer
from template_lib.d2template.models import build_model
from mymodels.model256 import normalize

from detectron2.data import (
  get_detection_dataset_dicts,
  DatasetFromList, DatasetMapper, MapDataset, samplers,
)
from template_lib.d2.data import build_dataset_mapper


def hinge_loss_discriminator(r_logit, f_logit):
    r_logit_mean = r_logit.mean()
    f_logit_mean = f_logit.mean()

    loss_real = torch.mean(F.relu(1. - r_logit))
    loss_fake = torch.mean(F.relu(1. + f_logit))
    D_loss = loss_real + loss_fake
    return r_logit_mean, f_logit_mean, D_loss
def hinge_loss_generator2(f_logit):
    f_logit_mean = f_logit.mean()
    G_loss = - f_logit_mean
    return f_logit_mean, G_loss


@TRAINER_REGISTRY.register()
class AutoEncoderTrainer(BaseTrainer):

    def __init__(self, cfg, myargs, iter_every_epoch, **kwargs):
      super().__init__(cfg, myargs, iter_every_epoch)

      self.ngpu              = get_attr_kwargs(cfg.trainer, 'ngpu', **kwargs)

      self.build_models(cfg=cfg)
      self.to(self.device)

    def build_models(self, **kwargs):
      self.models = {}
      self.optims = {}

      self.encoder = build_model(self.cfg.encoder, ngpu=self.ngpu)
      self.models['encoder'] = self.encoder

      self.decoder = build_model(self.cfg.decoder, ngpu=self.ngpu)
      self.models['decoder'] = self.decoder

      params = chain(self.encoder.parameters(), self.decoder.parameters())
      optim_cfg = self.cfg.optimizer
      self.encoder_decoder_optim = optim.Adam(
          params, lr=optim_cfg.lr, betas=(optim_cfg.beta1, optim_cfg.beta2))
      self.optims['encoder_decoder_optim'] = self.encoder_decoder_optim

      print(self)
      self._print_number_params(self.models)

    def train_func(self, data, iteration, pbar):
        """Perform architecture search by training a controller and shared_cnn.
        """
        if comm.is_main_process() and iteration % self.iter_every_epoch == 0:
            pbar.set_postfix_str(s="BaseTrainer ")

        images, labels = self._preprocess_image(data)
        images = images.tensor

        self.encoder.train()
        self.decoder.train()

        z_code = self.encoder(images)
        x_recon = self.decoder(z_code)

        # ae_loss = F.mse_loss(x_recon, images)
        ae_loss = F.l1_loss(x_recon, images)

        self.encoder_decoder_optim.zero_grad()
        ae_loss.backward()
        self.encoder_decoder_optim.step()

        if iteration % self.iter_every_epoch == 0:
            summary_dict = {}
            summary_dict['ae_loss'] = ae_loss.item()
            summary_dict2txtfig(summary_dict, prefix='train', step=iteration,
                                textlogger=self.myargs.textlogger, save_fig_sec=60)

            if iteration % (5*self.iter_every_epoch) == 0:
                num_img = 6
                saved_img = torch.cat(
                    [images[:num_img].unsqueeze(1), x_recon[:num_img].unsqueeze(1)], dim=1)\
                    .view(-1, *images.shape[1:])
                saved_img_path = os.path.join(self.myargs.args.imgdir, f'{iteration:08}.png')
                save_image(saved_img, saved_img_path, nrow=2, normalize=True)
        pass

@TRAINER_REGISTRY.register()
class AEGAN_acaiTrainer(BaseTrainer):

    def __init__(self, cfg, myargs, iter_every_epoch, **kwargs):
      super().__init__(cfg, myargs, iter_every_epoch)

      self.ngpu              = get_attr_kwargs(cfg.trainer, 'ngpu', **kwargs)
      self.updates_d         = get_attr_kwargs(cfg.trainer, 'updates_d', **kwargs)
      self.updates_D         = get_attr_kwargs(cfg.trainer, 'updates_D', **kwargs)
      self.reg               = get_attr_kwargs(cfg.trainer, 'reg', **kwargs)
      self.noise             = get_attr_kwargs(cfg.trainer, 'noise', **kwargs)
      self.embedding         =get_attr_kwargs(cfg.encoder, 'embedding', **kwargs)
      self.batch_size        =get_attr_kwargs(cfg.start, 'IMS_PER_BATCH', **kwargs)//2
      self.emb_channel       =get_attr_kwargs(cfg.encoder, 'emb_channel', **kwargs)
      self.build_models(cfg=cfg)
      self.to(self.device)

    def build_models(self, **kwargs):
      self.models = {}
      self.optims = {}
      self.z= torch.FloatTensor(self.batch_size, self.emb_channel, 16, 16).cuda()
      self.encoder = build_model(self.cfg.encoder, ngpu=self.ngpu)
      self.models['encoder'] = self.encoder

      self.decoder = build_model(self.cfg.decoder, ngpu=self.ngpu)
      self.models['decoder'] = self.decoder

      self.netd = build_model(self.cfg.netd, ngpu=self.ngpu)
      self.models['netd'] = self.netd

      self.netD = build_model(self.cfg.netD, ngpu=self.ngpu)
      self.models['netD'] = self.netD

      self.netg = build_model(self.cfg.netg, ngpu=self.ngpu)
      self.models['netg'] = self.netg

      params_AE = chain(self.encoder.parameters(), self.decoder.parameters())
      params_d = self.netd.parameters()
      params_D = self.netD.parameters()
      params_g = self.netg.parameters()

      optim_cfg = self.cfg.optimizer
      self.encoder_decoder_optim = optim.Adam(
          params_AE, lr=optim_cfg.lr, betas=(optim_cfg.beta1, optim_cfg.beta2))
      self.netd_optim = optim.Adam(
          params_d, lr=optim_cfg.lr, betas=(optim_cfg.beta1, optim_cfg.beta2))
      self.netD_optim = optim.Adam(
          params_D, lr=optim_cfg.lr, betas=(optim_cfg.beta1, optim_cfg.beta2))
      self.netg_optim = optim.Adam(
          params_g, lr=optim_cfg.lr, betas=(optim_cfg.beta1, optim_cfg.beta2))
      self.optims['encoder_decoder_optim'] = self.encoder_decoder_optim
      self.optims['netd_optim'] = self.netd_optim
      self.optims['netD_optim'] = self.netD_optim
      self.optims['netg_optim'] = self.netg_optim
      print(self)
      self._print_number_params(self.models)

    def train_func(self, data, iteration, pbar):
        """Perform architecture search by training a controller and shared_cnn.
        """
        if comm.is_main_process() and iteration % self.iter_every_epoch == 0:
            pbar.set_postfix_str(s="BaseTrainer ")

        images, labels = self._preprocess_image(data)
        images = images.tensor
        list_images = torch.chunk(images, dim=0, chunks=2)
        images1 = list_images[0]
        images2 = list_images[1]

        self.encoder.train()
        self.decoder.train()
        self.netd.train()
        self.netD.train()
        self.netg.train()

        encode1 = self.encoder(images1)
        encode2 = self.encoder(images2)
        alpha = torch.rand(images1.shape[0], 1,1,1).cuda()
        alpha = 0.5 - torch.abs(alpha - 0.5)  # Make interval [0, 0.5]
        encode_mix = alpha * encode1 + (1 - alpha) * encode2
        if self.embedding == 'sphere':
            encode_mix = normalize(encode_mix,dim=[2,3])
        x_alpha = self.decoder(encode_mix)
        AE = self.decoder(encode1)
        self.z.data.normal_(0, 1)
        if self.noise == 'sphere':
            normalize(self.z.data,dim=[2,3])
        xz = self.netd(self.decoder(self.netg(self.z)).detach())
        loss_disc = torch.mean((self.netd(x_alpha.detach()) - alpha.squeeze() - self.reg).pow(2))\
                    + torch.mean((xz - 0.7).pow(2))
        loss_disc_real = torch.mean((self.netd(AE.detach() + self.reg * (images1 - AE.detach())) - self.reg).pow(2)) \
                         + torch.mean(self.netd(images1).pow(2))
        # loss_ae_disc = torch.mean(torch.square(netd(x_alpha)))
        d_loss = loss_disc + loss_disc_real
        self.netd.zero_grad()
        d_loss.backward()
        self.netd_optim.step()

        if iteration % self.updates_d == 0:
            loss_ae_disc = torch.mean((self.netd(x_alpha)).pow(2))
            loss_ae_real = torch.mean(self.netd(AE).pow(2))
            err =F.l1_loss(AE, images1)
            AE_loss = err + loss_ae_disc + loss_ae_real
            self.encoder_decoder_optim.zero_grad()
            AE_loss.backward()
            self.encoder_decoder_optim.step()

        encode1 = self.encoder(images1)
        D_real = self.netD(encode1)
        D_fake = self.netD(self.netg(self.z).detach())
        r_logit_mean, f_logit_mean, D_loss = hinge_loss_discriminator(r_logit=D_real, f_logit=D_fake)
        self.netD.zero_grad()
        D_loss.backward()
        self.netD_optim.step()

        if iteration % self.updates_D == 0:
            g = self.netg(self.z)
            g_fake = self.netD(g)
            fake=self.decoder(g)
            gx_fake = self.netd(fake)
            G_f_logit_mean, g_fake_loss = hinge_loss_generator2(f_logit=g_fake)
            loss_recon = torch.mean(gx_fake.pow(2))
            g_loss = g_fake_loss  + loss_recon
            self.netg.zero_grad()
            g_loss.backward()
            self.netg_optim.step()
        if iteration % self.iter_every_epoch == 0:
            summary_dict = {}
            if iteration % self.updates_d == 0:
                summary_dict['AE_loss'] = AE_loss.item()
            summary_dict['d_loss'] = d_loss.item()
            summary_dict['D_loss'] = D_loss.item()
            if iteration % self.updates_D == 0:
                summary_dict['g_loss'] = D_loss.item()
            summary_dict2txtfig(summary_dict, prefix='train', step=iteration,
                                textlogger=self.myargs.textlogger, save_fig_sec=60)

        if iteration % (5*self.iter_every_epoch) == 0:
            num_img = 6
            saved_img = torch.cat(
                [images1[:num_img].unsqueeze(1), AE[:num_img].unsqueeze(1)], dim=1)\
                .view(-1, *images1.shape[1:])
            saved_img_path = os.path.join(self.myargs.args.imgdir, f'recon{iteration:08}.png')
            save_image(saved_img, saved_img_path, nrow=2, normalize=True)
            if iteration % self.updates_D == 0:
                saved_img2 = fake .view(-1, *images1.shape[1:])
                saved_img_path = os.path.join(self.myargs.args.imgdir, f'fake{iteration:08}.png')
                save_image(saved_img2, saved_img_path, nrow=8, normalize=True)
        pass