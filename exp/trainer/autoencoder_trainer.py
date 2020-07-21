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

from detectron2.data import (
  get_detection_dataset_dicts,
  DatasetFromList, DatasetMapper, MapDataset, samplers,
)
from template_lib.d2.data import build_dataset_mapper

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

        ae_loss = F.mse_loss(x_recon, images)

        self.encoder_decoder_optim.zero_grad()
        ae_loss.backward()
        self.encoder_decoder_optim.step()

        if iteration % self.iter_every_epoch == 0:
            summary_dict = {}
            summary_dict['ae_loss'] = ae_loss.item()
            summary_dict2txtfig(summary_dict, prefix='train', step=iteration,
                                textlogger=self.myargs.textlogger, save_fig_sec=60)

            num_img = 6
            saved_img = torch.cat(
                [images[:num_img].unsqueeze(1), x_recon[:num_img].unsqueeze(1)], dim=1)\
                .view(-1, *images.shape[1:])
            saved_img_path = os.path.join(self.myargs.args.imgdir, f'{iteration:08}.png')
            save_image(saved_img, saved_img_path, nrow=2, normalize=True)
        pass