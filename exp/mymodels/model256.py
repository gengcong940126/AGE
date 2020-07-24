import torch.nn.functional as F
import torch
import torch.nn as nn

from template_lib.utils import get_attr_kwargs
from template_lib.d2template.models import MODEL_REGISTRY


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Reshape(nn.Module):
  def __init__(self, *args):
    super(Reshape, self).__init__()
    self.shape = args

  def forward(self, x):
    return x.view((x.size(0),)+self.shape)


def normalize(x, dim):
    '''
    Projects points to a sphere.
    '''
    return x.div(x.norm(2, dim=dim).view(x.shape[0],1,1,1))


class UpSample(nn.Module):

  def __init__(self, cfg, **kwargs):
    super(UpSample, self).__init__()

    self.scale_factor                   = get_attr_kwargs(cfg, 'scale_factor', default=2, **kwargs)
    self.mode                           = get_attr_kwargs(cfg, 'mode', default='bilinear',
                                                          choices=['bilinear', 'nearest'], **kwargs)
    self.align_corners                  = get_attr_kwargs(cfg, 'align_corners', default=None, **kwargs)


  def forward(self, x):
    x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
    return x



@MODEL_REGISTRY.register()
class Encoder(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(Encoder, self).__init__()

        nc                         = get_attr_kwargs(cfg, 'nc', default=3, **kwargs)
        ndf                        = get_attr_kwargs(cfg, 'ndf', **kwargs)
        padding_type               = get_attr_kwargs(cfg, 'padding_type', **kwargs)
        use_dropout                = get_attr_kwargs(cfg, 'use_dropout', **kwargs)
        use_bias                   = get_attr_kwargs(cfg, 'use_bias', **kwargs)
        nemb                       = get_attr_kwargs(cfg, 'nemb', **kwargs)
        self.embedding             = get_attr_kwargs(cfg, 'embedding', **kwargs)
        self.ngpu                  = get_attr_kwargs(cfg, 'ngpu', **kwargs)

        norm_layer = nn.BatchNorm2d
        self.main = nn.Sequential(
            # state size. (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf * 2, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf * 4, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf * 8, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(ndf*8, nemb, 4, 1, 0, bias=False),
            Reshape(ndf * 8 * 4 * 4),
            nn.Linear(ndf * 8 * 4 * 4, nemb),
            # nn.Linear(1024, nemb)
            )
        pass

    def forward(self, input):
        gpu_ids = None
        if isinstance(input, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
            output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        else:
            output = self.main(input)

        output = output.view(output.size(0), -1)
        if self.embedding == 'sphere':
            output = normalize(output)
        # output=utils.normalize_data(output)
        return output

@MODEL_REGISTRY.register()
class Encoder_conv(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(Encoder_conv, self).__init__()

        nc                         = get_attr_kwargs(cfg, 'nc', default=3, **kwargs)
        ndf                        = get_attr_kwargs(cfg, 'ndf', **kwargs)
        padding_type               = get_attr_kwargs(cfg, 'padding_type', **kwargs)
        use_dropout                = get_attr_kwargs(cfg, 'use_dropout', **kwargs)
        use_bias                   = get_attr_kwargs(cfg, 'use_bias', **kwargs)
        emb_channel                 = get_attr_kwargs(cfg, 'emb_channel', **kwargs)
        self.embedding             = get_attr_kwargs(cfg, 'embedding', **kwargs)
        self.ngpu                  = get_attr_kwargs(cfg, 'ngpu', **kwargs)

        norm_layer = nn.BatchNorm2d
        self.main = nn.Sequential(
            # state size. (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf * 2, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(2*ndf, emb_channel, 3, 1, 1, bias=False)
        )

        pass

    def forward(self, input):
        gpu_ids = None
        if isinstance(input, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
            output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        else:
            output = self.main(input)

        if self.embedding == 'sphere':
            output = normalize(output,dim=[1,2,3])
        # output=utils.normalize_data(output)
        return output

@MODEL_REGISTRY.register()
class Decoder_conv(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(Decoder_conv, self).__init__()

        emb_channel                     = get_attr_kwargs(cfg, 'emb_channel', **kwargs)
        ngf                      = get_attr_kwargs(cfg, 'ngf', **kwargs)
        padding_type             = get_attr_kwargs(cfg, 'padding_type', **kwargs)
        use_dropout              = get_attr_kwargs(cfg, 'use_dropout', **kwargs)
        use_bias                 = get_attr_kwargs(cfg, 'use_bias', **kwargs)
        nc                       = get_attr_kwargs(cfg, 'nc', **kwargs)
        self.ngpu                = get_attr_kwargs(cfg, 'ngpu', **kwargs)

        norm_layer = nn.BatchNorm2d

        self.main = nn.Sequential(
            nn.Conv2d(emb_channel, 2*ngf, 3, 1, 1, bias=False),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ngf x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf) x 128 x 128
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (nc) x 256 x 256
            nn.Conv2d(ngf, nc, 3, 1, 1),
            nn.Tanh()
        )
        pass

    def forward(self, input):

        # Check input is either (B,C,1,1) or (B,C)
        # assert input.nelement() == input.size(0) * input.size(1), 'wtf'
        # input = input.view(input.size(0), input.size(1), 1, 1)

        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 0:
            gpu_ids = range(self.ngpu)
            return nn.parallel.data_parallel(self.main, input, gpu_ids)
        else:
            return self.main(input)
@MODEL_REGISTRY.register()
class Decoder(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(Decoder, self).__init__()

        nemb                     = get_attr_kwargs(cfg, 'nemb', **kwargs)
        ngf                      = get_attr_kwargs(cfg, 'ngf', **kwargs)
        padding_type             = get_attr_kwargs(cfg, 'padding_type', **kwargs)
        use_dropout              = get_attr_kwargs(cfg, 'use_dropout', **kwargs)
        use_bias                 = get_attr_kwargs(cfg, 'use_bias', **kwargs)
        nc                       = get_attr_kwargs(cfg, 'nc', **kwargs)
        self.ngpu                = get_attr_kwargs(cfg, 'ngpu', **kwargs)

        norm_layer = nn.BatchNorm2d

        self.main = nn.Sequential(
            Reshape(nemb),
            nn.Linear(nemb, ngf * 8 * 4 * 4),
            Reshape(ngf * 8, 4, 4),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ngf x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf) x 128 x 128
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (nc) x 256 x 256
            nn.Conv2d(ngf, nc, 3, 1, 1),
            nn.Tanh()
        )
        pass

    def forward(self, input):

        # Check input is either (B,C,1,1) or (B,C)
        assert input.nelement() == input.size(0) * input.size(1), 'wtf'
        input = input.view(input.size(0), input.size(1), 1, 1)

        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 0:
            gpu_ids = range(self.ngpu)
            return nn.parallel.data_parallel(self.main, input, gpu_ids)
        else:
            return self.main(input)


@MODEL_REGISTRY.register()
class DecoderBilinear(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(DecoderBilinear, self).__init__()

        nemb                     = get_attr_kwargs(cfg, 'nemb', **kwargs)
        ngf                      = get_attr_kwargs(cfg, 'ngf', **kwargs)
        padding_type             = get_attr_kwargs(cfg, 'padding_type', **kwargs)
        use_dropout              = get_attr_kwargs(cfg, 'use_dropout', **kwargs)
        use_bias                 = get_attr_kwargs(cfg, 'use_bias', **kwargs)
        nc                       = get_attr_kwargs(cfg, 'nc', **kwargs)
        self.ngpu                = get_attr_kwargs(cfg, 'ngpu', **kwargs)

        norm_layer = nn.BatchNorm2d

        # self.main = nn.Sequential(
        #     Reshape(nemb),
        #     nn.Linear(nemb, ngf * 8 * 4 * 4),
        #     Reshape(ngf * 8, 4, 4),
        #     nn.BatchNorm2d(ngf * 8),
        #     nn.ReLU(True),
        #     # state size. (ngf*8) x 4 x 4
        #     UpSample(cfg={}),
        #     nn.Conv2d(ngf*8, ngf*4, 3, 1, 1),
        #     nn.BatchNorm2d(ngf * 4),
        #     nn.ReLU(True),
        #     # state size. (ngf*4) x 8 x 8
        #     UpSample(cfg={}),
        #     nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1),
        #     nn.BatchNorm2d(ngf * 2),
        #     nn.ReLU(True),
        #     # state size. (ngf*2) x 16 x 16
        #     UpSample(cfg={}),
        #     nn.Conv2d(ngf * 2, ngf, 3, 1, 1),
        #     nn.BatchNorm2d(ngf),
        #     nn.ReLU(True),
        #     # state size. ngf x 32 x 32
        #     UpSample(cfg={}),
        #     nn.Conv2d(ngf, ngf, 3, 1, 1),
        #     nn.BatchNorm2d(ngf),
        #     nn.ReLU(True),
        #     ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer,
        #                 use_dropout=use_dropout, use_bias=use_bias),
        #     # state size. (ngf) x 64 x 64
        #     UpSample(cfg={}),
        #     nn.Conv2d(ngf, ngf, 3, 1, 1),
        #     nn.BatchNorm2d(ngf),
        #     nn.ReLU(True),
        #     ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer,
        #                 use_dropout=use_dropout, use_bias=use_bias),
        #     # state size. (ngf) x 128 x 128
        #     UpSample(cfg={}),
        #     nn.Conv2d(ngf, nc, 3, 1, 1),
        #     # state size. (nc) x 256 x 256
        #     nn.Tanh()
        # )

        self.main = nn.Sequential(
            Reshape(nemb),
            nn.Linear(nemb, ngf * 8 * 4 * 4),
            Reshape(ngf * 8, 4, 4),
            ResnetBlock(ngf*8, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # 8x8
            UpSample(cfg={}),
            nn.Conv2d(ngf*8, ngf*4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ngf*4, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # 16*16
            UpSample(cfg={}),
            nn.Conv2d(ngf*4, ngf*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ngf*2, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # 32x32
            UpSample(cfg={}),
            nn.Conv2d(ngf*2, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # 64x64
            UpSample(cfg={}),
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # 128x128
            UpSample(cfg={}),
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # 256x256
            UpSample(cfg={}),
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            nn.Conv2d(ngf, nc, 3, 1, 1),
            nn.Tanh()
        )
        pass

    def forward(self, input):

        # Check input is either (B,C,1,1) or (B,C)
        assert input.nelement() == input.size(0) * input.size(1), 'wtf'
        input = input.view(input.size(0), input.size(1), 1, 1)

        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 0:
            gpu_ids = range(self.ngpu)
            return nn.parallel.data_parallel(self.main, input, gpu_ids)
        else:
            return self.main(input)

@MODEL_REGISTRY.register()
class netd(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(netd, self).__init__()

        ndf                      = get_attr_kwargs(cfg, 'ndf', **kwargs)
        padding_type             = get_attr_kwargs(cfg, 'padding_type', **kwargs)
        use_dropout              = get_attr_kwargs(cfg, 'use_dropout', **kwargs)
        use_bias                 = get_attr_kwargs(cfg, 'use_bias', **kwargs)
        nc                       = get_attr_kwargs(cfg, 'nc', **kwargs)
        self.ngpu                = get_attr_kwargs(cfg, 'ngpu', **kwargs)

        norm_layer = nn.BatchNorm2d
        self.main = nn.Sequential(
            # state size. (ndf) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(2*ndf, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf, 4, 1, 0, bias=False),
            # state size. (ndf) x 1 x 1
            nn.Conv2d(ndf, 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

        pass

    def forward(self, input):

        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 0:
            gpu_ids = range(self.ngpu)
            output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        else:
            output = self.main(input)

        return output.squeeze()

@MODEL_REGISTRY.register()
class netD(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(netD, self).__init__()

        ndf                      = get_attr_kwargs(cfg, 'ndf', **kwargs)
        padding_type             = get_attr_kwargs(cfg, 'padding_type', **kwargs)
        use_dropout              = get_attr_kwargs(cfg, 'use_dropout', **kwargs)
        use_bias                 = get_attr_kwargs(cfg, 'use_bias', **kwargs)
        emb_channel                       = get_attr_kwargs(cfg, 'emb_channel', **kwargs)
        self.ngpu                = get_attr_kwargs(cfg, 'ngpu', **kwargs)

        norm_layer = nn.BatchNorm2d
        self.main = nn.Sequential(
            nn.Conv2d(emb_channel, ndf, 3, 1, 1, bias=False),
            # state size. ndf x 16 x 1
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf * 2, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf*4, ndf, 3, 1, 1, bias=False),
            # state size. (ndf) x 4 x 4
            nn.Conv2d(ndf, 1, 4, 1, 0, bias=False),
            # state size. 1 x 1 x 1
            nn.Sigmoid()
        )

        pass

    def forward(self, input):

        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 0:
            gpu_ids = range(self.ngpu)
            output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        else:
            output = self.main(input)

        return output.squeeze()

@MODEL_REGISTRY.register()
class netg(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(netg, self).__init__()

        ngf                      = get_attr_kwargs(cfg, 'ngf', **kwargs)
        padding_type             = get_attr_kwargs(cfg, 'padding_type', **kwargs)
        use_dropout              = get_attr_kwargs(cfg, 'use_dropout', **kwargs)
        use_bias                 = get_attr_kwargs(cfg, 'use_bias', **kwargs)
        emb_channel              = get_attr_kwargs(cfg, 'emb_channel', **kwargs)
        self.ngpu                = get_attr_kwargs(cfg, 'ngpu', **kwargs)
        self.embedding           = get_attr_kwargs(cfg, 'embedding', **kwargs)

        norm_layer = nn.BatchNorm2d
        self.main = nn.Sequential(
            nn.Conv2d(emb_channel, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            nn.Conv2d(ngf, ngf*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            ResnetBlock(ngf * 2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            nn.Conv2d(ngf*2, ngf*4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            ResnetBlock(ngf * 2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            nn.Conv2d(ngf * 2, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            nn.Conv2d(ngf, emb_channel, 3, 1, 1, bias=False),
            nn.Tanh()
        )

        pass

    def forward(self, input):

        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 0:
            gpu_ids = range(self.ngpu)
            output = nn.parallel.data_parallel(self.main, input, gpu_ids)
            if self.embedding == 'sphere':
                output = normalize(output,dim=[1,2,3])

        else:
            output = self.main(input)
            if self.embedding == 'sphere':
                output = normalize(output,dim=[1,2,3])
        # output = utils.normalize_data(output)
        return output

