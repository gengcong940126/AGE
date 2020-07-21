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


def normalize(x, dim=1):
    '''
    Projects points to a sphere.
    '''
    return x.div(x.norm(2, dim=dim).unsqueeze(dim).expand_as(x))


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
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            # state size. (nc) x 256 x 256
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