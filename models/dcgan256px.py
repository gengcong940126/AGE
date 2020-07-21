import torch.nn as nn
from .base import _netE_Base, _netG_Base, _netg_Base, _nete_Base,_netD_Base, _netd_Base

def _netd(opt):
    ndf = opt.ndf
    nc = opt.nc
    nz = opt.nz


    main = nn.Sequential(
            # state size. (ndf) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    return _netd_Base(opt,main)

def _netG(opt):
    ngf = opt.ngf
    nc = opt.nc
    nz = opt.nz
    nemb=opt.nemb
    padding_type = 'reflect'
    norm_layer = nn.BatchNorm2d
    use_dropout = False
    use_bias = False
    main = nn.Sequential(
            Reshape(nemb),
            nn.Linear(nemb,ngf*8*4*4),
            Reshape(ngf*8,4,4),
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
            ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf) x 128 x 128
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            # state size. (nc) x 256 x 256
            nn.Tanh()
        )

    return _netG_Base(opt, main)

def _netE(opt):
    ndf = opt.ndf
    nc = opt.nc
    nz = opt.nz
    nemb = opt.nemb
    padding_type = 'reflect'
    norm_layer = nn.BatchNorm2d
    use_dropout = False
    use_bias = False
    main = nn.Sequential(
            # state size. (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf * 8, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
             # state size. (ndf*8) x 4 x 4
            #nn.Conv2d(ndf*8, nemb, 4, 1, 0, bias=False),
            Reshape(ndf*8*4*4),
            nn.Linear(ndf*8*4*4,nemb),
            #nn.Linear(1024, nemb)

        )

    return _netE_Base(opt, main)
def _netE2(opt):
    ndf = opt.ndf
    nc = opt.nc
    nz = opt.nz
    nemb = opt.nemb
    padding_type = 'reflect'
    norm_layer = nn.BatchNorm2d
    use_dropout = False
    use_bias = False
    main = nn.Sequential(
            # state size. (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf*2) x 16 x 16

            #nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
           # nn.BatchNorm2d(ndf * 4),
            #nn.LeakyReLU(0.2, inplace=True),
            #ResnetBlock(ndf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),

            # state size. (ndf*4) x 8 x 8

            #nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 8),
           # nn.LeakyReLU(0.2, inplace=True),
            #ResnetBlock(ndf * 8, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),

             # state size. (ndf*8) x 4 x 4
            #nn.Conv2d(ndf*8, nemb, 4, 1, 0, bias=False),

            #Reshape(ndf*8*4*4),
            #nn.Linear(ndf*8*4*4,1024),
            #nn.Linear(1024, 256)

        )

    return _netE_Base(opt, main)
def _netG2(opt):
    ngf = opt.ngf
    nc = opt.nc
    nz = opt.nz
    nemb=opt.nemb
    padding_type = 'reflect'
    norm_layer = nn.BatchNorm2d
    use_dropout = False
    use_bias = False
    main = nn.Sequential(
            #Reshape(nemb),
            #nn.Linear(nemb,ngf*8*4*4),
            #Reshape(ngf*8,4,4),


           # nn.Upsample(scale_factor=2, mode='nearest'),
            #nn.Conv2d(ngf*8, ngf*4, 3, 1, 1, bias=False),
           # ResnetBlock(ngf*4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf*4) x 8 x 8
           # nn.Upsample(scale_factor=2, mode='nearest'),
           # nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1, bias=False),
            #ResnetBlock(ngf*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf*2) x 16 x 16

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf * 2, ngf , 3, 1, 1, bias=False),
            ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. ngf x 32 x 32
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf) x 64 x 64
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf) x 128 x 128
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            nn.Conv2d(ngf , nc, 3, 1, 1, bias=False),

            # state size. (nc) x 256 x 256
            nn.Tanh()
        )

    return _netG_Base(opt, main)
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