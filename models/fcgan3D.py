import torch.nn as nn
from .base import _netE_Base, _netG_Base, _netg_Base, _nete_Base,_netG_Base_3D


# ------------------------
#         E
# ------------------------


def _netE(opt):
    ndf = opt.ndf
    nc = opt.nc
    nz = opt.nz

    main =  nn.Sequential(nn.Linear(nc, 400),
                      nn.BatchNorm1d(400),
                      nn.ReLU(),
                      nn.Linear(400, nz)
                      )

    return _netE_Base(opt, main)


# ------------------------
#         G
# ------------------------


def _netG(opt):
    ngf = opt.ngf
    nc = opt.nc
    nz = opt.nz

    main = nn.Sequential(
        # input is Z, going into a convolution
        nn.Linear(nz, 400),
        nn.BatchNorm1d(400),
        nn.ReLU(),
        nn.Linear(400, nc),
        nn.Tanh()
        # nn.Sigmoid()
    )

    return _netG_Base_3D(opt, main)


# ------------------------
#         g
# ------------------------

def _netg(opt):
    ngf = opt.ngf
    nc = opt.nc
    nz = opt.nz

    main = nn.Sequential(
        # input is Z, going into a convolution
        nn.Linear(nz, 400),
        nn.BatchNorm1d(400),
        nn.ReLU(),
        nn.Linear(400, nz),
        # nn.Sigmoid()
        # nn.Sigmoid()
    )
    return _netg_Base(opt, main)


# ------------------------
#         g
# ------------------------

def _nete(opt):
    ndf = opt.ndf
    nc = opt.nc
    nz = opt.nz

    main = nn.Sequential(
        # input is (nc) x 32 x 32
        nn.Sequential(nn.Linear(nz, 400),
                      nn.BatchNorm1d(400),
                      nn.ReLU(),
                      nn.Linear(400, nz)
                      )

    )

    return _nete_Base(opt, main)