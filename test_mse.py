from __future__ import print_function
import argparse
import torch
import scipy.io as scio
import torch.nn.parallel
from easydict import EasyDict
import yaml
from tensorboardX import SummaryWriter
import torch.optim as optim
import src.utils
import matplotlib.pyplot as plt
import shutil
import torchvision.utils as vutils
from torch.autograd import Variable
import pytorch_fid.fid_score as fid
from src.utils import *
import src.losses as losses
from evaluation import build_GAN_metric
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='celeba',
                    help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', type=str, help='path to dataset',default='./datasets/celeba_all/img_align_celeba')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=0)
parser.add_argument('--batch_size', type=int,
                    default=64, help='batch size')
parser.add_argument('--image_size', type=int, default=64,
                    help='the resolution of the input image to network')
parser.add_argument('--nz', type=int, default=128,
                    help='size of the latent z vector')
parser.add_argument('--nemb', type=int, default=128,
                    help='size of the latent embedding')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int,default=3)
parser.add_argument('--reg', type=int,default=0.2)

parser.add_argument('--nepoch', type=int, default=25,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cpu', action='store_true',
                    help='use CPU instead of GPU')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')

parser.add_argument('--netG', default='dcgan64px',
                    help="path to netG config")
parser.add_argument('--netE', default='dcgan64px',
                    help="path to netE config")
parser.add_argument('--netg', default='dcgan32px',
                    help="path to netg config")
parser.add_argument('--nete', default='dcgan32px',
                    help="path to nete config")
parser.add_argument('--netD', default='dcgan32px',
                    help="path to netD config")
parser.add_argument('--netd', default='dcgan64px',
                    help="path to netd config")
parser.add_argument('--netG_chp', default='./results_celeba/introvae_128/netG_epoch_24.pth',
                    help="path to netG (to continue training)")
parser.add_argument('--netD_chp', default='./results_celeba/vae_128/netD_epoch_24.pth',
                    help="path to netD (to continue training)")
parser.add_argument('--vae_netE_chp', default='./results_celeba/introvae_128/netE_epoch_24.pth',
                    help="path to netE (to continue training)")
#./results_loage/netg_epoch_35.pth
parser.add_argument('--netg_chp', default='./results_celeba/vae_128/netg_epoch_24.pth',
                    help="path to netG (to continue training)")
parser.add_argument('--nete_chp', default='./results_celeba/vae_128/nete_epoch_24.pth',
                    help="path to netE (to continue training)")
parser.add_argument('--netd_chp', default='./results_celeba/vae_128/netd_epoch_24.pth',
                    help="path to netd (to continue training)")
parser.add_argument('--save_dir', default='./results_celeba/recon',
                    help='folder to output images and model checkpoints')
parser.add_argument('--criterion', default='param',
                    help='param|nonparam, How to estimate KL')
parser.add_argument('--KL', default='qp', help='pq|qp')
parser.add_argument('--noise', default='sphere', help='normal|sphere')
parser.add_argument('--embedding', default='sphere', help='normal|sphere')
parser.add_argument('--match_z', default='L2', help='none|L1|L2|cos')
parser.add_argument('--match_x', default='L2', help='none|L1|L2|cos')

parser.add_argument('--drop_lr', default=40, type=int, help='')
parser.add_argument('--save_every', default=50, type=int, help='')

parser.add_argument('--manual_seed', type=int, default=123, help='manual seed')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number to start with')

parser.add_argument(
    '--D_updates', default="5;KL_fake:1,KL_real:1,match_z:0,match_x:10",
    help='Update plan for encoder <number of updates>;[<term:weight>]'
)

parser.add_argument(
    '--G_updates', default="1;KL_fake:1,match_z:10,match_x:0",
    help='Update plan for generator <number of updates>;[<term:weight>]'
)
parser.add_argument(
    '--e_updates', default="1;KL_fake:1,KL_real:1,match_z:0,match_x:10",
    help='Update plan for encoder <number of updates>;[<term:weight>]'
)
parser.add_argument(
    '--d_updates', default="5;KL_fake:1,KL_real:1,match_z:0,match_x:10",
    help='Update plan for encoder <number of updates>;[<term:weight>]'
)

parser.add_argument(
    '--g_updates', default="1;KL_fake:1,match_z:10,match_x:0",
    help='Update plan for generator <number of updates>;[<term:weight>]'
)
opt = parser.parse_args()

# Setup cudnn, seed, and parses updates string.
updates = setup(opt)

# Setup dataset
dataloader = dict(train=setup_dataset(opt, train=True,shuffle=False),
                   val=setup_dataset(opt, train=False,shuffle=False))

# Load generator
netG = load_G(opt).to('cuda')

# Load encoder
netE = load_vae_E(opt).to('cuda')
#netE = load_E(opt).to('cuda')

# Load generator_latent
#netg = load_g(opt).to('cuda')

# Load encoder_latent
#nete = load_e(opt).to('cuda')

#netD = load_D(opt).to('cuda')

netE.eval()
netG.eval()
#os.makedirs('./results_celeba/generate_images',exist_ok=True)
z2= torch.FloatTensor(opt.batch_size, opt.nz, 1, 1).cuda()
# =================== #
# Calcualte FID score #
# =================== #
# STAT_FILE = "data/fid_stats_celeba.npz"
x = torch.FloatTensor(opt.batch_size, opt.nc,
                       opt.image_size, opt.image_size).cuda()

# f = np.load(STAT_FILE)
# mu_real, sigma_real = f['mu'][:], f['sigma'][:]
# f.close()

# for j in range(FID_EVAL_SIZE // FID_SAMPLE_BATCH_SIZE):
#     z2= torch.FloatTensor(FID_SAMPLE_BATCH_SIZE, opt.nz, 1, 1).normal_(0, 1).cuda()
#     if opt.noise == 'sphere':
#         normalize_(z2)
#     fake = netG(z2.squeeze())
#     #fake = netG(netg(z2.squeeze()))
#     for i in range(fake.shape[0]):
#         save_path = '%s/generate_images/images_%d.png' % (opt.save_dir,i+j*FID_SAMPLE_BATCH_SIZE)
#         vutils.save_image(fake[i] / 2 + 0.5, save_path)

# for j in range(FID_EVAL_SIZE // opt.batch_size):
#     populate_x(x, dataloader['train'])
#
#     #fake = netG(netg(z2.squeeze()))
#     for i in range(x.shape[0]):
#         save_path = '%s/generate_images/images_%d.png' % (opt.save_dir,i+j*opt.batch_size)
#         vutils.save_image(x[i] / 2 + 0.5, save_path)

# fid_value = fid.calculate_fid_given_paths([STAT_FILE,os.path.join(opt.save_dir,'generate_images')],
#                                           batch_size=64,dims=2048,cuda=0)

#fid_value = fid.calculate_fid_given_paths([STAT_FILE,STAT_FILE],
                                       #   batch_size=50,dims=2048,cuda=0)
#print('FID: ', fid_value)
#for j in range(100):
populate_x(x, dataloader['val'])
            # z = self.z2
with torch.no_grad():
    z,_,_=netE(x)
    AE=netG(z)
criterion=torch.nn.MSELoss()
mse=criterion(AE,x)
print(mse)
t = torch.FloatTensor(x.size(0)* 2, x.size(1),
                          x.size(2), x.size(3))

t[0::2] = x.data[:]
t[1::2] = AE.data[:]

save_path = '%s/reconstructions_introvae.png' % (opt.save_dir)
grid = vutils.save_image(t[:12]/2+0.5 , save_path,nrow=2)
file = './index.mat'
scio.savemat(file, {'data': x.detach().cpu().numpy()})