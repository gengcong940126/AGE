from __future__ import print_function
import argparse
import torch
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
parser.add_argument('--dataset', required=False, default='mnist',
                    help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', type=str, help='path to dataset',default='./data')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=0)
parser.add_argument('--batch_size', type=int,
                    default=64, help='batch size')
parser.add_argument('--image_size', type=int, default=32,
                    help='the resolution of the input image to network')
parser.add_argument('--nz', type=int, default=256,
                    help='size of the latent z vector')
parser.add_argument('--nemb', type=int, default=256,
                    help='size of the latent embedding')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int,default=1)
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

parser.add_argument('--netG', default='dcgan32px',
                    help="path to netG config")
parser.add_argument('--netE', default='dcgan32px',
                    help="path to netE config")
parser.add_argument('--netg', default='dcgan32px',
                    help="path to netg config")
parser.add_argument('--nete', default='dcgan32px',
                    help="path to nete config")
parser.add_argument('--netD', default='dcgan32px',
                    help="path to netD config")
parser.add_argument('--netd', default='dcgan32px',
                    help="path to netd config")
parser.add_argument('--netG_chp', default='./results_AEGAN_acai_sphere_256/netG_epoch_306.pth',
                    help="path to netG (to continue training)")
parser.add_argument('--netD_chp', default='./results_AEGAN_acai_sphere_256/netG_epoch_306.pth',
                    help="path to netD (to continue training)")
parser.add_argument('--netE_chp', default='./results_AEGAN_acai_sphere_256/netE_epoch_306.pth',
                    help="path to netE (to continue training)")
#./results_loage/netg_epoch_35.pth
parser.add_argument('--netg_chp', default='./results_AEGAN_acai_sphere_256/netg_epoch_306.pth',
                    help="path to netG (to continue training)")
parser.add_argument('--nete_chp', default='./results_AEGAN_acai_sphere_256/netG_epoch_306.pth',
                    help="path to netE (to continue training)")
parser.add_argument('--netd_chp', default='./results_AEGAN_acai_sphere_256/netG_epoch_306.pth',
                    help="path to netd (to continue training)")
parser.add_argument('--save_dir', default='./results/results_ablation/acai256',
                    help='folder to output images and model checkpoints')
parser.add_argument('--criterion', default='param',
                    help='param|nonparam, How to estimate KL')
parser.add_argument('--KL', default='qp', help='pq|qp')
parser.add_argument('--noise', default='normal', help='normal|sphere')
parser.add_argument('--embedding', default='normal', help='normal|sphere')
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
#os.makedirs('./results_celeba/aae2_128',exist_ok=True)
os.makedirs('./results/results_ablation/normal256/tb',exist_ok=True)
writer=SummaryWriter(log_dir='./results/results_ablation/normal256/tb')

# Setup cudnn, seed, and parses updates string.
updates = setup(opt)

# Setup dataset
dataloader = dict(train=setup_dataset(opt, train=True,shuffle=False),
                  val=setup_dataset(opt, train=False,shuffle=False))

# Load generator
netG = load_G(opt)

# Load encoder
netE = load_E(opt)

netg = load_g(opt).to('cuda')
#netD = load_D(opt).to('cuda')


x = torch.FloatTensor(opt.batch_size, opt.nc,
                      opt.image_size, opt.image_size)
x2 = torch.FloatTensor(opt.batch_size, opt.nc,
                      opt.image_size, opt.image_size)
z = torch.FloatTensor(opt.batch_size, opt.nz, 1, 1)
fixed_z = torch.FloatTensor(opt.batch_size, opt.nz, 1, 1).normal_(0, 1)

if opt.noise == 'sphere':
    normalize_(fixed_z)

if opt.cuda:
    netE.cuda()
    netG.cuda()
    x = x.cuda()
    x2=x2.cuda()
    z, fixed_z = z.cuda(), fixed_z.cuda()

x = Variable(x)
z = Variable(z)
x2 = Variable(x2)
fixed_z = Variable(fixed_z)

# Setup optimizers
optimizerE = optim.Adam(netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


real_cpu = torch.FloatTensor()


def save_images():
    populate_x(x, dataloader['val'])
    real_cpu.resize_(x.data.size()).copy_(x.data)

    # Real samples
    save_path = '%s/real_samples.png' % opt.save_dir
    vutils.save_image(real_cpu[:64]/2+0.5 , save_path)
    gex = netG(netE(x))
    save_path = '%s/reconstructions.png' % (opt.save_dir)
    grid = vutils.save_image(gex.data[:64] / 2 + 0.5, save_path)
    #netG.eval()
    populate_z(z, opt)
    fake = netG(netg(z.squeeze()))

    # Fake samples
    save_path = '%s/fake_samples.png' % (opt.save_dir)
    vutils.save_image(fake.data[:64]/2+0.5 , save_path)

    # Save reconstructions
    #netE.eval()
    populate_x(x, dataloader['val'])
    populate_x(x2, dataloader['val'])
    alpha = torch.rand(x.shape[0], 1).cuda()
    alpha = 0.5 - torch.abs(alpha - 0.5)  # Make interval [0, 0.5]
    encode_mix = alpha * netE(x) + (1 - alpha) * netE(x2)
    if opt.embedding == 'sphere':
        encode_mix=normalize(encode_mix)
    x_alpha = netG(encode_mix)
    save_path = '%s/interpolate_samples.png' % (opt.save_dir)
    vutils.save_image(x_alpha.data[:64] / 2 + 0.5, save_path)



    #t = torch.FloatTensor(x.size(0) * 2, x.size(1),
                          #x.size(2), x.size(3))

    #t[0::2] = x.data[:]
    #t[1::2] = gex.data[:]


    #netG.train()
    #netE.train()

save_images()