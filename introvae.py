from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
from tensorboardX import SummaryWriter
import torch.optim as optim
import src.utils
import matplotlib.pyplot as plt
import shutil
import torchvision.utils as vutils
from torch.autograd import Variable
from src.utils import *
import src.losses as losses

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='cifar10',
                    help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', type=str, help='path to dataset',default='./data/raw/cifar10')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=0)
parser.add_argument('--batch_size', type=int,
                    default=64, help='batch size')
parser.add_argument('--image_size', type=int, default=32,
                    help='the resolution of the input image to network')
parser.add_argument('--nz', type=int, default=128,
                    help='size of the latent z vector')
parser.add_argument('--nemb', type=int, default=128,
                    help='size of the latent embedding')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int,default=3)
parser.add_argument('--reg', type=int,default=0.2)

parser.add_argument('--nepoch', type=int, default=100,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cpu', action='store_true',
                    help='use CPU instead of GPU')
parser.add_argument('--ngpu', type=int, default=0,
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
parser.add_argument('--netG_chp', default='',
                    help="path to netG (to continue training)")
parser.add_argument('--netD_chp', default='',
                    help="path to netD (to continue training)")
parser.add_argument('--vae_netE_chp', default='',
                    help="path to netE (to continue training)")
#./results_loage/netg_epoch_35.pth
parser.add_argument('--netg_chp', default='',
                    help="path to netG (to continue training)")
parser.add_argument('--nete_chp', default='',
                    help="path to netE (to continue training)")
parser.add_argument('--netd_chp', default='',
                    help="path to netd (to continue training)")
parser.add_argument('--save_dir', default='./results_cifar10/introvae_128',
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
os.makedirs('./results_cifar10/introvae_128',exist_ok=True)
os.makedirs('./results_cifar10/introvae_128/tb',exist_ok=True)
writer=SummaryWriter(log_dir='./results_cifar10/introvae_128/tb')

#export CUDA_VISIBLE_DEVICES = 0

# Setup cudnn, seed, and parses updates string.
updates = setup(opt)

# Setup dataset
dataloader = dict(train=setup_dataset(opt, train=True),
                  val=setup_dataset(opt, train=False))

# Load generator
netG = load_G(opt)

# Load encoder
netE = load_vae_E(opt)


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


real_cpu = torch.FloatTensor()


def save_images(epoch):

    real_cpu.resize_(x.data.size()).copy_(x.data)

    # Real samples
    save_path = '%s/real_samples.png' % opt.save_dir
    vutils.save_image(real_cpu[:64]/2+0.5 , save_path)

    netG.eval()
    populate_z(z, opt)
    fake = netG(z.squeeze())

    # Fake samples
    save_path = '%s/fake_samples_epoch_%03d.png' % (opt.save_dir, epoch)
    vutils.save_image(fake.data[:64]/2+0.5 , save_path)

    # Save reconstructions
    netE.eval()
    populate_x(x, dataloader['val'])
    populate_x(x2, dataloader['val'])
    alpha = torch.rand(x.shape[0], 1).cuda()
    alpha = 0.5 - torch.abs(alpha - 0.5)  # Make interval [0, 0.5]
    sample_z1,_,_=netE(x)
    sample_z2, _, _ = netE(x2)
    encode_mix = alpha * sample_z1 + (1 - alpha) * sample_z2
    if opt.embedding == 'sphere':
        normalize(encode_mix)
    x_alpha = netG(encode_mix)
    save_path = '%s/interpolate_samples_epoch_%03d.png' % (opt.save_dir, epoch)
    vutils.save_image(x_alpha.data[:64] / 2 + 0.5, save_path)

    gex = netG(sample_z1)

    t = torch.FloatTensor(x.size(0) * 2, x.size(1),
                          x.size(2), x.size(3))

    t[0::2] = x.data[:]
    t[1::2] = gex.data[:]

    save_path = '%s/reconstructions_epoch_%03d.png' % (opt.save_dir, epoch)
    grid = vutils.save_image(t[:64]/2+0.5 , save_path)
    netG.train()
    netE.train()
def adjust_lr(epoch):
    if epoch % opt.drop_lr == (opt.drop_lr - 1):
        opt.lr /= 2
        for param_group in optimizerE.param_groups:
            param_group['lr'] = opt.lr

        for param_group in optimizerG.param_groups:
            param_group['lr'] = opt.lr

stats = {}
batches_done=0
for epoch in range(opt.start_epoch, opt.nepoch):

    # Adjust learning rate
    adjust_lr(epoch)

    for i in range(len(dataloader['train'])):
        batches_done=batches_done+1
        if epoch < 0:
            # ---------------------------
            #        VAE
            # ---------------------------
            netE.zero_grad()
            netG.zero_grad()
            # X
            populate_x(x, dataloader['train'])
            # E(X)
            Ex,mu,logsigma = netE(x)
            AE = netG(Ex)
            #err = F.binary_cross_entropy_with_logits(AE, x,size_average=False).div(opt.batch_size)
            err=match(AE, x, opt.match_x)
            kl_loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())/opt.batch_size/opt.nz
            loss = kl_loss+ err
            stats['AE_loss'] = err
            stats['KL_loss'] = kl_loss
            loss.backward()
            optimizerE.step()
            optimizerG.step()
            print('[{epoch}/{nepoch}][{iter}/{niter}] '
                  'KL_loss/AE_loss: {KL_loss:.3f}/{AE_loss:.3f}'
                  ''.format(epoch=epoch,
                            nepoch=opt.nepoch,
                            iter=i,
                            niter=len(dataloader['train']),
                            **stats))
            if i % opt.save_every == 0:
                writer.add_scalar('AE_loss', stats['AE_loss'], batches_done)
                writer.add_scalar('KL_loss', stats['KL_loss'], batches_done)
        else:
            # ---------------------------
            #        introVAE
            # ---------------------------
            netE.zero_grad()
            populate_x(x, dataloader['train'])
            populate_z(z, opt)
            Gz = netG(z.squeeze())
            Ex, real_mu, real_logsigma = netE(x)
            AE = netG(Ex)
            _,rec_mu, rec_logsigma = netE(AE.detach())
            _,fake_mu, fake_logsigma = netE(Gz.detach())
            loss_rec = match(AE, x, opt.match_x)
            lossE_real_kl = -0.5 * torch.sum(1 + real_logsigma - real_mu.pow(2) - real_logsigma.exp())/opt.batch_size
            lossE_rec_kl = -0.5 * torch.sum(1 + rec_logsigma - rec_mu.pow(2) - rec_logsigma.exp())/opt.batch_size
            lossE_fake_kl = -0.5 * torch.sum(1 + fake_logsigma - fake_mu.pow(2) - fake_logsigma.exp())/opt.batch_size
            loss_margin = lossE_real_kl + \
                          (F.relu(100 - lossE_rec_kl) + F.relu(100 - lossE_fake_kl)) * 0.25 * 1
            lossE = loss_rec*5  + loss_margin*0.001
            stats['lossE'] = lossE
            lossE.backward(retain_graph=True)
            # nn.utils.clip_grad_norm(model.encoder.parameters(), 1.0)
            optimizerE.step()

            # ---------------------------
            #        Optimize over G
            # ---------------------------
            netG.zero_grad()
            _,rec_mu, rec_logsigma = netE(AE)
            _,fake_mu, fake_logsigma = netE(Gz)

            lossG_rec_kl = -0.5 * torch.sum(1 + rec_logsigma - rec_mu.pow(2) - rec_logsigma.exp())/opt.batch_size
            lossG_fake_kl = -0.5 * torch.sum(1 + fake_logsigma - fake_mu.pow(2) - fake_logsigma.exp())/opt.batch_size

            lossG = (lossG_rec_kl + lossG_fake_kl) * 0.25 * 0.001+ match(AE, x, opt.match_x)*5
            stats['lossG'] = lossG
            # optimizerG.zero_grad()
            lossG.backward()
            # nn.utils.clip_grad_norm(model.decoder.parameters(), 1.0)
            optimizerG.step()

            print('[{epoch}/{nepoch}][{iter}/{niter}] '
                      'lossE/lossG: {lossE:.3f}/{lossG:.3f}'
                      ''.format(epoch=epoch,
                                nepoch=opt.nepoch,
                                iter=i,
                                niter=len(dataloader['train']),
                                **stats))

            if i % opt.save_every == 0:
                writer.add_scalar('lossE', stats['lossE'], batches_done)
                writer.add_scalar('lossG', stats['lossG'], batches_done)

        if i % opt.save_every == 0:
            save_images(epoch)

        # If an epoch takes long time, dump intermediate
        # if opt.dataset in ['lsun', 'imagenet'] and (i % 5000 == 0):
        #     torch.save(netG, '%s/netG_epoch_%d_it_%d.pth' %
        #                (opt.save_dir, epoch, i))
        #     torch.save(netE, '%s/netE_epoch_%d_it_%d.pth' %
        #                (opt.save_dir, epoch, i))


    # do checkpointing
    torch.save(netG, '%s/netG_epoch_%d.pth' % (opt.save_dir, epoch))
    torch.save(netE, '%s/netE_epoch_%d.pth' % (opt.save_dir, epoch))

