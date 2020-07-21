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
parser.add_argument('--dataset', required=False, default='imagenet',
                    help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', type=str, help='path to dataset',default=os.path.expanduser('~/user/code/AEGAN/data/raw/imagenet'))
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=0)
parser.add_argument('--batch_size', type=int,
                    default=32, help='batch size')
parser.add_argument('--image_size', type=int, default=256,
                    help='the resolution of the input image to network')
parser.add_argument('--nz', type=int, default=512,
                    help='size of the latent z vector')
parser.add_argument('--nemb', type=int, default=512,
                    help='size of the latent embedding')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int,default=3)
parser.add_argument('--reg', type=int,default=0.2)

parser.add_argument('--nepoch', type=int, default=500,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cpu', action='store_true',
                    help='use CPU instead of GPU')
parser.add_argument('--ngpu', type=int, default=4,
                    help='number of GPUs to use')

parser.add_argument('--netG', default='dcgan256px',
                    help="path to netG config")
parser.add_argument('--netE', default='dcgan256px',
                    help="path to netE config")
parser.add_argument('--netg', default='dcgan32px',
                    help="path to netg config")
parser.add_argument('--nete', default='dcgan32px',
                    help="path to nete config")
parser.add_argument('--netD', default='dcgan32px',
                    help="path to netD config")
parser.add_argument('--netd', default='dcgan256px',
                    help="path to netd config")
#'./results_volcano/AEGAN_acai_256/netG_epoch_49.pth'
parser.add_argument('--netG_chp', default='',
                    help="path to netG (to continue training)")
parser.add_argument('--netD_chp', default='',
                    help="path to netD (to continue training)")
parser.add_argument('--netE_chp', default='',
                    help="path to netE (to continue training)")
#./results_loage/netg_epoch_35.pth
parser.add_argument('--netg_chp', default='',
                    help="path to netG (to continue training)")
parser.add_argument('--nete_chp', default='',
                    help="path to netE (to continue training)")
parser.add_argument('--netd_chp', default='',
                    help="path to netd (to continue training)")
parser.add_argument('--save_dir', default='./results_volcano/AEGAN_acai_1024',
                    help='folder to output images and model checkpoints')
parser.add_argument('--criterion', default='param',
                    help='param|nonparam, How to estimate KL')
parser.add_argument('--KL', default='qp', help='pq|qp')
parser.add_argument('--noise', default='sphere', help='normal|sphere')
parser.add_argument('--embedding', default='sphere', help='normal|sphere')
parser.add_argument('--match_z', default='L2', help='none|L1|L2|cos')
parser.add_argument('--match_x', default='L1', help='none|L1|L2|cos')

parser.add_argument('--drop_lr', default=40, type=int, help='')
parser.add_argument('--save_every', default=250, type=int, help='')

parser.add_argument('--manual_seed', type=int, default=123, help='manual seed')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number to start with')

parser.add_argument(
    '--D_updates', default="1;KL_fake:1,KL_real:1,match_z:0,match_x:10",
    help='Update plan for encoder <number of updates>;[<term:weight>]'
)

parser.add_argument(
    '--G_updates', default="1;KL_fake:1,match_z:10,match_x:0",
    help='Update plan for generator <number of updates>;[<term:weight>]'
)
parser.add_argument(
    '--e_updates', default="1;KL_fake:1,KL_real:1,match_z:0,match_x:0",
    help='Update plan for encoder <number of updates>;[<term:weight>]'
)
parser.add_argument(
    '--d_updates', default="1;KL_fake:1,KL_real:1,match_z:0,match_x:10",
    help='Update plan for encoder <number of updates>;[<term:weight>]'
)

parser.add_argument(
    '--g_updates', default="1;KL_fake:1,match_z:10,match_x:0",
    help='Update plan for generator <number of updates>;[<term:weight>]'
)
opt = parser.parse_args()
os.makedirs('./results_volcano/AEGAN_acai2_256',exist_ok=True)
os.makedirs('./results_volcano/AEGAN_acai2_256/tb',exist_ok=True)
writer=SummaryWriter(log_dir='./results_volcano/AEGAN_acai2_256/tb')

#export CUDA_VISIBLE_DEVICES = 0,1,2
# Setup cudnn, seed, and parses updates string.
updates = setup(opt)

# Setup dataset
dataloader = dict(train=setup_dataset(opt, train=True),
                  val=setup_dataset(opt, train=False))

# Load generator
netG = load_G(opt)

# Load encoder
netE = load_E(opt)

# Load generator_latent
netg = load_g(opt).to('cuda')

# Load encoder_latent
nete = load_e(opt).to('cuda')

netD = load_D(opt).to('cuda')

netd = load_d(opt).to('cuda')

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
optimizerg = optim.Adam(netg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizere = optim.Adam(nete.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerd = optim.Adam(netd.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# Setup criterions
if opt.criterion == 'param':
    print('Using parametric criterion KL_%s' % opt.KL)
    KL_minimizer = losses.KLN01Loss(direction=opt.KL, minimize=True)
    KL_maximizer = losses.KLN01Loss(direction=opt.KL, minimize=False)
elif opt.criterion == 'nonparam':
    print('Using NON-parametric criterion KL_%s' % opt.KL)
    KL_minimizer = losses.SampleKLN01Loss(direction=opt.KL, minimize=True)
    KL_maximizer = losses.SampleKLN01Loss(direction=opt.KL, minimize=False)
else:
    assert False, 'criterion?'

real_cpu = torch.FloatTensor()


def save_images(epoch):

    real_cpu.resize_(x.data.size()).copy_(x.data)

    # Real samples
    save_path = '%s/real_samples.png' % opt.save_dir
    vutils.save_image(real_cpu[:64]/2+0.5 , save_path)

    #netG.eval()
    #netg.eval()

    populate_z(z, opt)
    fake = netG(netg(z.squeeze()))

    # Fake samples
    save_path = '%s/fake_samples_epoch_%03d.png' % (opt.save_dir, epoch)
    vutils.save_image(fake.data[:64]/2+0.5 , save_path)

    # Save reconstructions
    #netE.eval()
    populate_x(x, dataloader['val'])
    populate_x(x2, dataloader['val'])
    alpha = torch.rand(x.shape[0], 1).cuda()
    alpha = 0.5 - torch.abs(alpha - 0.5)  # Make interval [0, 0.5]
    encode_mix = alpha * netE(x) + (1 - alpha) * netE(x2)
    if opt.embedding == 'sphere':
        normalize(encode_mix)
    x_alpha = netG(encode_mix)
    save_path = '%s/interpolate_samples_epoch_%03d.png' % (opt.save_dir, epoch)
    vutils.save_image(x_alpha.data[:64] / 2 + 0.5, save_path)

    gex = netG(netE(x))

    t = torch.FloatTensor(x.size(0) * 2, x.size(1),
                          x.size(2), x.size(3))

    t[0::2] = x.data[:]
    t[1::2] = gex.data[:]

    save_path = '%s/reconstructions_epoch_%03d.png' % (opt.save_dir, epoch)
    grid = vutils.save_image(t[:64]/2+0.5 , save_path)
    #netG.train()
    #netg.train()
    #netE.train()
def adjust_lr(epoch):
    if epoch % opt.drop_lr == (opt.drop_lr - 1):
        opt.lr /= 2
        for param_group in optimizerD.param_groups:
            param_group['lr'] = opt.lr

        for param_group in optimizerG.param_groups:
            param_group['lr'] = opt.lr

        for param_group in optimizere.param_groups:
            param_group['lr'] = opt.lr

        for param_group in optimizerg.param_groups:
            param_group['lr'] = opt.lr

        for param_group in optimizerd.param_groups:
            param_group['lr'] = opt.lr

        for param_group in optimizerE.param_groups:
            param_group['lr'] = opt.lr

stats = {}
batches_done=0
#populate_x(x, dataloader['train'])
for epoch in range(opt.start_epoch, opt.nepoch):

    # Adjust learning rate
    adjust_lr(epoch)

    for i in range(len(dataloader['train'])):
        batches_done=batches_done+1
        # ---------------------------
        #        Optimize over d
        # ---------------------------
        for d_iter in range(updates['d']['num_updates']):
            netd.zero_grad()
            populate_x(x, dataloader['train'])
            populate_x(x2, dataloader['train'])
            encode1=netE(x)
            encode2=netE(x2)
            alpha=torch.rand(x.shape[0],1).cuda()
            alpha = 0.5 - torch.abs(alpha - 0.5)  # Make interval [0, 0.5]
            encode_mix = alpha * encode1 + (1 - alpha) * encode2
            if opt.embedding == 'sphere':
                encode_mix=normalize(encode_mix)
            x_alpha = netG(encode_mix)
            AE = netG(encode1)
            populate_z(z, opt)
            xz = netd(netG(netg(z.squeeze())).detach())
            loss_disc = torch.mean((netd(x_alpha.detach()) - alpha.squeeze()-opt.reg).pow(2))+torch.mean((xz - 0.7).pow(2))
            loss_disc_real = torch.mean((netd(AE.detach() + opt.reg * (x - AE.detach()))-opt.reg).pow(2))+torch.mean(netd(x).pow(2))
            #loss_ae_disc = torch.mean(torch.square(netd(x_alpha)))
            d_loss=loss_disc+loss_disc_real
            stats['d_loss'] = d_loss
            d_loss.backward()
            optimizerd.step()
        # ---------------------------
        #        Optimize over AE
        # ---------------------------
        for ae_iter in range(updates['G']['num_updates']):
            #AE_losses = []
            netE.zero_grad()
            netG.zero_grad()
            # X
            #populate_x(x, dataloader['train'])
            # E(X)
            #Ex = netE(x)
            #AE = netG(Ex)
            loss_ae_disc = torch.mean((netd(x_alpha)).pow(2))
            loss_ae_real = torch.mean(netd(AE).pow(2))
            err = match(AE, x, opt.match_x)
            AE_loss=err+loss_ae_disc+loss_ae_real
            stats['AE_loss'] = AE_loss
            stats['err'] = err
            err.backward()
            optimizerE.step()
            optimizerG.step()


        # ---------------------------
        #        Optimize over D
        # ---------------------------

        for D_iter in range(updates['D']['num_updates']):
            netD.zero_grad()

            # X
            populate_x(x, dataloader['train'])
            #populate_x(x2, dataloader['train'])
            #E(x1),E(x2)
            encode1 = netE(x)
            D_real = netD(encode1)
            #encode2 = netE(x2)
            #alpha = torch.rand(x.shape[0], 5,1).cuda()
            #alpha = 0.5 - torch.abs(alpha - 0.5)  # Make interval [0, 0.5]
            #encode_mix = alpha * encode1.unsqueeze(1) + (1 - alpha) * encode2.unsqueeze(1)
            #encode_mix=encode_mix.view(-1,opt.nemb)
            #if opt.embedding == 'sphere':
               # encode_mix = normalize(encode_mix)
            # Z
            #D_mix=netD(encode_mix)
            #D_real_all = torch.cat([D_real,D_mix],0)
            populate_z(z, opt)
            D_fake = netD(netg(z.squeeze()).detach())
            r_logit_mean, f_logit_mean, D_loss = hinge_loss_discriminator(r_logit=D_real, f_logit=D_fake)
            # Save some stats
            stats['D_loss'] = D_loss

            D_loss.backward()
            optimizerD.step()

        # ---------------------------
        #        Minimize over  g e
        # ---------------------------

        for g_iter in range(updates['e']['num_updates']):
            #netE.zero_grad()
            netg.zero_grad()
            nete.zero_grad()

            # Z
            populate_z(z, opt)
            # Gg(Z)
            g=netg(z.squeeze())
            g_fake = netD(g)
            gx_fake=netd(netG(g))
            # X
            #populate_x(x, dataloader['train'])
            # E(X)
            #Ex = netE(x)
            #g_real = netD(Ex)
            G_f_logit_mean, g_loss = hinge_loss_generator2(f_logit=g_fake)
            loss_recon = torch.mean(gx_fake.pow(2))
            egz=nete(netg(z.squeeze()))
            err = match(egz, z, opt.match_z)
            g_loss=g_loss+err*2+loss_recon
            stats['g_loss'] = g_loss
            stats['err'] = err
            # Step g
            g_loss.backward()
            #optimizerE.step()
            optimizerg.step()
            optimizere.step()



        print('[{epoch}/{nepoch}][{iter}/{niter}] '
              'D_loss/g_loss: {D_loss:.3f}/{g_loss:.3f} '
              'AE_loss/err: {AE_loss:.3f}/{err:.3f}'
              'd_loss: {d_loss:.3f}'
              ''.format(epoch=epoch,
                        nepoch=opt.nepoch,
                        iter=i,
                        niter=len(dataloader['train']),
                        **stats))

        if i % opt.save_every == 0:
            writer.add_scalar('d_loss', stats['d_loss'], batches_done)
            writer.add_scalar('AE_loss', stats['AE_loss'], batches_done)
            writer.add_scalar('err', stats['err'], batches_done)
            writer.add_scalar('D_loss',stats['D_loss'],batches_done)
            writer.add_scalar('g_loss', stats['g_loss'], batches_done)

        if i % opt.save_every == 0 and epoch % 10 ==9:
            save_images(epoch)

        # If an epoch takes long time, dump intermediate
        # if opt.dataset in ['lsun', 'imagenet'] and (i % 5000 == 0):
        #     torch.save(netG, '%s/netG_epoch_%d_it_%d.pth' %
        #                (opt.save_dir, epoch, i))
        #     torch.save(netE, '%s/netE_epoch_%d_it_%d.pth' %
        #                (opt.save_dir, epoch, i))


    # do checkpointing
    if epoch % 10 ==9:
        torch.save(netG, '%s/netG_epoch_%d.pth' % (opt.save_dir, epoch))
        torch.save(netE, '%s/netE_epoch_%d.pth' % (opt.save_dir, epoch))
        torch.save(nete, '%s/nete_epoch_%d.pth' % (opt.save_dir, epoch))
        torch.save(netg, '%s/netg_epoch_%d.pth' % (opt.save_dir, epoch))
        torch.save(netD, '%s/netD_epoch_%d.pth' % (opt.save_dir, epoch))
        torch.save(netd, '%s/netd_epoch_%d.pth' % (opt.save_dir, epoch))

