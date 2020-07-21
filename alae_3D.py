from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
from tensorboardX import SummaryWriter
import torch.optim as optim
import matplotlib.pyplot as plt
from latent_3d_points.src import general_utils

import shutil
import torchvision.utils as vutils
from torch.autograd import Variable
from src.utils import *
import src.losses as losses

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='celeba',
                    help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--data_type', required=False, default='swiss_roll',
                    help='ball| swiss_roll ')
parser.add_argument('--dataroot', type=str, help='path to dataset',default='./data')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=0)
parser.add_argument('--batch_size', type=int,
                    default=5000, help='batch size')
parser.add_argument('--image_size', type=int, default=64,
                    help='the resolution of the input image to network')
parser.add_argument('--nz', type=int, default=2,
                    help='size of the latent z vector')
parser.add_argument('--nemb', type=int, default=2,
                    help='size of the latent embedding')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int,default=3)
parser.add_argument('--reg', type=int,default=0.2)

parser.add_argument('--nepoch', type=int, default=150,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cpu', action='store_true',
                    help='use CPU instead of GPU')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')

parser.add_argument('--netG', default='fcgan3D',
                    help="path to netG config")
parser.add_argument('--netE', default='fcgan3D',
                    help="path to netE config")
parser.add_argument('--netg', default='fcgan3D',
                    help="path to netg config")
parser.add_argument('--nete', default='fcgan3D',
                    help="path to nete config")
parser.add_argument('--netD', default='dcgan32px',
                    help="path to netD config")
parser.add_argument('--netd', default='dcgan64px',
                    help="path to netd config")
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
parser.add_argument('--save_dir', default='./results_3D/alae',
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
os.makedirs('./results_3D/alae',exist_ok=True)
os.makedirs('./results_3D/alae/tb',exist_ok=True)
writer=SummaryWriter(log_dir='./results_3D/alae/tb')



# Setup cudnn, seed, and parses updates string.
updates = setup(opt)

# Setup dataset
#dataloader = dict(train=setup_dataset(opt, train=True),
                  #val=setup_dataset(opt, train=False))
pclouds=data_load(opt)
# Load generator
netG = load_G(opt)

# Load encoder
netE = load_E(opt)

# Load generator_latent
netg = load_g(opt).to('cuda')


netD = load_D(opt).to('cuda')
x = torch.FloatTensor(5000,opt.nc)
z = torch.FloatTensor(5000, opt.nz, 1, 1)
fixed_z = torch.FloatTensor(opt.batch_size, opt.nz, 1, 1).normal_(0, 1)

if opt.noise == 'sphere':
    normalize_(fixed_z)

if opt.cuda:
    netE.cuda()
    netG.cuda()
    x = x.cuda()
    z, fixed_z = z.cuda(), fixed_z.cuda()

x = Variable(x,requires_grad=True)
z = Variable(z)
fixed_z = Variable(fixed_z)

# Setup optimizers
optimizerE = optim.Adam(netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerg = optim.Adam(netg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

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
    z2 = torch.FloatTensor(5000, opt.nz, 1, 1).normal_(0, 1).cuda()
    if opt.noise == 'sphere':
        normalize_(z2)
    fig = plt.figure(figsize=(5 * 4, 5 * 1))
    fig.tight_layout()
    ori_ax = fig.add_subplot(1, 4, 1, projection='3d')
    real_cpu.resize_(x.data.size()).copy_(x.data)
    general_utils.plot_3d_point_cloud(real_cpu.squeeze().cpu().numpy()[:, 0],
                        real_cpu.squeeze().cpu().numpy()[:, 1],
                        real_cpu.squeeze().cpu().numpy()[:, 2],
                        axis=ori_ax, in_u_sphere=False, show=False, alpha=0.1)
    ori_ax.set_title('original')
    gen_ax = fig.add_subplot(1, 4, 2, projection='3d')
    netG.eval()
    netg.eval()
    fake = netG(netg(z2.squeeze()))
    general_utils.plot_3d_point_cloud(fake.detach().cpu().numpy()[:, 0],
                                      fake.detach().cpu().numpy()[:, 1],
                                      fake.detach().cpu().numpy()[:, 2],
                                      axis=gen_ax, in_u_sphere=False, show=False, alpha=0.1)
    gen_ax.set_title('generate')
    # Save reconstructions
    #populate_x(x, dataloader['val'])
    netE.eval()
    netD.eval()
    GEx = netG((netE(x)))
    recon_ax = fig.add_subplot(1, 4, 3, projection='3d')
    general_utils.plot_3d_point_cloud(GEx.detach().cpu().numpy()[:, 0],
                                      GEx.detach().cpu().numpy()[:, 1],
                                      GEx.detach().cpu().numpy()[:, 2],
                                      axis=recon_ax, in_u_sphere=False, show=False, alpha=0.1)
    recon_ax.set_title('recon')
    fig.savefig("%s/fig_epoch_%03d.png" % (opt.save_dir, epoch))
    fig2 = plt.figure(figsize=(5 * 3, 5 * 1))
    fig2.tight_layout()
    ax1= fig2.add_subplot(131)
    gz=netg(z2.squeeze()).cpu().detach().numpy()
    sc1 = ax1.scatter(gz[:, 0], gz[:, 1], s=10)

    ax1.set_title('gz')
    ax2 = fig2.add_subplot(132)
    Ex = netE(x).cpu().detach().numpy()
    sc2 = ax2.scatter(Ex[:, 0], Ex[:, 1], s=10)

    ax2.set_title('Ex')
    fig2.savefig("%s/figemb_epoch_%03d.png" % (opt.save_dir, epoch))
def adjust_lr(epoch):
    if epoch % opt.drop_lr == (opt.drop_lr - 1):
        opt.lr /= 2
        for param_group in optimizerD.param_groups:
            param_group['lr'] = opt.lr

        for param_group in optimizerG.param_groups:
            param_group['lr'] = opt.lr

        for param_group in optimizerE.param_groups:
            param_group['lr'] = opt.lr

        for param_group in optimizerg.param_groups:
            param_group['lr'] = opt.lr
def plot_embedding(Ex,eEx,z,gz,dir):
    Ex=Ex.cpu().detach().numpy()
    eEx=eEx.cpu().detach().numpy()
    z=z.cpu().detach().numpy()
    gz=gz.cpu().detach().numpy()
    fig = plt.figure()
    ax1 = plt.subplot(221)
    sc1 = ax1.scatter(Ex[:, 0], Ex[:, 1], s=10)
    ax1.set_title('Ex')
    ax2 = plt.subplot(222)
    sc2 = ax2.scatter(eEx[:, 0], eEx[:, 1], s=10)
    ax2.set_title('eEx')
    ax3 = plt.subplot(223)
    sc3 = ax3.scatter(z[:, 0], z[:, 1], s=10)
    ax3.set_title('z')
    ax4 = plt.subplot(224)
    sc4 = ax4.scatter(gz[:, 0], gz[:, 1], s=10)
    ax4.set_title('gz')
    #c1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.xticks([])
    plt.yticks([])
    #handles = [plt.plot([], color=sc.get_cmap()(sc.norm(c)), ls="", marker="o")[0] for c in c1]
    # c=[c for c in label]
    #plt.legend(handles, c1)
    # fig.show()

    fig.savefig("%s/latent" % (dir))
    return fig
def test_embedding(netE,netG):
    netE.eval()
    netG.eval()
    netg.eval()
    nete.eval()
    #populate_x(x, dataloader['train'])
    idx = torch.randperm(dataloader['train'].__len__())
    x,_=dataloader['val'].next()
    #x = x.permute(0, 3, 1, 2)
    for i in range(len(dataloader['val'])-1):
        real_cpu,_ = dataloader['val'].next()
        #real_cpu=real_cpu.permute(0,3,1,2)
        x=torch.cat((x,real_cpu),0)
    z=torch.randn(50000, opt.nz).to('cuda')
    normalize_(z)
    Ex = netE(x.cuda())
    eEx= nete(Ex)
    gz=netg(z)
    #Ggz=netG(netg(z))
    #os.mkdir('./results_loage/embedding',exist_ok=True)
    plot_embedding(Ex,eEx,z,gz, dir='./results_loage/embedding')
    exit(0)
#test_embedding(netE,netG)
stats = {}
batches_done=0
for epoch in range(opt.start_epoch, opt.nepoch):

    # Adjust learning rate
    adjust_lr(epoch)

    for i in range(500):
        batches_done=batches_done+1
        # ---------------------------
        #        Optimize over D,E
        # ---------------------------
        netE.zero_grad()
        netD.zero_grad()
        # X
        populate_x_3D(x, pclouds)
        # E(X)
        Ex = netE(x)
        DEx = netD(Ex)
        populate_z(z, opt)
        DEGg=netD(netE(netG(netg(z.squeeze()))))
        loss = (F.softplus(DEGg) + F.softplus(-DEx))
        real_loss = DEx.sum()
        real_grads = torch.autograd.grad(real_loss, x,create_graph=True, retain_graph=True)[0]
        r1_penalty = torch.sum(real_grads.pow(2.0), dim=1)
        d_loss = (loss + r1_penalty * (10 * 0.5)).mean()
        stats['d_loss']=d_loss
        d_loss.backward()
        optimizerD.step()
        optimizerE.step()
        # ---------------------------
        #        Optimize g,G
        # ---------------------------
        netG.zero_grad()
        netg.zero_grad()
        # z
        populate_z(z, opt)
        DEGg=netD(netE(netG(netg(z.squeeze()))))
        g_loss=F.softplus(-DEGg).mean()
        stats['g_loss'] = g_loss
        g_loss.backward()
        optimizerG.step()
        optimizerg.step()
        # ---------------------------
        #        Optimize E,G
        # ---------------------------

        netE.zero_grad()
        netG.zero_grad()
        populate_z(z, opt)
        EGg=netE(netG(netg(z.squeeze())))
        lae=match(netg(z.squeeze()), EGg, opt.match_z)
        stats['lae'] = lae*1000
        lae.backward()
        optimizerG.step()
        optimizerE.step()


        print('[{epoch}/{nepoch}][{iter}/{niter}] '
              'd_loss/g_loss: {d_loss:.5f}/{g_loss:.5f} '
              'lae: {lae:.5f}'
              ''.format(epoch=epoch,
                        nepoch=opt.nepoch,
                        iter=i,
                        niter=500,
                        **stats))

        if i % opt.save_every == 0:
            writer.add_scalar('d_loss',stats['d_loss'],batches_done)
            writer.add_scalar('g_loss', stats['g_loss'], batches_done)
            writer.add_scalar('lae', stats['lae'], batches_done)


        if i % opt.save_every == 0:
            save_images(epoch)

        # If an epoch takes long time, dump intermediate
        if opt.dataset in ['lsun', 'imagenet'] and (i % 5000 == 0):
            torch.save(netG, '%s/netG_epoch_%d_it_%d.pth' %
                       (opt.save_dir, epoch, i))
            torch.save(netE, '%s/netE_epoch_%d_it_%d.pth' %
                       (opt.save_dir, epoch, i))


    # do checkpointing
    torch.save(netG, '%s/netG_epoch_%d.pth' % (opt.save_dir, epoch))
    torch.save(netE, '%s/netE_epoch_%d.pth' % (opt.save_dir, epoch))
    torch.save(netD, '%s/netD_epoch_%d.pth' % (opt.save_dir, epoch))
    torch.save(netg, '%s/netg_epoch_%d.pth' % (opt.save_dir, epoch))

