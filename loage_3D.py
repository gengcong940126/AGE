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
parser.add_argument('--dataset', required=False, default='cifar10',
                    help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--data_type', required=False, default='swiss_roll',
                    help='ball| swiss_roll ')
parser.add_argument('--dataroot', type=str, help='path to dataset',default='./data')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=0)
parser.add_argument('--batch_size', type=int,
                    default=64, help='batch size')
parser.add_argument('--image_size', type=int, default=32,
                    help='the resolution of the input image to network')
parser.add_argument('--nz', type=int, default=2,
                    help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int,default=3)

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
parser.add_argument('--netG_chp', default='',
                    help="path to netG (to continue training)")
parser.add_argument('--netE_chp', default='',
                    help="path to netE (to continue training)")
#./results_loage/netg_epoch_35.pth
parser.add_argument('--netg_chp', default='',
                    help="path to netG (to continue training)")
parser.add_argument('--nete_chp', default='',
                    help="path to netE (to continue training)")
parser.add_argument('--save_dir', default='./results_loage_3D',
                    help='folder to output images and model checkpoints')
parser.add_argument('--criterion', default='param',
                    help='param|nonparam, How to estimate KL')
parser.add_argument('--KL', default='qp', help='pq|qp')
parser.add_argument('--noise', default='normal', help='normal|sphere')
parser.add_argument('--match_z', default='L2', help='none|L1|L2|cos')
parser.add_argument('--match_x', default='L1', help='none|L1|L2|cos')

parser.add_argument('--drop_lr', default=40, type=int, help='')
parser.add_argument('--save_every', default=250, type=int, help='')

parser.add_argument('--manual_seed', type=int, default=123, help='manual seed')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number to start with')

parser.add_argument(
    '--E_updates', default="1;KL_fake:1,KL_real:1,match_z:0,match_x:100",
    help='Update plan for encoder <number of updates>;[<term:weight>]'
)

parser.add_argument(
    '--G_updates', default="2;KL_fake:1,match_z:10,match_x:0",
    help='Update plan for generator <number of updates>;[<term:weight>]'
)
parser.add_argument(
    '--e_updates', default="1;KL_fake:1,KL_real:1,match_z:0,match_x:10",
    help='Update plan for encoder <number of updates>;[<term:weight>]'
)

parser.add_argument(
    '--g_updates', default="2;KL_fake:1,match_z:10,match_x:0",
    help='Update plan for generator <number of updates>;[<term:weight>]'
)
opt = parser.parse_args()
os.makedirs('./results_loage_3D',exist_ok=True)
os.makedirs('./results_loage_3D/tb',exist_ok=True)
writer=SummaryWriter(log_dir='./results_loage/tb')
if 'PORT' not in os.environ:
    os.environ['PORT'] = '6007'


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

# Load encoder_latent
nete = load_e(opt).to('cuda')

x = torch.FloatTensor(pclouds).squeeze()
z = torch.FloatTensor(opt.batch_size, opt.nz, 1, 1)
fixed_z = torch.FloatTensor(opt.batch_size, opt.nz, 1, 1).normal_(0, 1)

if opt.noise == 'sphere':
    normalize_(fixed_z)

if opt.cuda:
    netE.cuda()
    netG.cuda()
    x = x.cuda()
    z, fixed_z = z.cuda(), fixed_z.cuda()

x = Variable(x)
z = Variable(z)
fixed_z = Variable(fixed_z)

# Setup optimizers
optimizerD = optim.Adam(netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerg = optim.Adam(netg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizere = optim.Adam(nete.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

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
    fig = plt.figure(figsize=(5 * 3, 5 * 1))
    fig.tight_layout()
    ori_ax = fig.add_subplot(1, 3, 1, projection='3d')
    real_cpu.resize_(x.data.size()).copy_(x.data)
    general_utils.plot_3d_point_cloud(real_cpu.squeeze().cpu().numpy()[:, 0],
                        real_cpu.squeeze().cpu().numpy()[:, 1],
                        real_cpu.squeeze().cpu().numpy()[:, 2],
                        axis=ori_ax, in_u_sphere=False, show=False, alpha=0.1)
    ori_ax.set_title('original')
    gen_ax = fig.add_subplot(1, 3, 2, projection='3d')
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
    nete.eval()
    gex = netG(netg((nete((netE(x))))))
    recon_ax = fig.add_subplot(1, 3, 3, projection='3d')
    general_utils.plot_3d_point_cloud(gex.detach().cpu().numpy()[:, 0],
                                      gex.detach().cpu().numpy()[:, 1],
                                      gex.detach().cpu().numpy()[:, 2],
                                      axis=recon_ax, in_u_sphere=False, show=False, alpha=0.1)
    recon_ax.set_title('recon')
    fig.savefig("%s/fig_epoch_%03d.png" % (opt.save_dir, epoch))

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
        #        Optimize over AE
        # ---------------------------
        for ae_iter in range(3):
            #AE_losses = []
            netE.zero_grad()
            netG.zero_grad()
            # X
            #populate_x(x, dataloader['train'])
            # E(X)
            Ex = netE(x)
            GEx = netG(Ex)
            err = match(GEx, x, opt.match_x)
            AE_losses=err * 10
            AE_losses.backward()
            optimizerD.step()
            optimizerG.step()
            # ---------------------------
            #        Optimize over AE_latent
            # ---------------------------
        for ae_iter in range(3):
            AE_losses = []
            nete.zero_grad()
            netg.zero_grad()
            # z
            populate_z(z, opt)
            Ggz = netG(netg(z.squeeze()))
            eEGgz = nete(netE((Ggz)))
            err = match(eEGgz, z, opt.match_z)
            AE_losses.append( err * 10)
            # x
            #populate_x(x, dataloader['train'])
            eEx=nete(netE(x))
            GgeEx=netG(netg(eEx))
            err2 = match(GgeEx, x, opt.match_x)
            AE_losses.append(err2 * 10)
            sum(AE_losses).backward()
            optimizere.step()
            optimizerg.step()
        # ---------------------------
        #        Optimize over E
        # ---------------------------

        for e_iter in range(updates['E']['num_updates']):
            E_losses = []
            netE.zero_grad()

            # X
            #populate_x(x, dataloader['train'])
            # E(X)
            Ex = netE(x)

            # KL_real: - \Delta( E(X) , Z ) -> max_E
            KL_real = KL_minimizer(Ex)
            E_losses.append(KL_real * updates['E']['KL_real'])

            if updates['E']['match_x'] != 0:
                # g(e(X))
                GEx = netG(Ex)

                # match_x: E_x||G(E(x)) - x|| -> min_E
                err = match(GEx, x, opt.match_x)
                E_losses.append(err * updates['E']['match_x'])

            # Save some stats
            stats['real_mean'] = KL_minimizer.samples_mean.data.mean()
            stats['real_var'] = KL_minimizer.samples_var.data.mean()
            stats['KL_real'] = KL_real.data

            # ================================================

            # Z
            populate_z(z, opt)
            # Gg(Z)
            fake = netG(netg(z.squeeze())).detach()
            # E(G(g(Z)))
            EGgz = netE(fake)

            # KL_fake: \Delta( EGgz , Z ) -> max_e
            KL_fake = KL_maximizer(EGgz)
            E_losses.append(KL_fake * updates['E']['KL_fake'])

            if updates['E']['match_z'] != 0:
                # match_z: E_z||EGgz - g(z)|| -> min_e
                err = match(EGgz, netg(z), opt.match_z)
                E_losses.append(err * updates['E']['match_z'])

            # Save some stats
            stats['fake_mean'] = KL_maximizer.samples_mean.data.mean()
            stats['fake_var'] = KL_maximizer.samples_var.data.mean()
            stats['KL_fake'] = -KL_fake.data

            # Update E
            sum(E_losses).backward()
            optimizerD.step()

        # ---------------------------
        #        Minimize over G g
        # ---------------------------

        for g_iter in range(updates['G']['num_updates']):
            G_losses = []
            netG.zero_grad()
            netg.zero_grad()
            # Z
            populate_z(z, opt)
            # Gg(Z)
            fake = netG(netg((z.squeeze())))
            # E(G(g(Z)))
            EGgz = netE(fake)

            # KL_fake: \Delta( EGgz , Z ) -> min_G
            KL_fake_g = KL_minimizer(EGgz)
            G_losses.append(KL_fake_g * updates['G']['KL_fake'])

            if updates['G']['match_z'] != 0:
                # match_z: E_z||EGgz - g(z)|| -> min_G
                err = match(EGgz, netg(z.squeeze()), opt.match_z)
                err = err * updates['G']['match_z']
                G_losses.append(err)

            # ==================================

            if updates['G']['match_x'] != 0:
                # X
                #populate_x(x, dataloader['train'])
                # e(X)
                Ex = netE(x)

                # GEx
                GEx = netG(Ex)

                # match_x: E_x||GEx - x|| -> min_G
                err = match(GEx, x, opt.match_x)
                err = err * updates['G']['match_x']
                G_losses.append(err)

            # Step g
            sum(G_losses).backward()
            optimizerG.step()
            optimizerg.step()


        print('[{epoch}/{nepoch}][{iter}/{niter}] '
              'KL_real/fake: {KL_real:.3f}/{KL_fake:.3f} '
              'mean_real/fake: {real_mean:.3f}/{fake_mean:.3f} '
              'var_real/fake: {real_var:.3f}/{fake_var:.3f} '
              ''.format(epoch=epoch,
                        nepoch=opt.nepoch,
                        iter=i,
                        niter=500,
                        **stats))

        if i % opt.save_every == 0:
            writer.add_scalar('KL_real',stats['KL_real'],batches_done)
            writer.add_scalar('KL_fake', stats['KL_real'], batches_done)
        # ---------------------------
        #        Optimize over e
        # ---------------------------

        for e_iter in range(updates['e']['num_updates']):
            e_losses = []
            nete.zero_grad()

            # X
            #populate_x(x, dataloader['train'])
            # e(E(X))
            eEx = nete(netE(x))

            # KL_real: - \Delta(eEx , Z ) -> max_e
            KL_real = KL_minimizer(eEx)
            e_losses.append(KL_real * updates['e']['KL_real'])

            if updates['e']['match_x'] != 0:
                # g(e(E(X)))
                geEx = netg(eEx)

                # match_x: E_x||geEx - Ex|| -> min_e
                err = match(geEx, netE(x), opt.match_x)
                e_losses.append(err * updates['e']['match_x'])

            # Save some stats
            stats['real_mean_latent'] = KL_minimizer.samples_mean.data.mean()
            stats['real_var_latent'] = KL_minimizer.samples_var.data.mean()
            stats['KL_real_latent'] = KL_real.data

            # ================================================

            # Z
            populate_z(z, opt)
            # g(Z)
            fake = netg(z.squeeze()).detach()
            # e(g(Z))
            egz = nete(fake)

            # KL_fake: \Delta( e(g(Z)) , Z ) -> max_e
            KL_fake = KL_maximizer(egz)
            e_losses.append(KL_fake * updates['e']['KL_fake'])

            if updates['e']['match_z'] != 0:
                # match_z: E_z||e(g(z)) - z|| -> min_e
                err = match(egz, z, opt.match_z)
                e_losses.append(err * updates['e']['match_z'])

            # Save some stats
            stats['fake_mean_latent'] = KL_maximizer.samples_mean.data.mean()
            stats['fake_var_latent'] = KL_maximizer.samples_var.data.mean()
            stats['KL_fake_latent'] = -KL_fake.data

            # Update e
            sum(e_losses).backward()
            optimizere.step()

        # ---------------------------
        #        Minimize over g
        # ---------------------------

        for g_iter in range(updates['g']['num_updates']):
            g_losses = []
            netg.zero_grad()

            # Z
            populate_z(z, opt)
            # g(Z)
            fake = netg(z.squeeze())
            # e(g(Z))
            egz = nete(fake)

            # KL_fake: \Delta( e(g(Z)) , Z ) -> min_g
            KL_fake_g = KL_minimizer(egz)
            g_losses.append(KL_fake_g * updates['g']['KL_fake'])

            if updates['g']['match_z'] != 0:
                # match_z: E_z||e(g(z)) - z|| -> min_g
                err = match(egz, z, opt.match_z)
                err = err * updates['g']['match_z']
                g_losses.append(err)

            # ==================================

            if updates['g']['match_x'] != 0:
                # X
                #populate_x(x, dataloader['train'])
                # E(X)
                Ex = netE(x)

                # g(e(E(X)))
                eEx = nete(Ex)
                geEx = netg(eEx)

                # match_x: E_x||geEx - Ex|| -> min_g
                err = match(geEx, Ex, opt.match_x)
                err = err * updates['g']['match_x']
                g_losses.append(err)

            # Step g
            sum(g_losses).backward()
            optimizerg.step()

        print('[{epoch}/{nepoch}][{iter}/{niter}] '
              '_latent'
              'KL_real/fake: {KL_real_latent:.3f}/{KL_fake_latent:.3f} '
              'mean_real/fake: {real_mean_latent:.3f}/{fake_mean_latent:.3f} '
              'var_real/fake: {real_var_latent:.3f}/{fake_var_latent:.3f} '
              ''.format(epoch=epoch,
                        nepoch=opt.nepoch,
                        iter=i,
                        niter=500,
                        **stats))
        if i % opt.save_every == 0:
            writer.add_scalar('KL_real_latent',stats['KL_real_latent'],batches_done)
            writer.add_scalar('KL_fake_latent', stats['KL_real_latent'], batches_done)
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
    torch.save(nete, '%s/nete_epoch_%d.pth' % (opt.save_dir, epoch))
    torch.save(netg, '%s/netg_epoch_%d.pth' % (opt.save_dir, epoch))

