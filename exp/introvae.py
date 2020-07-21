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


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False, default='celeba',
                        help='cifar10 | lsun | imagenet | folder | lfw ')
    parser.add_argument('--dataroot', type=str, help='path to dataset',
                        default=os.path.expanduser('~/data/celebA/img_align_celeba'))
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
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--reg', type=int, default=0.2)

    parser.add_argument('--nepoch', type=int, default=25,
                        help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0003,
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
    parser.add_argument('--netG_chp', default='',
                        help="path to netG (to continue training)")
    parser.add_argument('--netD_chp', default='',
                        help="path to netD (to continue training)")
    parser.add_argument('--vae_netE_chp', default='',
                        help="path to netE (to continue training)")
    # ./results_loage/netg_epoch_35.pth
    parser.add_argument('--netg_chp', default='',
                        help="path to netG (to continue training)")
    parser.add_argument('--nete_chp', default='',
                        help="path to netE (to continue training)")
    parser.add_argument('--netd_chp', default='',
                        help="path to netd (to continue training)")
    parser.add_argument('--save_dir', default='./results_celeba/vaegan_128',
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
    return parser


def save_images(epoch):
    real_cpu.resize_(x.data.size()).copy_(x.data)

    # Real samples
    save_path = '%s/real_samples.png' % opt.save_dir
    vutils.save_image(real_cpu[:64] / 2 + 0.5, save_path)

    netG.eval()
    populate_z(z, opt)
    fake = netG(z.squeeze())

    # Fake samples
    save_path = '%s/fake_samples_epoch_%03d.png' % (opt.save_dir, epoch)
    vutils.save_image(fake.data[:64] / 2 + 0.5, save_path)

    # Save reconstructions
    netE.eval()
    populate_x(x, dataloader['val'])
    populate_x(x2, dataloader['val'])
    alpha = torch.rand(x.shape[0], 1).cuda()
    alpha = 0.5 - torch.abs(alpha - 0.5)  # Make interval [0, 0.5]
    sample_z1, _, _ = netE(x)
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
    grid = vutils.save_image(t[:64] / 2 + 0.5, save_path)
    netG.train()
    netE.train()


def adjust_lr(epoch, opt, ):
    if epoch % opt.drop_lr == (opt.drop_lr - 1):
        opt.lr /= 2
        for param_group in optimizerE.param_groups:
            param_group['lr'] = opt.lr

        for param_group in optimizerG.param_groups:
            param_group['lr'] = opt.lr


def main(opt, myargs):


    # export CUDA_VISIBLE_DEVICES = 0

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
        x2 = x2.cuda()
        z, fixed_z = z.cuda(), fixed_z.cuda()

    x = Variable(x)
    z = Variable(z)
    x2 = Variable(x2)
    fixed_z = Variable(fixed_z)

    # Setup optimizers
    optimizerE = optim.Adam(netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    real_cpu = torch.FloatTensor()

    stats = {}
    batches_done = 0
    for epoch in range(opt.start_epoch, opt.nepoch):

        # Adjust learning rate
        adjust_lr(epoch)

        for i in range(len(dataloader['train'])):
            batches_done = batches_done + 1
            if epoch < opt.num_vae:
                # ---------------------------
                #        Optimize over E
                # ---------------------------
                netE.zero_grad()
                # X
                populate_x(x, dataloader['train'])
                # E(X)
                Ex, mu, logsigma = netE(x)
                AE = netG(Ex)
                # err = F.binary_cross_entropy_with_logits(AE, x,size_average=False).div(opt.batch_size)
                err = match(AE, x, opt.match_x)
                kl_loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
                loss = kl_loss * 0.0001 + err
                stats['AE_loss'] = err
                stats['KL_loss'] = kl_loss
                loss.backward()
                optimizerE.step()
            # ---------------------------
            #        Optimize over G
            # ---------------------------
            netG.zero_grad()
            populate_x(x, dataloader['train'])
            Ex, _, _ = netE(x)
            AE = netG(Ex)
            feature_AE, fake_AE = netd(AE)
            featrue_x, _ = netd(x)
            err = match(feature_AE, featrue_x, opt.match_x)
            populate_z(z, opt)
            Gz = netG(z.squeeze())
            _, fake_z = netd(Gz)
            fake = torch.cat([fake_z, fake_AE], 0)
            G_f_logit_mean, g_loss = hinge_loss_generator2(f_logit=fake)
            G_loss = err * 0.01 + g_loss
            stats['G_loss'] = G_loss
            stats['g_loss'] = g_loss
            G_loss.backward()
            optimizerG.step()
            # ---------------------------
            #        Optimize over d
            # ---------------------------
            netd.zero_grad()
            _, real = netd(x)
            Gz = netG(z.squeeze())
            _, fake_z = netd(Gz)
            Ex, mu, logsigma = netE(x)
            AE = netG(Ex)
            feature_AE, fake_AE = netd(AE)
            fake = torch.cat([fake_z, fake_AE], 0)
            r_logit_mean, f_logit_mean, d_loss = hinge_loss_discriminator(r_logit=real, f_logit=fake)
            stats['d_loss'] = d_loss
            d_loss.backward()
            optimizerd.step()

            print('[{epoch}/{nepoch}][{iter}/{niter}] '
                  'KL_loss/AE_loss: {KL_loss:.3f}/{AE_loss:.3f}'
                  'G_loss/d_loss: {G_loss:.3f}/{d_loss:.3f}'
                  ''.format(epoch=epoch,
                            nepoch=opt.nepoch,
                            iter=i,
                            niter=len(dataloader['train']),
                            **stats))

            if i % opt.save_every == 0:
                writer.add_scalar('AE_loss', stats['AE_loss'], batches_done)
                writer.add_scalar('KL_loss', stats['KL_loss'], batches_done)
                writer.add_scalar('G_loss', stats['G_loss'], batches_done)
                writer.add_scalar('d_loss', stats['d_loss'], batches_done)

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
        torch.save(netG, '%s/netd_epoch_%d.pth' % (opt.save_dir, epoch))


def run(argv_str=None):
    from template_lib.utils.config import parse_args_and_setup_myargs, config2args
    from template_lib.utils.modelarts_utils import prepare_dataset
    run_script = os.path.relpath(__file__, os.getcwd())
    args1, myargs, _ = parse_args_and_setup_myargs(argv_str, run_script=run_script, start_tb=False)
    myargs.args = args1
    myargs.config = getattr(myargs.config, args1.command)

    if hasattr(myargs.config, 'datasets'):
        prepare_dataset(myargs.config.datasets, cfg=myargs.config)

    parser = build_parser()
    opt = parser.parse_args([])

    opt = config2args(myargs.config.args, opt)

    main(opt, myargs)


if __name__ == '__main__':
    run()
