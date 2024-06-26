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
parser.add_argument('--dataset', required=False, default='mnist')
parser.add_argument('--dataroot', type=str, help='path to dataset',default='./data')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=0)
parser.add_argument('--batch_size', type=int,
                    default=64, help='batch size')
parser.add_argument('--image_size', type=int, default=32,
                    help='the resolution of the input image to network')
parser.add_argument('--nz', type=int, default=32,
                    help='size of the latent z vector')
parser.add_argument('--nemb', type=int, default=256,
                    help='size of the latent embedding')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int,default=3)
parser.add_argument('--reg', type=int,default=0.2)

parser.add_argument('--nepoch', type=int, default=500,
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
parser.add_argument('--save_dir', default='./results_AEGAN_acai_latent_normal',
                    help='folder to output images and model checkpoints')
parser.add_argument('--criterion', default='param',
                    help='param|nonparam, How to estimate KL')
parser.add_argument('--KL', default='qp', help='pq|qp')
parser.add_argument('--noise', default='normal', help='normal|sphere')
parser.add_argument('--match_z', default='L2', help='none|L1|L2|cos')
parser.add_argument('--match_x', default='L1', help='none|L1|L2|cos')

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
os.makedirs('./results_AEGAN_acai_latent_normal',exist_ok=True)
os.makedirs('./results_AEGAN_acai_latent_normal/tb',exist_ok=True)
writer=SummaryWriter(log_dir='./results_AEGAN_acai_latent_normal/tb')
if 'PORT' not in os.environ:
    os.environ['PORT'] = '6006'
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

    netG.eval()
    netg.eval()
    populate_z(z, opt)
    fake = netG(netg(z.squeeze()))

    # Fake samples
    save_path = '%s/fake_samples_epoch_%03d.png' % (opt.save_dir, epoch)
    vutils.save_image(fake.data[:64]/2+0.5 , save_path)

    # Save reconstructions
    netE.eval()
    populate_x(x, dataloader['val'])
    populate_x(x2, dataloader['val'])
    alpha = torch.rand(x.shape[0], 1).cuda()
    alpha = 0.5 - torch.abs(alpha - 0.5)  # Make interval [0, 0.5]
    encode_mix = alpha * netE(x) + (1 - alpha) * netE(x2)
    if opt.noise == 'sphere':
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
    netG.train()
    netg.train()
    netE.train()
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
            if opt.noise == 'sphere':
                encode_mix=normalize(encode_mix)
            x_alpha = netG(encode_mix)
            AE = netG(encode1)
            loss_disc = torch.mean((netd(x_alpha.detach()) - alpha.squeeze()).pow(2))
            loss_disc_real = torch.mean((netd(AE.detach() + opt.reg * (x - AE.detach()))).pow(2))
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
            #GEx = netG(Ex)
            loss_ae_disc = torch.mean((netd(x_alpha)).pow(2))
            err = match(AE, x, opt.match_x)
            AE_loss=err+0.5*loss_ae_disc
            stats['AE_loss'] = AE_loss
            AE_loss.backward()
            optimizerE.step()
            optimizerG.step()


        # ---------------------------
        #        Optimize over D
        # ---------------------------

        for D_iter in range(updates['D']['num_updates']):
            netD.zero_grad()

            # X
            populate_x(x, dataloader['train'])
            populate_x(x2, dataloader['train'])
            #E(x1),E(x2)
            encode1 = netE(x)
            D_real = netD(encode1)
            encode2 = netE(x2)
            alpha = torch.rand(x.shape[0],3, 1).cuda()
            alpha = 0.5 - torch.abs(alpha - 0.5)  # Make interval [0, 0.5]
            encode_mix = alpha * encode1.unsqueeze(1) + (1 - alpha) * encode2.unsqueeze(1)
            encode_mix=encode_mix.view(-1,opt.nz)
            if opt.noise == 'sphere':
                encode_mix = normalize(encode_mix)
            # Z
            D_mix=netD(encode_mix)
            D_real_all = torch.cat([D_real,D_mix],0)
            populate_z(z, opt)
            D_fake = netD(netg(z.squeeze()).detach())
            loss_disc = torch.mean((D_mix - alpha.view(-1,1)).pow(2))+torch.mean((D_fake - 1).pow(2))
            loss_disc_real = torch.mean(D_real.pow(2))
            D_loss=loss_disc+loss_disc_real
            #r_logit_mean, f_logit_mean, D_loss = hinge_loss_discriminator(r_logit=D_real_all, f_logit=D_fake)
            # Save some stats
            stats['D_loss'] = D_loss

            D_loss.backward()
            optimizerD.step()

        # ---------------------------
        #        Minimize over  g e
        # ---------------------------

        for g_iter in range(updates['G']['num_updates']):
            #netE.zero_grad()
            netg.zero_grad()
            nete.zero_grad()

            # Z
            populate_z(z, opt)
            # Gg(Z)
            g_fake = netD(netg((z.squeeze())))
            # X
            #populate_x(x, dataloader['train'])
            # E(X)
            #Ex = netE(x)
            #g_real = netD(Ex)
            #G_f_logit_mean, g_loss = hinge_loss_generator2(f_logit=g_fake)
            g_loss=torch.mean(g_fake.pow(2))
            egz=nete(netg(z.squeeze()))
            #err = match(egz, z, opt.match_z)
            g_loss=g_loss
            stats['g_loss'] = g_loss
            #stats['err'] = err
            # Step g
            g_loss.backward()
            #optimizerE.step()
            optimizerg.step()
            optimizere.step()



        print('[{epoch}/{nepoch}][{iter}/{niter}] '
              'D_loss/g_loss: {D_loss:.3f}/{g_loss:.3f} '
              'AE_loss: {AE_loss:.3f}'
              'd_loss: {d_loss:.3f}'
              ''.format(epoch=epoch,
                        nepoch=opt.nepoch,
                        iter=i,
                        niter=len(dataloader['train']),
                        **stats))

        if i % opt.save_every == 0:
            print(batches_done)
            writer.add_scalar('d_loss', stats['d_loss'], batches_done)
            writer.add_scalar('AE_loss', stats['AE_loss'], batches_done)
            writer.add_scalar('D_loss',stats['D_loss'],batches_done)
            writer.add_scalar('g_loss', stats['g_loss'], batches_done)

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
    torch.save(netD, '%s/netD_epoch_%d.pth' % (opt.save_dir, epoch))
    torch.save(netd, '%s/netd_epoch_%d.pth' % (opt.save_dir, epoch))

