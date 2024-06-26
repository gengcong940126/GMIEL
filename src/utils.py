import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data
import torch.nn.functional as F
import importlib
import numpy as np
from .dataset import FolderWithImages
from sklearn.datasets import make_swiss_roll
import random
import os
import torch.backends.cudnn as cudnn
from PIL import Image


def setup(opt):
    '''
    Setups cudnn, seeds and parses updates string.
    '''
    opt.cuda = not opt.cpu

    torch.set_num_threads(4)

    if opt.nc is None:
        opt.nc = 1 if opt.dataset == 'mnist' else 3

    try:
        os.makedirs(opt.save_dir)
    except OSError:
        print('Directory was not created.')

    if opt.manual_seed is None:
        opt.manual_seed = random.randint(1, 10000)

    print("Random Seed: ", opt.manual_seed)
    random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed_all(opt.manual_seed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device,"
              "so you should probably run with --cuda")

    updates = {'E': {}, 'G': {},'e': {}, 'g':{}, 'D':{},'d':{}}
    updates['D']['num_updates'] = int(opt.D_updates.split(';')[0])
    updates['D'].update({x.split(':')[0]: float(x.split(':')[1])
                         for x in opt.D_updates.split(';')[1].split(',')})

    updates['G']['num_updates'] = int(opt.G_updates.split(';')[0])
    updates['G'].update({x.split(':')[0]: float(x.split(':')[1])
                         for x in opt.G_updates.split(';')[1].split(',')})
    updates['e']['num_updates'] = int(opt.e_updates.split(';')[0])
    updates['e'].update({x.split(':')[0]: float(x.split(':')[1])
                         for x in opt.e_updates.split(';')[1].split(',')})

    updates['g']['num_updates'] = int(opt.g_updates.split(';')[0])
    updates['g'].update({x.split(':')[0]: float(x.split(':')[1])
                         for x in opt.g_updates.split(';')[1].split(',')})

    updates['d']['num_updates'] = int(opt.d_updates.split(';')[0])
    updates['d'].update({x.split(':')[0]: float(x.split(':')[1])
                         for x in opt.d_updates.split(';')[1].split(',')})
    return updates


def setup_dataset(opt, train=True, shuffle=True, drop_last=True):
    '''
    Setups dataset.
    '''
    # Usual transform
    # t = transforms.Compose([
    #     transforms.Scale([opt.image_size, opt.image_size]),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    if opt.dataset=='mnist':
        t = transforms.Compose([
            transforms.CenterCrop(opt.image_size),
            transforms.Scale([opt.image_size, opt.image_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    else:
        t = transforms.Compose([
            transforms.CenterCrop(opt.image_size),
            transforms.Scale([opt.image_size, opt.image_size]),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        #imdir = 'train' if train else 'val'
        #dataroot = os.path.join(opt.dataroot, imdir)
        dataroot =opt.dataroot
        dataset = dset.ImageFolder(root=dataroot, transform=t)
    elif opt.dataset == 'lsun':
        dataset = dset.LSUN(root=opt.dataroot,
                            classes=['bedroom_val'],
                            transform=t)
        #dataset = dset.LSUNClass(root=opt.dataroot,transform=t)
    elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root='data/raw/mnist',
                             download=True,
                             train=train,
                             transform=t
                             )
    elif opt.dataset == 'svhn':
        dataset = dset.SVHN(root='data/raw/svhn',
                            download=True,
                            train=train,
                            transform=t)

    else:
        assert False, 'Wrong dataset name.'

    assert len(dataset) > 0, 'No images found, check your paths.'

    # Shuffle and drop last when training
    #tr=transforms.Scale([opt.image_size, opt.image_size])
    #dataset=tr(dataset)
    #dataset.data[dataset.train_labels == 1]
    #dataset.data[torch.Tensor(dataset.targets) == 1]
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=shuffle,
                                             num_workers=int(opt.workers),
                                             pin_memory=True,
                                             drop_last=drop_last)

    return InfiniteDataLoader(dataloader)


class InfiniteDataLoader(object):
    """docstring for InfiniteDataLoader"""

    def __init__(self, dataloader):
        super(InfiniteDataLoader, self).__init__()
        self.dataloader = dataloader
        self.data_iter = None

    def next(self):
        try:
            data = self.data_iter.next()
        except Exception:
            # Reached end of the dataset
            self.data_iter = iter(self.dataloader)
            data = self.data_iter.next()

        return data

    def __len__(self):
        return len(self.dataloader)
def normalize_data(data):
  N, c = data.shape
  min_scale = 1
  for i in range(c):
    vmin = data[:, i].min()
    vmax = data[:, i].max()
    data[:, i] = data[:, i] - vmin
    scale = 1. / (vmax - vmin)
    if scale < min_scale:
      min_scale = scale
  data = data * min_scale
  data = (data - 0.5) / 0.5
  return data
def data_load(opt):
  if opt.data_type == 'ball':
    import pandas as pd
    ball_data = pd.read_csv(opt.data_root, sep='\t', header=1,
                            dtype=np.float32)
    pclouds = ball_data.values
    pclouds = torch.from_numpy(pclouds)
    pclouds = pclouds.unsqueeze(-1).contiguous()
    return pclouds
  elif opt.data_type == 'swiss_roll':
    n_samples = 5000
    noise = 0.05
    X, _ = make_swiss_roll(n_samples, noise)
    # Make it thinner
    X[:, 1] -= 10
    X[:, 1] *= 0.5
    data = X
    data = normalize_data(data)
    data = data.astype(np.float32)
    #pclouds = data.values
    pclouds = torch.from_numpy(data)
    pclouds = pclouds.unsqueeze(-1).contiguous()
    points_dataset = torch.utils.data.TensorDataset(pclouds)
    points_dataset_loader = torch.utils.data.DataLoader(dataset=points_dataset,
                                                       batch_size=5000,
                                                        shuffle=True,
                                                        num_workers=opt.workers)
    #return pclouds
    return InfiniteDataLoader(points_dataset_loader)
  elif opt.data_type == 'hyperbolic_paraboloid':
    n_samples = 5000
    z1 = torch.randn(n_samples)
    z2 = torch.randn(n_samples)
    z3 = (z1*z1-z2*z2)*0.5
    z=torch.cat((z1.unsqueeze(-1),z2.unsqueeze(-1),z3.unsqueeze(-1)),1)
    z = z.unsqueeze(-1).contiguous()
  return z

def weights_init(m):
    '''
    Custom weights initialization called on netG and netE
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def load_G(opt):
    '''
    Loads generator model.
    '''
    m = importlib.import_module('models.' + opt.netG)
    netG = m._netG(opt)
    netG.apply(weights_init)
    netG.train()
    if opt.netG_chp != '':
        netG.load_state_dict(torch.load(opt.netG_chp).state_dict())

    print('Generator\n', netG)
    return netG


def load_E(opt):
    '''
    Loads encoder model.
    '''
    m = importlib.import_module('models.' + opt.netE)
    netE = m._netE(opt)
    netE.apply(weights_init)
    netE.train()
    if opt.netE_chp != '':
        netE.load_state_dict(torch.load(opt.netE_chp).state_dict())

    print('Encoder\n', netE)

    return netE
def load_vae_E(opt):
    '''
    Loads encoder model.
    '''
    m = importlib.import_module('models.' + opt.netE)
    netE = m.vae_netE(opt)
    netE.apply(weights_init)
    netE.train()
    if opt.netE_chp != '':
        netE.load_state_dict(torch.load(opt.netE_chp).state_dict())

    print('Encoder\n', netE)

    return netE
def load_g(opt):
    '''
    Loads generator model.
    '''
    m = importlib.import_module('models.' + opt.netg)
    netg = m._netg(opt)
    netg.apply(weights_init)
    netg.train()
    if opt.netg_chp != '':
        netg.load_state_dict(torch.load(opt.netg_chp).state_dict())

    print('Generator_latent\n', netg)
    return netg

def load_e(opt):
    '''
    Loads generator model.
    '''
    m = importlib.import_module('models.' + opt.nete)
    nete = m._nete(opt)
    nete.apply(weights_init)
    nete.train()
    if opt.nete_chp != '':
        nete.load_state_dict(torch.load(opt.nete_chp).state_dict())

    print('Encoder_latent\n', nete)
    return nete
def load_D(opt):
    '''
    Loads generator model.
    '''
    m = importlib.import_module('models.' + opt.netD)
    netD = m._netD(opt)
    netD.apply(weights_init)
    netD.train()
    if opt.nete_chp != '':
        netD.load_state_dict(torch.load(opt.netD_chp).state_dict())

    print('Discriminator_latent\n', netD)
    return netD
def load_vaegan_d(opt):
    '''
    Loads generator model.
    '''
    m = importlib.import_module('models.' + opt.netd)
    netd = m.vaegan_netd(opt)
    netd.apply(weights_init)
    netd.train()
    if opt.netd_chp != '':
        netd.load_state_dict(torch.load(opt.netd_chp).state_dict())

    print('Discriminator_latent\n', netd)
    return netd

def load_d(opt):
    '''
    Loads generator model.
    '''
    m = importlib.import_module('models.' + opt.netd)
    netd = m._netd(opt)
    netd.apply(weights_init)
    netd.train()
    if opt.netd_chp != '':
        netd.load_state_dict(torch.load(opt.netd_chp).state_dict())

    print('Discriminator_latent\n', netd)
    return netd
def match(x, y, dist):
    '''
    Computes distance between corresponding points points in `x` and `y`
    using distance `dist`.
    '''
    if dist == 'L2':
        return (x - y).pow(2).mean()
    elif dist == 'L1':
        return (x - y).abs().mean()
    elif dist == 'cos':
        x_n = normalize(x)
        y_n = normalize(y)

        return 2 - (x_n).mul(y_n).mean()
    else:
        assert dist == 'none', 'wtf ?'


def populate_x(x, dataloader):
    '''
    Fills input variable `x` with data generated with dataloader
    '''
    real_cpu,_ = dataloader.next()
    #real_cpu=dataloader.next().float().cuda()/255
    #real_cpu=real_cpu.permute(0,3,1,2)
    #real_cpu=torch.nn.functional.interpolate(real_cpu,size=[32, 32],mode='bilinear')
    #x.data.resize_(real_cpu.size()).copy_(real_cpu)
    x.resize_(real_cpu.size()).copy_(real_cpu)

def populate_x_3D(x, dataloader):
    '''
    Fills input variable `x` with data generated with dataloader
    '''
    real_cpu = dataloader.next()[0].squeeze()
    #real_cpu=dataloader.next().float().cuda()/255
    #real_cpu=real_cpu.permute(0,3,1,2)
    #real_cpu=torch.nn.functional.interpolate(real_cpu,size=[32, 32],mode='bilinear')
    #x.data.resize_(real_cpu.size()).copy_(real_cpu)
    x.data.copy_(real_cpu)
def populate_z(z, opt):
    '''
    Fills noise variable `z` with noise U(S^M)
    '''
    z.data.resize_(opt.batch_size, opt.nz, 1, 1)
    z.data.normal_(0, 1)
    if opt.noise == 'sphere':
        normalize_(z.data)
def populate_z2(z, opt):
    '''
    Fills noise variable `z` with noise U(S^M)
    '''
    z.data.resize_(opt.batch_size*4, opt.nz, 1, 1)
    z.data.normal_(0, 1)
    if opt.noise == 'sphere':
        normalize_(z.data)

def normalize_(x, dim=1):
    '''
    Projects points to a sphere inplace.
    '''
    x.div_(x.norm(2, dim=dim).unsqueeze(dim).expand_as(x))


def normalize(x, dim=1):
    '''
    Projects points to a sphere.
    '''
    return x.div(x.norm(2, dim=dim).unsqueeze(dim).expand_as(x))


def var(x, dim=0):
    '''
    Calculates variance.
    '''
    x_zero_meaned = x - x.mean(dim).expand_as(x)
    return x_zero_meaned.pow(2).mean(dim)


class ALICropAndScale(object):
    def __call__(self, img):
        return img.resize((64, 78), Image.ANTIALIAS).crop((0, 7, 64, 64 + 7))

def hinge_loss_discriminator(r_logit, f_logit):
    r_logit_mean = r_logit.mean()
    f_logit_mean = f_logit.mean()

    loss_real = torch.mean(F.relu(1. - r_logit))
    loss_fake = torch.mean(F.relu(1. + f_logit))
    D_loss = loss_real + loss_fake
    return r_logit_mean, f_logit_mean, D_loss

def hinge_loss_generator(r_logit, f_logit):
    r_logit_mean = r_logit.mean()
    f_logit_mean = f_logit.mean()

    loss_real = torch.mean(F.relu(1. + r_logit))
    loss_fake = torch.mean(F.relu(1. - f_logit))
    D_loss = loss_real + loss_fake
    return r_logit_mean, f_logit_mean, D_loss

def hinge_loss_generator2(f_logit):
    f_logit_mean = f_logit.mean()
    G_loss = - f_logit_mean
    return f_logit_mean, G_loss
