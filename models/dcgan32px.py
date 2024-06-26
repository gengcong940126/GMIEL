import torch.nn as nn
from .base import _netE_Base, _netG_Base, _netg_Base, _nete_Base,_netD_Base, _netd_Base,vae_netE_Base
# ------------------------
#         E
# ------------------------


def _netE(opt):
    ndf = opt.ndf
    nc = opt.nc
    nz = opt.nz
    nemb=opt.nemb
    if opt.BN:
        main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, nemb, 4, 2, 1, bias=True),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(nemb),
        )
    else:
        main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, nemb, 4, 2, 1, bias=True),
            nn.AvgPool2d(2),)


    return _netE_Base(opt,main)

# ------------------------
#         D
# ------------------------
def _netD(opt):
    ndf = opt.ndf
    nc = opt.nc
    nz = opt.nz
    nemb=opt.nemb



    main = nn.Sequential(
        nn.utils.spectral_norm(nn.Linear(nemb, 400)),
        #nn.BatchNorm1d(400),
        nn.ReLU(),
        nn.utils.spectral_norm(nn.Linear(400, 400)),
        # nn.BatchNorm1d(400),
        nn.ReLU(),
        nn.utils.spectral_norm(nn.Linear(400, 400)),
        # nn.BatchNorm1d(400),
        nn.ReLU(),
        nn.utils.spectral_norm(nn.Linear(400, 64)),
        #nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

    return _netD_Base(opt, main)

# ------------------------
#         d
# ------------------------
def _netd(opt):
    ndf = opt.ndf
    nc = opt.nc
    nz = opt.nz

    main = nn.Sequential(
        # input is (nc) x 32 x 32
        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=True),
        nn.AvgPool2d(2),
    )

    return _netd_Base(opt, main)
# ------------------------
#         G
# ------------------------


def _netG(opt):
    ngf = opt.ngf
    nc = opt.nc
    nz = opt.nz
    nemb=opt.nemb

    main = nn.Sequential(
        # input is Z, going into a convolution
        nn.ConvTranspose2d(nemb, ngf * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(ngf * 8),
        nn.ReLU(True),
        # state size. (ngf*8) x 4 x 4
        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True),
        # state size. (ngf*4) x 8 x 8
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),
        # state size. (ngf*2) x 16 x 16
        nn.ConvTranspose2d(ngf * 2, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),

        nn.Conv2d(ngf * 2, nc, 1, bias=True),
        nn.Tanh()
    #nn.Sigmoid()
    )

    return _netG_Base(opt, main)

# ------------------------
#         g
# ------------------------

def _netg(opt):
    ngf = opt.ngf
    nc = opt.nc
    nz = opt.nz
    nemb=opt.nemb
    if opt.BN:
        main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(nz, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, nemb),
            nn.BatchNorm1d(nemb),
            #nn.Sigmoid()
        #nn.Sigmoid()
        )
    else:
        main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(nz, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, nemb),
        )
    return _netg_Base(opt, main)


def _netg2(opt):
    ngf = opt.ngf
    nc = opt.nc
    nz = opt.nz
    nemb=opt.nemb
    main = nn.Sequential(
        # input is Z, going into a convolution
        nn.Linear(nz, 400),
        nn.BatchNorm1d(400),
        # nn.ReLU(),
        # nn.Linear(400, 400),
        # nn.BatchNorm1d(400),
        # nn.ReLU(),
        # nn.Linear(400, 400),
        # nn.BatchNorm1d(400),
        # nn.ReLU(),
        # nn.Linear(400, 400),
        # nn.BatchNorm1d(400),
        nn.ReLU(),
        nn.Linear(400, nemb),
        #nn.Sigmoid()
    #nn.Sigmoid()
    )
    return _netg_Base(opt, main)
# ------------------------
#         e
# ------------------------

def _nete(opt):
    ndf = opt.ndf
    nc = opt.nc
    nz = opt.nz
    nemb=opt.nemb
    main = nn.Sequential(
        # input is (nc) x 32 x 32
        nn.Sequential(nn.Linear(nemb, 400),
                      nn.BatchNorm1d(400),
                      nn.ReLU(),
                      nn.Linear(400, nz)
                      )

    )

    return _nete_Base(opt, main)

def vae_netE(opt):
    ndf = opt.ndf
    nc = opt.nc
    nz = opt.nz
    nemb = opt.nemb
    main = nn.Sequential(
        # state size. (ndf) x 32 x 32
        nn.Conv2d(nc, ndf , 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf ),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 8 x 8
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 4 x 4
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*8) x 4 x 4
        nn.Conv2d(ndf * 4,  400 , 4, 1, 0, bias=True),
    )

    return vae_netE_Base(opt, main)
