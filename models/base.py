import torch
import torch.nn as nn
import src.utils as utils


class _netE_Base(nn.Module):
    def __init__(self, opt,main):
        super(_netE_Base, self).__init__()
        self.noise = opt.noise
        self.embedding=opt.embedding
        self.ngpu = opt.ngpu
        self.main = main


    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 0:
            gpu_ids = range(self.ngpu)
            output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        else:
            output = self.main(input)

        output = output.view(output.size(0), -1)
        if self.embedding == 'sphere':
            output = utils.normalize(output)
        #output=utils.normalize_data(output)

        return output

class vae_netE_Base(nn.Module):
    def __init__(self, opt,main):
        super(vae_netE_Base, self).__init__()
        self.noise = opt.noise
        self.embedding=opt.embedding
        self.ngpu = opt.ngpu
        self.main = main
        self.nemb=opt.nemb
        self.enc_log_sigma = nn.Linear(400,  self.nemb)
        self.enc_mu = nn.Linear(400,  self.nemb)

    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 0:
            gpu_ids = range(self.ngpu)
            output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        else:
            output = self.main(input)

        output = output.view(output.size(0), -1)
        mu=self.enc_mu(output)
        logsigma=self.enc_log_sigma(output)
        sigma = torch.exp(0.5 * logsigma)

        eps = torch.Tensor(sigma.shape).normal_().cuda()
        z=eps.mul(sigma).add_(mu)
        if self.embedding == 'sphere':
            z = utils.normalize(z)
        #output=utils.normalize_data(output)

        return z,mu,logsigma
class _netG_Base(nn.Module):
    def __init__(self, opt, main):
        super(_netG_Base, self).__init__()
        self.ngpu = opt.ngpu
        self.main = main

    def forward(self, input):

        # Check input is either (B,C,1,1) or (B,C)
        assert input.nelement() == input.size(0) * input.size(1), 'wtf'
        input = input.view(input.size(0), input.size(1), 1, 1)

        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 0:
            gpu_ids = range(self.ngpu)
            return nn.parallel.data_parallel(self.main, input, gpu_ids)
        else:
            return self.main(input)
class _netG_Base_3D(nn.Module):
    def __init__(self, opt, main):
        super(_netG_Base_3D, self).__init__()
        self.ngpu = opt.ngpu
        self.main = main
        self.noise = opt.noise

    def forward(self, input):

        # Check input is either (B,C,1,1) or (B,C)
        #assert input.nelement() == input.size(0) * input.size(1), 'wtf'
        #input = input.view(input.size(0), input.size(1), 1, 1)

        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 0:
            gpu_ids = range(self.ngpu)
            output=nn.parallel.data_parallel(self.main, input, gpu_ids)
            if self.noise == 'sphere':
                output = utils.normalize(output)
            return output
        else:
            output= self.main(input)
            if self.noise == 'sphere':
                output = utils.normalize(output)
            return output
class _netg_Base(nn.Module):
    def __init__(self, opt, main):
        super(_netg_Base, self).__init__()
        self.ngpu = opt.ngpu
        self.main = main
        self.embedding=opt.embedding
        self.noise = opt.noise

    def forward(self, input):

        # Check input is either (B,C,1,1) or (B,C)
        #assert input.nelement() == input.size(0) * input.size(1), 'wtf'
        #input = input.view(input.size(0), input.size(1), 1, 1)

        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 0:
            gpu_ids = range(self.ngpu)
            output=nn.parallel.data_parallel(self.main, input, gpu_ids)
            if self.embedding == 'sphere':
                output = utils.normalize(output)
           # return output
        else:
            output= self.main(input)
            if self.embedding == 'sphere':
                output = utils.normalize(output)
       # output = utils.normalize_data(output)
        return output
class _nete_Base(nn.Module):
    def __init__(self, opt, main):
        super(_nete_Base, self).__init__()
        self.noise = opt.noise
        self.ngpu = opt.ngpu
        self.main = main

    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 0:
            gpu_ids = range(self.ngpu)
            output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        else:
            output = self.main(input)

        #output = output.view(output.size(0), -1)
        if self.noise == 'sphere':
            output = utils.normalize(output)

        return output

class _netD_Base(nn.Module):
    def __init__(self, opt, main):
        super(_netD_Base, self).__init__()
        self.noise = opt.noise
        self.ngpu = opt.ngpu
        self.main = main

    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 0:
            gpu_ids = range(self.ngpu)
            output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        else:
            output = self.main(input)


        return output

class _netd_Base(nn.Module):
    def __init__(self, opt, main):
        super(_netd_Base, self).__init__()
        self.noise = opt.noise
        self.ngpu = opt.ngpu
        self.main = main

    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 0:
            gpu_ids = range(self.ngpu)
            output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        else:
            output = self.main(input)


        return output.squeeze()

class vaegan_netd_Base(nn.Module):
    def __init__(self, opt, main1,main2):
        super(vaegan_netd_Base, self).__init__()
        self.noise = opt.noise
        self.ngpu = opt.ngpu
        self.main1 = main1
        self.main2 = main2

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 0:
            gpu_ids = range(self.ngpu)
            feature = nn.parallel.data_parallel(self.main1, input, gpu_ids)
            output=nn.parallel.data_parallel(self.main2, feature, gpu_ids)
        else:
            feature = self.main1(input)
            output = self.main2(feature)

        return feature,output.squeeze()
