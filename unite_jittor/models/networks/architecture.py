from email.policy import default
import jittor
import jittor.nn as nn
from jittor import init
# jittor.nn == torch.nn.functional as F
# torchvision.models.vgg19 == jittor.models.vgg19
from models.networks.utils.spectral_norm import spectral_norm as spectral_norm
from models.networks.normalization import SPADE, equal_lr, SEACE
import os
# print(os.getcwd())
from jittor import models

# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SEACEResnetBlock(jittor.Module):
    def __init__(self, fin, fout, opt, use_se=False, dilation=1, feat_nc=None, atten=False, conf_map=None):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.opt = opt
        # self.pad_type = 'nozero'
        # self.use_se = use_se

        # create conv layers
        # if self.pad_type != 'zero':
        self.pad = nn.ReflectionPad2d(dilation)
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=0, dilation=dilation)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=0, dilation=dilation)
        # else:
        #     self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
        #     self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=dilation, dilation=dilation)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            if opt.eqlr_sn:
                self.conv_0 = equal_lr(self.conv_0)
                self.conv_1 = equal_lr(self.conv_1)
                if self.learned_shortcut:
                    self.conv_s = equal_lr(self.conv_s)
            else:
                self.conv_0 = spectral_norm(self.conv_0)
                self.conv_1 = spectral_norm(self.conv_1)
                if self.learned_shortcut:
                    self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        seace_config_str = opt.norm_G.replace('spectral', '')
        ic = opt.semantic_nc
        #print(seace_config_str) # spadesyncbatch3x3
        self.norm_0 = SEACE(seace_config_str, fin, ic, PONO=opt.PONO, use_apex=opt.apex, feat_nc=feat_nc, atten=atten)
        self.norm_1 = SEACE(seace_config_str, fmiddle, ic, PONO=opt.PONO, use_apex=opt.apex, feat_nc=feat_nc, atten=atten)
        if self.learned_shortcut:
            self.norm_s = SEACE(seace_config_str, fin, ic, PONO=opt.PONO, use_apex=opt.apex, feat_nc=feat_nc, atten=atten)
        if use_se:
            self.se_layar = SELayer(fout)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def execute(self, x, seg1, feat, atten_map, conf_map):
        x_s = self.shortcut(x, seg1, feat, atten_map, conf_map)
        dx = self.conv_0(self.pad(self.actvn(self.norm_0(x, seg1, feat, atten_map, conf_map))))
        dx = self.conv_1(self.pad(self.actvn(self.norm_1(dx, seg1, feat, atten_map, conf_map))))
        out = x_s + dx
        return out

    def shortcut(self, x, seg1, feat, atten_map, conf_map):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg1, feat, atten_map, conf_map))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return jittor.nn.leaky_relu(x, 2e-1)


class Ada_SPADEResnetBlock(jittor.Module):
    def __init__(self, fin, fout, opt, use_se=False, dilation=1):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.opt = opt
        self.pad_type = 'nozero'
        self.use_se = use_se

        # create conv layers
        if self.pad_type != 'zero':
            self.pad = nn.ReflectionPad2d(dilation)
            self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=0, dilation=dilation)
            self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=0, dilation=dilation)
        else:
            self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
            self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=dilation, dilation=dilation)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            if opt.eqlr_sn:
                self.conv_0 = equal_lr(self.conv_0)
                self.conv_1 = equal_lr(self.conv_1)
                if self.learned_shortcut:
                    self.conv_s = equal_lr(self.conv_s)
            else:
                self.conv_0 = spectral_norm(self.conv_0)
                self.conv_1 = spectral_norm(self.conv_1)
                if self.learned_shortcut:
                    self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')

        if 'spade_ic' in opt:
            ic = opt.spade_ic
        else:
            ic = 0 + (3 if 'warp' in opt.CBN_intype else 0) + (opt.semantic_nc if 'mask' in opt.CBN_intype else 0)

        self.norm_0 = SPADE(spade_config_str, fin, ic, PONO=opt.PONO, use_apex=opt.apex)
        self.norm_1 = SPADE(spade_config_str, fmiddle, ic, PONO=opt.PONO, use_apex=opt.apex)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, ic, PONO=opt.PONO, use_apex=opt.apex)

        if use_se:
            self.se_layar = SELayer(fout)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def execute(self, x, seg1):
        x_s = self.shortcut(x, seg1)
        if self.pad_type != 'zero':
            dx = self.conv_0(self.pad(self.actvn(self.norm_0(x, seg1))))
            dx = self.conv_1(self.pad(self.actvn(self.norm_1(dx, seg1))))
            if self.use_se:
                dx = self.se_layar(dx)
        else:
            dx = self.conv_0(self.actvn(self.norm_0(x, seg1)))
            dx = self.conv_1(self.actvn(self.norm_1(dx, seg1)))
            if self.use_se:
                dx = self.se_layar(dx)

        out = x_s + dx

        return out

    def shortcut(self, x, seg1):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg1))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return jittor.nn.leaky_relu(x, 2e-1)


class Attention(jittor.Module):
    def __init__(self, ch, use_sn):
        super(Attention, self).__init__()
        # Channel multiplier
        self.ch = ch
        self.theta = nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = nn.Conv2d(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = nn.Conv2d(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        if use_sn:
            self.theta = spectral_norm(self.theta)
            self.phi = spectral_norm(self.phi)
            self.g = spectral_norm(self.g)
            self.o = spectral_norm(self.o)
        # Learnable gain parameter
        self.gamma = jittor.Var(0.)

    def execute(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = jittor.nn.max_pool2d(self.phi(x), [2,2])
        g = jittor.nn.max_pool2d(self.g(x), [2,2])    
        # Perform reshapes
        theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = jittor.nn.softmax(jittor.nn.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(jittor.nn.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x


# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
class ResnetBlock(jittor.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size))
        )

    def execute(self, x):
        y = self.conv_block(x)
        out = x + y
        return out


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(jittor.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = jittor.nn.Sequential()
        self.slice2 = jittor.nn.Sequential()
        self.slice3 = jittor.nn.Sequential()
        self.slice4 = jittor.nn.Sequential()
        self.slice5 = jittor.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])  #r11
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])  #r21
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])  #r31
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])  #r41
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])  #r51
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def execute(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def test():
    model = jittor.nn.Sequential(
        spectral_norm(nn.Conv2d(3, 64, 2, 1, padding="same"))
    )
    print(model)

    ########### test opt ############
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument("--norm_G", type=str, default="spectralspadeinstance3x3")
    parse.add_argument("--eqlr_sn", action='store_true')
    parse.add_argument("--semantic_nc", type=int, default=1 + 29, choices=[0, 1])
    parse.add_argument("--apex", action='store_true')
    parse.add_argument("--PONO", action='store_true')
    
    opt = parse.parse_args()

    print(opt.norm_G)
    G_test = SEACEResnetBlock(16 * 64, 16 * 64, opt, feat_nc=512, atten=True)
    print(G_test)

if __name__ == '__main__':
    test()
    pass
    