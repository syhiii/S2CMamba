import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sota.pannet import ConvBlock
import math


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.kernel_size = 3
        self.stride = 2
        self.padding = 1

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal(self.g.weight)
        nn.init.constant(self.g.bias, 0)
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.kaiming_normal(self.W[0].weight)
            nn.init.constant(self.W[0].bias, 0)
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)

            self.W_pan = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.kaiming_normal(self.W_pan[0].weight)
            nn.init.constant(self.W_pan[0].bias, 0)
            nn.init.constant(self.W_pan[1].weight, 0)
            nn.init.constant(self.W_pan[1].bias, 0)

        else:
            self.W_pan = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                                 kernel_size=1, stride=1, padding=0)
            nn.init.kaiming_normal(self.W_pan.weight)
            nn.init.constant(self.W_pan.bias, 0)

            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.kaiming_normal(self.W.weight)
            nn.init.constant(self.W.bias, 0)

        self.theta = None
        self.phi = None
        self.phi_pan = None

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.phi_pan = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        ## PAN
        self.g_pan = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal(self.g_pan.weight)
        nn.init.constant(self.g_pan.bias, 0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            self.g_pan = nn.Sequential(self.g_pan, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
                self.phi_pan = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))
                self.phi_pan = nn.Sequential(self.phi_pan, max_pool(kernel_size=2))

    def forward(self, x, x_pan):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        output = self._embedded_gaussian(x, x_pan)
        return output

    def _embedded_gaussian(self, x, x_pan):
        batch_size = x.size(0)
        patch_size = x.size(2)

        # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h/scale, w/scale)]->(b, thw/scale^2, 0.5c)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h/scale, w/scale)]->(b, 0.5c, thw/scale^2)
        # f=>(b, thw/scale^2, 0.5c)dot(b, 0.5c, thw/scale^2) = (b, thw/scale^2, thw/scale^2)
        theta_x = self.theta(x)
        theta_x = self.up(theta_x)
        theta_x = theta_x.view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x)
        phi_x = self.up(phi_x)
        phi_x = phi_x.view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        # (b, thw, thw)dot(b, thw, 0.5c) = (b, thw, 0.5c)->(b, 0.5c, t, h, w)->(b, c, t, h, w)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, int(patch_size), int(patch_size))
        W_y = self.W(y)

        # PAN
        x1 = x_pan
        g_pan_x = self.g_pan(x1).view(batch_size, self.inter_channels, -1)
        g_pan_x = g_pan_x.permute(0, 2, 1)

        phi_pan_x = self.phi_pan(x1)
        phi_pan_x = self.up(phi_pan_x)
        phi_pan_x = phi_pan_x.view(batch_size, self.inter_channels, -1)

        f_pan = torch.matmul(theta_x, phi_pan_x)
        f_pan_div_C = F.softmax(f_pan, dim=-1)
        y_pan = torch.matmul(f_pan_div_C, g_pan_x)
        y_pan = y_pan.permute(0, 2, 1).contiguous()
        y_pan = y_pan.view(batch_size, self.inter_channels, int(patch_size), int(patch_size))
        W_pan_y = self.W_pan(y_pan)

        z = torch.cat([W_y, W_pan_y], 1)

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class att_spatial(nn.Module):
    def __init__(self):
        super(att_spatial, self).__init__()
        kernel_size = 7
        block = [
            ConvBlock(2, 32, 3, 1, 1, activation='prelu', norm=None, bias=False),
        ]
        for i in range(6):
            block.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        self.block = nn.Sequential(*block)
        self.spatial = ConvBlock(2, 1, 3, 1, 1, activation='prelu', norm=None, bias=False)

    def forward(self, x):
        x = self.block(x)
        x_compress = torch.cat([torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)], dim=1)
        x_out = self.spatial(x_compress)

        scale = F.sigmoid(x_out)  # broadcasting
        return scale


class Upsampler(torch.nn.Module):
    def __init__(self, scale, n_feat, bn=False, activation='prelu', bias=True):
        super(Upsampler, self).__init__()
        modules = []
        if scale == 3:
            modules.append(ConvBlock(n_feat, 9 * n_feat, 3, 1, 1, bias, activation=None, norm=None))
            modules.append(torch.nn.PixelShuffle(3))
            if bn:
                modules.append(torch.nn.BatchNorm2d(n_feat))
        else:
            for _ in range(int(math.log(scale, 2))):
                modules.append(ConvBlock(n_feat, 4 * n_feat, 3, 1, 1, bias, activation=None, norm=None))
                modules.append(torch.nn.PixelShuffle(2))
                if bn:
                    modules.append(torch.nn.BatchNorm2d(n_feat))

        self.up = torch.nn.Sequential(*modules)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.up(x)
        if self.activation is not None:
            out = self.act(out)
        return out

class ResnetBlock(torch.nn.Module):
    def __init__(self, input_size, kernel_size=3, stride=1, padding=1, bias=True, scale=1, activation='prelu',
                 norm='batch', pad_model=None):
        super().__init__()

        self.norm = norm
        self.pad_model = pad_model
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.scale = scale

        if self.norm == 'batch':
            self.normlayer = torch.nn.BatchNorm2d(input_size)
        elif self.norm == 'instance':
            self.normlayer = torch.nn.InstanceNorm2d(input_size)
        else:
            self.normlayer = None

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        else:
            self.act = None

        if self.pad_model == None:
            self.conv1 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding, bias=bias)
            self.conv2 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding, bias=bias)
            self.pad = None
        elif self.pad_model == 'reflection':
            self.pad = nn.Sequential(nn.ReflectionPad2d(padding))
            self.conv1 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, 0, bias=bias)
            self.conv2 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, 0, bias=bias)

        layers = filter(lambda x: x is not None,
                        [self.pad, self.conv1, self.normlayer, self.act, self.pad, self.conv2, self.normlayer,
                         self.act])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = x
        out = self.layers(x)
        out = out * self.scale
        out = torch.add(out, residual)
        return out


class Net(nn.Module):
    def __init__(self, num_channels):
        super(Net, self).__init__()

        out_channels = num_channels
        n_resblocks = 11

        # pixel domain
        res_block_s1 = [
            ConvBlock(num_channels, 32, 3, 1, 1, activation='prelu', norm=None, bias=False),
        ]
        for i in range(n_resblocks):
            res_block_s1.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        res_block_s1.append(Upsampler(2, 32, activation='prelu'))
        res_block_s1.append(ConvBlock(32, out_channels, 3, 1, 1, activation='prelu', norm=None, bias=False))
        self.res_block_s1 = nn.Sequential(*res_block_s1)

        res_block_s2 = [
            ConvBlock(num_channels, 32, 3, 1, 1, activation='prelu', norm=None, bias=False),
        ]
        for i in range(n_resblocks):
            res_block_s2.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        self.res_block_s2 = nn.Sequential(*res_block_s2)

        pan_res_block_s2 = [
            ConvBlock(1, 32, 3, 1, 1, activation='prelu', norm=None, bias=False),
        ]
        for i in range(n_resblocks):
            pan_res_block_s2.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        self.pan_res_block_s2 = nn.Sequential(*pan_res_block_s2)
        self.nn1 = NONLocalBlock2D(in_channels=32, mode='embedded_gaussian')
        self.funsion1 = ConvBlock(64, out_channels, 1, 1, 0, activation='prelu', norm=None, bias=False)

        res_block_s3 = [
            ConvBlock(num_channels, 32, 3, 1, 1, activation='prelu', norm=None, bias=False),
        ]
        for i in range(n_resblocks):
            res_block_s3.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        res_block_s3.append(Upsampler(2, 32, activation='prelu'))
        res_block_s3.append(ConvBlock(32, out_channels, 3, 1, 1, activation='prelu', norm=None, bias=False))
        self.res_block_s3 = nn.Sequential(*res_block_s3)

        # res_block_s4 = [
        #     ConvBlock(num_channels, 32, 3, 1, 1, activation='prelu', norm=None, bias = False),
        # ]
        # for i in range(n_resblocks):
        #     res_block_s4.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        # self.res_block_s4 = nn.Sequential(*res_block_s4)

        pan_res_block_s4 = [
            ConvBlock(1, 32, 3, 1, 1, activation='prelu', norm=None, bias=False),
        ]
        for i in range(n_resblocks):
            pan_res_block_s4.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        self.pan_res_block_s4 = nn.Sequential(*pan_res_block_s4)
        self.nn2 = NONLocalBlock2D(in_channels=32, mode='embedded_gaussian')
        self.funsion2 = ConvBlock(64, out_channels, 1, 1, 0, activation='prelu', norm=None, bias=False)

        res_block_s4 = [
            ConvBlock(num_channels, 32, 3, 1, 1, activation='prelu', norm=None, bias=False),
        ]
        for i in range(n_resblocks):
            res_block_s4.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        res_block_s4.append(ConvBlock(32, 32, 3, 1, 1, activation='prelu', norm=None, bias=False))
        self.res_block_s4 = nn.Sequential(*res_block_s4)

        self.rm1 = att_spatial()
        self.rm2 = att_spatial()

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, l_ms,  x_pan):
        hp_pan_4 = x_pan - F.interpolate(F.interpolate(x_pan, scale_factor=1 / 4, mode='bicubic'), scale_factor=4,
                                         mode='bicubic')
        lr_pan = F.interpolate(x_pan, scale_factor=1 / 2, mode='bicubic')
        hp_pan_2 = lr_pan - F.interpolate(F.interpolate(lr_pan, scale_factor=1 / 2, mode='bicubic'), scale_factor=2,
                                          mode='bicubic')

        # NN, pixel domain
        s1 = self.res_block_s1(l_ms)
        s1 = s1 + F.interpolate(l_ms, scale_factor=2, mode='bicubic')
        s2 = self.funsion1(self.nn1(self.res_block_s2(s1), self.pan_res_block_s2(lr_pan)))

        # residual modification
        rm_s2_0 = hp_pan_2 + self.rm1(torch.cat([torch.unsqueeze(s2[:, 0, :, :], 1), lr_pan], 1)) * hp_pan_2
        rm_s2_1 = hp_pan_2 + self.rm1(torch.cat([torch.unsqueeze(s2[:, 1, :, :], 1), lr_pan], 1)) * hp_pan_2
        rm_s2_2 = hp_pan_2 + self.rm1(torch.cat([torch.unsqueeze(s2[:, 2, :, :], 1), lr_pan], 1)) * hp_pan_2
        rm_s2_3 = hp_pan_2 + self.rm1(torch.cat([torch.unsqueeze(s2[:, 3, :, :], 1), lr_pan], 1)) * hp_pan_2
        rm_s2_pan = torch.cat([rm_s2_0, rm_s2_1, rm_s2_2, rm_s2_3], 1)

        s2 = s2 + F.interpolate(l_ms, scale_factor=2, mode='bicubic') + rm_s2_pan

        s3 = self.res_block_s3(s2)
        # s4 = self.res_block_s4(torch.cat([s3, x_pan], 1))
        s4_x = self.res_block_s4(s3)
        s4_pan = self.pan_res_block_s4(x_pan)
        x4 = self.nn2(s4_x, s4_pan)
        s4 = self.funsion2(x4)

        # residual modification
        rm_s4_0 = hp_pan_4 + self.rm2(torch.cat([torch.unsqueeze(s4[:, 0, :, :], 1), x_pan], 1)) * hp_pan_4
        rm_s4_1 = hp_pan_4 + self.rm2(torch.cat([torch.unsqueeze(s4[:, 1, :, :], 1), x_pan], 1)) * hp_pan_4
        rm_s4_2 = hp_pan_4 + self.rm2(torch.cat([torch.unsqueeze(s4[:, 2, :, :], 1), x_pan], 1)) * hp_pan_4
        rm_s4_3 = hp_pan_4 + self.rm2(torch.cat([torch.unsqueeze(s4[:, 3, :, :], 1), x_pan], 1)) * hp_pan_4
        rm_s4_pan = torch.cat([rm_s4_0, rm_s4_1, rm_s4_2, rm_s4_3], 1)

        s4 = s4 + rm_s4_pan

        return s4
