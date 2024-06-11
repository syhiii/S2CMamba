
import torch
import torch.nn.functional as F
import torch.nn as nn


if torch.cuda.is_available():
    device = 'cuda:0'
else:
    raise 'CUDA is not available'

def sobel_conv(data, channel):
    conv_op_x = nn.Conv2d(channel, channel, 3, stride=1, padding=1, groups=channel, bias=False)
    conv_op_y = nn.Conv2d(channel, channel, 3, stride=1, padding=1, groups=channel, bias=False)
    sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1).to(device)

    sobel_kernel_y = torch.tensor([[-1,-2,-1],
                                   [ 0, 0, 0],
                                   [ 1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1).to(device)
    conv_op_x.weight.data = sobel_kernel_x
    conv_op_y.weight.data = sobel_kernel_y
    edge_x, edge_y = conv_op_x(data), conv_op_y(data)
    result = 0.5*abs(edge_x) + 0.5*abs(edge_y)
    return result
class FEB(nn.Module):
    def __init__(self, input_size, output_size=32, ker_size=3, stride=1):
        super(FEB, self).__init__()
        self.conv_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=ker_size, stride=stride, padding=0),
            nn.LeakyReLU())

        self.dilated_conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.LeakyReLU())

        self.dilated_conv2 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0, dilation=2),
            nn.LeakyReLU())

        self.dilated_conv3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0, dilation=3),
            nn.LeakyReLU())

        self.dilated_conv4 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0, dilation=4),
            nn.LeakyReLU())

        self.conv_tail = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=output_size, out_channels=output_size, kernel_size=ker_size, stride=stride, padding=0),
            nn.LeakyReLU())
    def forward(self, data):
        conv_head = self.conv_head(data)
        x1, x2, x3, x4 = torch.split(conv_head, 8, dim=1)
        x1 = self.dilated_conv1(x1)
        x2 = self.dilated_conv2(x2)
        x3 = self.dilated_conv3(x3)
        x4 = self.dilated_conv4(x4)
        cat = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.conv_tail(cat)
        output = conv_head + out
        return output

class LDB(nn.Module):
    def __init__(self):
        super(LDB, self).__init__()
    def forward(self, fea_panlow, fea_ms):
        fea_panlow, fea_ms = (fea_panlow/torch.max(fea_panlow)), (fea_ms/torch.max(fea_ms))
        edge_pan = sobel_conv(fea_panlow, 32)
        edge_ms = sobel_conv(fea_ms, 32)
        dis_map = (1-edge_pan)*(1-edge_ms) + edge_pan*edge_ms + 1 - torch.abs(edge_pan - edge_ms)
        return dis_map, edge_pan, edge_ms

class conv_block1(nn.Module):
    def __init__(self, input_size, output_size=32, ker_size=3, stride=1):
        super(conv_block1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=ker_size, stride=stride, padding=0))
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=ker_size, stride=stride, padding=0),
            nn.LeakyReLU())
    def forward(self, data):
        conv1 = self.conv1(data)
        bicubic = F.interpolate(conv1, scale_factor=1, mode='bicubic', align_corners=True)
        out = self.conv2(bicubic)
        return out


class conv_block2(nn.Module):
    def __init__(self, input_size, output_size=64, ker_size=3, stride=1):
        super(conv_block2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=ker_size, stride=stride, padding=0))
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=output_size, out_channels=output_size, kernel_size=ker_size, stride=stride, padding=0),
            nn.LeakyReLU())

    def forward(self, data):
        conv1 = self.conv1(data)
        bicubic = F.interpolate(conv1, size=(48, 48), mode='bicubic', align_corners=True)
        out = self.conv2(bicubic)
        return out


class conv_block3(nn.Module):
    def __init__(self, input_size, output_size=64, ker_size=3, stride=1):
        super(conv_block3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=ker_size, stride=stride, padding=0))
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=output_size, out_channels=output_size, kernel_size=ker_size, stride=stride, padding=0),
            nn.LeakyReLU())

    def forward(self, data):
        conv1 = self.conv1(data)
        bicubic = F.interpolate(conv1, size=(64, 64), mode='bicubic', align_corners=True)
        out = self.conv2(bicubic)
        return out


class reconstruction(nn.Module):
    def __init__(self, input_size, output_size, ker_size=3, stride=1):
        super(reconstruction, self).__init__()
        self.conv = nn.Sequential(
        nn.ReflectionPad2d(1),
        torch.nn.Conv2d(in_channels=input_size, out_channels=input_size, kernel_size=ker_size, stride=stride, padding=0),
        torch.nn.LeakyReLU(),
        nn.ReflectionPad2d(1),
        torch.nn.Conv2d(in_channels=input_size, out_channels=32, kernel_size=ker_size, stride=stride, padding=0),
        torch.nn.LeakyReLU(),
        nn.ReflectionPad2d(1),
        torch.nn.Conv2d(in_channels=32, out_channels=output_size, kernel_size=ker_size, stride=stride, padding=0),
        torch.nn.LeakyReLU())
    def forward(self, data):
        out = self.conv(data)
        return out


class conv_solo(nn.Module):
    def __init__(self, input_size, output_size, ker_size=3, stride=1):
        super(conv_solo, self).__init__()
        self.conv = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=ker_size, stride=stride, padding=0))
    def forward(self, data):
        out = self.conv(data)
        return out


class EGDU(nn.Module):
    def __init__(self):
        super(EGDU, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0))

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0))

        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0))

        self.conv4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0))

        # self.batchnorm = nn.BatchNorm2d()
    def forward(self, fea_pan, dis_map):
        fea_pan = fea_pan/torch.max(fea_pan)
        edge_pan = sobel_conv(fea_pan, 32)
        conv3 = self.conv3(edge_pan)
        dis_map = self.conv1(dis_map)
        bicubic = F.interpolate(dis_map, scale_factor=1, mode='bicubic', align_corners=True)  # 2 32 128 128
        conv2 = self.conv2(bicubic)
        conv4 = self.conv4(conv3 * bicubic)
        out = conv2 + conv3 + conv4
        return out

class EGDU2(nn.Module):
    def __init__(self):
        super(EGDU2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0))

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0))

        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0))

        self.conv4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0))

        # self.batchnorm = nn.BatchNorm2d()
    def forward(self, fea_pan, dis_map):
        fea_pan = fea_pan/torch.max(fea_pan)
        edge_pan = sobel_conv(fea_pan, 32)
        conv3 = self.conv3(edge_pan)
        dis_map = self.conv1(dis_map)
        bicubic = F.interpolate(dis_map, size=(48, 48), mode='bicubic', align_corners=True)
        conv2 = self.conv2(bicubic)
        conv4 = self.conv4(conv3 * bicubic)
        out = conv2 + conv3 + conv4
        return out

class EGDU3(nn.Module):
    def __init__(self):
        super(EGDU3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0))

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0))

        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0))

        self.conv4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0))

        # self.batchnorm = nn.BatchNorm2d()
    def forward(self, fea_pan, dis_map):
        fea_pan = fea_pan/torch.max(fea_pan)
        edge_pan = sobel_conv(fea_pan, 32)
        conv3 = self.conv3(edge_pan)
        dis_map = self.conv1(dis_map)
        bicubic = F.interpolate(dis_map, size=(64, 64), mode='bicubic', align_corners=True)
        conv2 = self.conv2(bicubic)
        conv4 = self.conv4(conv3 * bicubic)
        out = conv2 + conv3 + conv4
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        )

        self.relu = nn.LeakyReLU()
    def forward(self, data):
        out = self.conv(data)
        out = self.relu(out + data)
        return out


class LocalDissimilarity(nn.Module):
    def __init__(self, in_channels=4):
        super(LocalDissimilarity, self).__init__()
        self.FEB_pan = FEB(input_size=1)
        self.FEB_ms = FEB(input_size=in_channels)
        self.FEB_mid1 = FEB(input_size=32)
        self.FEB_mid2 = FEB(input_size=32)

        self.LDB = LDB()
        self.EGDU = EGDU()
        self.EGDU2 = EGDU2()
        self.EGDU3 = EGDU3()

        self.conv_block1 = conv_block1(input_size=32)
        self.conv_block2 = conv_block2(input_size=32)
        self.conv_block3 = conv_block3(input_size=64)

        self.reconstruction = reconstruction(input_size=64, output_size=in_channels)
        # self.pooling = torch.nn.AdaptiveAvgPool2d(2)  # 128 or 32  # 128 or 32
        self.pooling1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((48, 48))
            )  # 128 or 32
        self.pooling2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((16*8, 16*8))
        )
        # 48 16 when train qb
        self.conv_solo1 = conv_solo(input_size=32, output_size=32)
        self.conv_solo2 = conv_solo(input_size=32, output_size=64)
        self.conv_solo3 = conv_solo(input_size=32, output_size=64)


    def forward(self, ms ,pan_low, pan):
        fea_pan = self.FEB_pan(pan)   # 4 32 64 64
        fea_panlow = self.FEB_pan(pan_low) # 4 32 16 16
        fea_ms = self.FEB_ms(ms)
        conv_block1 = self.conv_block1(fea_ms)
        dissimilaritymap1, edge_pan, edge_ms = self.LDB(fea_panlow, fea_ms)
        pool1 = self.pooling1(fea_pan)
        fea_pan1 = self.FEB_mid1(pool1)
        pool2 = self.pooling2(fea_pan1)
        fea_pan2 = self.FEB_mid2(pool2)
        # print(fea_pan2.shape) #  4 32 16 16
        # print(dissimilaritymap1.shape) # 4 32 16 16
        EGDU_out1 = self.EGDU(fea_pan2, dissimilaritymap1)
        Conv1 = self.conv_solo1(EGDU_out1 * fea_pan2)
        conv_block2 = self.conv_block2(conv_block1 + Conv1)
        dissimilaritymap2, edge_pan2, edge_ms2 = self.LDB(fea_pan2, EGDU_out1)
        EGDU_out2 = self.EGDU2(fea_pan1, dissimilaritymap2)
        Conv2 = self.conv_solo2(EGDU_out2 * fea_pan1)
        conv_block3 = self.conv_block3(conv_block2 + Conv2)
        dissimilaritymap3, edge_pan3, edge_ms3 = self.LDB(fea_pan1, EGDU_out2)
        EGDU_out3 = self.EGDU3(fea_pan, dissimilaritymap3)
        Conv3 = self.conv_solo3(EGDU_out3 * fea_pan)
        result = self.reconstruction(Conv3+conv_block3)
        return result, dissimilaritymap1, edge_pan, edge_ms