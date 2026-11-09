import torch
import torch.nn as nn
import torch.nn.functional as F


from models import register
import utils


def gaussian(x, amplitude, mean, stddev):
    return amplitude * torch.exp(-((x - mean) / 4 / stddev)**2)

def wrap_gaussian(x, amplitude, mean, stddev):
    stddev_new = abs(stddev) + 1e-6
    return gaussian(x, amplitude, mean, stddev_new)

class upsample3dhw(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale=scale

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=(1,self.scale,self.scale),align_corners=False,mode='trilinear')
        return x

class upsample3hw_module(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale=scale

    def forward(self, inp):
        x, cd = inp
        x = torch.nn.functional.interpolate(x, scale_factor=(1,self.scale,self.scale),align_corners=False,mode='trilinear')
        return (x, cd)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class RefocusConv(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

        self.depthConv = nn.Sequential(nn.Conv3d(1))

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class attenconv(nn.Module):
    def __init__(self, inChannels, outChannels, negative_slope=0.1, usingbias=False, ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(inChannels, outChannels, 3, 1, 1, bias=usingbias), nn.LeakyReLU(negative_slope, inplace=True),
        )

        self.get_atten = nn.Sequential(
            nn.Conv3d(outChannels, outChannels, 1, 1, 0), nn.ReLU(),
            nn.Conv3d(outChannels, outChannels, 1, 1, 0), nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.conv(x)
        B,C,D,H,W = feat.shape
        atten = F.adaptive_avg_pool2d(feat.reshape(B, -1, H, W), [1,1]).reshape(B,C,D,1,1)
        atten = self.get_atten(atten)
        return feat*atten


class DepthConv(nn.Module):
    def __init__(self, inChannels, outChannels, negative_slope=0.1, usingbias=False, ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1*inChannels, outChannels, (3,3,3), 1, (1,1,1), bias=usingbias), nn.LeakyReLU(negative_slope, inplace=True),
        )

        self.get_atten = nn.Sequential(
            nn.Conv3d(49, 1*inChannels, (7,1,1), 1, (3,0,0), ), nn.Sigmoid(),
            nn.Conv3d(1*inChannels, 1*inChannels, (7,1,1), 1, (3,0,0), ), nn.Sigmoid(),
            nn.Conv3d(1*inChannels, 1*inChannels, (7,1,1), 1, (3,0,0), ), nn.Sigmoid(),
            nn.Conv3d(1*inChannels, 1*inChannels, (7,1,1), 1, (3,0,0), ), nn.Sigmoid(),
        )

        for m in self.conv.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.get_atten.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inp):
        x, cd = inp
        B,C,D,H,W = x.shape
        atten = self.get_atten(cd)
        x = x*atten
        x = self.conv(x)
        return (x, cd)

class SpatialConv(nn.Module):
    def __init__(self, ch, outChannels, midChannels, negative_slope=0.1, usingbias = False):
        super().__init__()
        self.spaconv_s = nn.Sequential(
                    nn.Conv3d(in_channels = ch, out_channels = midChannels, kernel_size = (1,3,3), stride = (1,1,1), padding = (0,1,1), dilation=(1,1,1),bias=usingbias),
                    nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                    nn.Conv3d(in_channels = midChannels, out_channels = outChannels, kernel_size = (1,3,3), stride = (1,1,1), padding = (0,1,1), dilation=(1,1,1),bias=usingbias),
                    nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
                    )
    def forward(self,fm):
        return self.spaconv_s(fm)

class SpatialAngularConv(nn.Module):
    def __init__(self, ch, outChannels, midChannels, num_views, negative_slope=0.1, usingbias = False):
        super().__init__()
        self.spaAngConv = nn.Sequential(
            nn.Conv3d(ch*num_views, midChannels, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=usingbias),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv3d(midChannels, outChannels * num_views, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=usingbias),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )

    def forward(self,fm):
        b, c, n, h, w = fm.shape
        a_in = fm.view(b,c*n,1,h,w)
        out = self.spaAngConv(a_in).view(b,-1,n,h,w)
        return out

class AngularConv(nn.Module):
    def __init__(self, ch, num_views, AngChannel, negative_slope=0.1, usingbias = False):
        super().__init__()
        self.angconv = nn.Sequential(
            nn.Conv3d(ch*num_views, ch, kernel_size=1, stride=1, padding=0, bias=usingbias),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv3d(ch, AngChannel * num_views, kernel_size=1, stride=1, padding=0, bias=usingbias),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )

    def forward(self,fm):
        b, c, n, h, w = fm.shape
        a_in = fm.view(b,c*n,1,h,w)
        out = self.angconv(a_in).view(b,-1,n,h,w)
        return out

class LFConvBlk(nn.Module):
    def __init__(self, ch, outChannels, num_views, negative_slope=0.1, usingbias = False):
        super().__init__()

        S_ch, A_ch, SA_ch = ch , ch//4, ch//4
        self.inConv = nn.Sequential(
            nn.Conv3d(in_channels = ch, out_channels = ch, kernel_size = (1,3,3), stride = (1,1,1), padding = (0,1,1), dilation=(1,1,1),bias=usingbias),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv3d(in_channels = ch, out_channels = SA_ch, kernel_size = (1,3,3), stride = (1,1,1), padding = (0,1,1), dilation=(1,1,1),bias=usingbias),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )
        self.spaAngConv  = SpatialAngularConv(SA_ch, outChannels=SA_ch, midChannels=ch, num_views=num_views,  negative_slope=negative_slope, usingbias = usingbias)
        self.outConv  = nn.Sequential(
            nn.Conv3d(in_channels = SA_ch, out_channels = outChannels, kernel_size = (1,3,3), stride = (1,1,1), padding = (0,1,1), dilation=(1,1,1),bias=usingbias),
        )

    def forward(self,x):
        b, c, n, h, w = x.shape

        out = self.inConv(x)
        out = self.spaAngConv(out)
        out = self.outConv(out) + x

        return out

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x2 = torch.relu(x2)
        x2 = (x2 / ( x2.mean(dim=1,keepdim=True) + 1e-6) )
        return x1*x2
class NAFBlk(nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.1, usingbias = False):
        super().__init__()

        self.conv1  = nn.Conv3d(in_channels = in_channels, out_channels = 2*in_channels, kernel_size = (1,1,1), stride = 1, padding = (0,0,0), dilation=1,bias=usingbias)
        self.conv2  = nn.Conv3d(in_channels = 2*in_channels, out_channels = 2*in_channels, kernel_size = (3,3,3), stride = 1, padding = (1,1,1), dilation=1, groups=2*in_channels, bias=usingbias)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=usingbias),
        )
        self.conv3  = nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = (3,3,3), stride = 1, padding = (1,1,1), dilation=1, bias=usingbias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x*self.sca(x)
        x = self.conv3(x)

        return x


class Residual2d(nn.Module):
    def __init__(self, ch, negative_slope=0.1, usingbias = False):
        super().__init__()

        self.fuse = nn.Sequential(
                nn.Conv3d(in_channels = ch, out_channels = ch, kernel_size = (1,3,3), stride = 1, padding = (0,1,1), dilation=1,bias=usingbias),
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                nn.Conv3d(ch, ch, kernel_size = (1,3,3), stride = 1, padding = (0,1,1),dilation=1,bias=usingbias))

    def forward(self,x):
        out = self.fuse(x) + x

        return out


def lf_refocus_stack(feat, shift):
    B,C,A,H,W = feat.shape
    D = shift.shape[-2]
    grid = utils.make_coord([i for i in [H, W]], flatten=False, device=feat.device).flip(-1).unsqueeze(0).to(feat.device)
    input = feat.permute(2, 0, 1, 3, 4).reshape(A, B*C, H, W)
    new_grid = grid + shift[:, 0, :].flip(-1).unsqueeze(1).unsqueeze(1) * 2 / torch.tensor([feat.shape[-1], feat.shape[-2]], device=feat.device)
    ret = ( F.grid_sample(input, new_grid, mode='bilinear', align_corners=False).reshape(A,B,C,1,H,W)  ).mean(dim=0,keepdim=False)


    for dd in range(1, D):
        new_grid = grid + shift[:, dd, :].flip(-1).unsqueeze(1).unsqueeze(1) * 2 / torch.tensor([feat.shape[-1], feat.shape[-2]], device=feat.device)
        ret = torch.cat( (ret, ( F.grid_sample(input, new_grid, mode='bilinear', align_corners=False).reshape(A,B,C,1,H,W) ).mean(dim=0,keepdim=False) ),  dim=2)


    return ret

def lf_weightedRefocusStack(feat, shift, weight):
    B,C,A,H,W = feat.shape
    D = shift.shape[-2]
    grid = utils.make_coord([i for i in [H, W]], flatten=False, device=feat.device).flip(-1).unsqueeze(0)
    input = feat.permute(2, 0, 1, 3, 4).reshape(A, B*C, H, W)
    new_grid = grid + shift[:, 0, :].flip(-1).unsqueeze(1).unsqueeze(1) * 2 / torch.tensor([feat.shape[-1], feat.shape[-2]], device=feat.device)
    ret = ( F.grid_sample(input, new_grid, mode='bilinear', align_corners=False).reshape(A,B,C,1,H,W) * weight[:,:,0,:,:].reshape(A,1,1,1,1,1) ).mean(dim=0,keepdim=False)
    for dd in range(1, D):
        new_grid = grid + shift[:, dd, :].flip(-1).unsqueeze(1).unsqueeze(1) * 2 / torch.tensor([feat.shape[-1], feat.shape[-2]], device=feat.device)

        ret = torch.cat( (ret, ( F.grid_sample(input, new_grid, mode='bilinear', align_corners=False).reshape(A,B,C,1,H,W) * weight[:,:,dd,:,:].reshape(A,1,1,1,1,1) ).mean(dim=0,keepdim=False) ),  dim=2)

    return ret


@register('serenetfrv1')
class SeReNetFeatureRefocusv1(nn.Module):
    def __init__(self, inChannels, negative_slope=0.1, usingbias=False, reset_param=False):
        super().__init__()


        self.init_feature = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
            LFConvBlk(32, 32, num_views=inChannels, negative_slope=negative_slope, usingbias = usingbias),
            LFConvBlk(32, 32, num_views=inChannels, negative_slope=negative_slope, usingbias = usingbias),
            LFConvBlk(32, 32, num_views=inChannels, negative_slope=negative_slope, usingbias = usingbias),
        )


        self.fusion =  nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(3,3,3),stride=1,padding=(1,1,1), bias=usingbias ),nn.LeakyReLU(negative_slope, inplace=True),

            nn.Conv3d(32, 16, kernel_size=(3,3,3),stride=1,padding=(1,1,1), bias=usingbias ),nn.LeakyReLU(negative_slope, inplace=True),
            upsample3dhw(2),
            nn.Conv3d(16,16,kernel_size=(3,3,3),stride=1,padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),

            nn.Conv3d(16,8,kernel_size=(3,3,3),stride=1,padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),
            upsample3dhw(2),
            nn.Conv3d(8,8,kernel_size=(3,3,3),stride=1, padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),

            nn.Conv3d(8,4,kernel_size=(3,3,3),stride=1, padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv3d(4,1,kernel_size=(3,3,3),stride=1, padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),
        )


        self.register_buffer('depth_encoding',torch.linspace(-50,50,101, dtype=torch.float32).reshape(1, 1, 101, 1, 1) )


    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal(0, 0.01)
                m.bias.data.zero_()


    def forward(self, inp, shift, scale, psfsum):

        buffer = self.init_feature(inp.unsqueeze(0))

        psfsum_depthnorm = psfsum / (psfsum).mean(dim=(2,), keepdim = True)
        psfsum_angledepthnorm = psfsum_depthnorm / psfsum_depthnorm.sum(dim = 1, keepdim = True)
        refocus_weight = psfsum_depthnorm * psfsum_angledepthnorm
        buffer = lf_refocus_stack(buffer, shift)

        buffer = self.fusion(buffer)

        buffer = buffer.squeeze(1)

        if len(buffer.shape) == 4:

            ret = F.interpolate(buffer, size = (round((inp.shape[-2])*scale), round((inp.shape[-1])*scale)),mode='bilinear',align_corners=False).unsqueeze(1)
        else:
            ret = F.interpolate(buffer, size=(inp.shape[-3],round((inp.shape[-2])*scale), round((inp.shape[-1])*scale)) ,mode='trilinear',align_corners=False)

        if self.training:
            return ret, buffer
        else:
            return ret

