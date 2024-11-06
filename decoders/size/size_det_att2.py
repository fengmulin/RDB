from collections import OrderedDict

import torch
import torch.nn as nn
import math
BatchNorm2d = nn.BatchNorm2d

class CAttentionLayer(nn.Module):
    def __init__(self, hidden_dim,  dropout=0.1):
        super().__init__()

        self.conv_k = nn.Conv2d(hidden_dim, hidden_dim,kernel_size=1, stride=1)

        self.hidden_dim = hidden_dim
        self.scale = math.sqrt(self.hidden_dim)

    def forward(self, q, k, v, mask=None):
        b, c, h, w = k.shape
        q = q.flatten(2)
        k = self.conv_k(k).flatten(2).permute(0, 2, 1)
                                                       
        att = torch.matmul(q / self.scale, k)
        if mask is not None:
            att = att.masked_fill(mask == 0, -1e10)
        att = torch.softmax(att, dim=-1)

        return att.view(b,32,1,1)

class Det_size_att2(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256,
                 num_heads=2,
                 part_dim = 16,
                 bias=False, adaptive=False, smooth=False, 
                 *args, **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(Det_size_att2, self).__init__()
        #self.k = k
        #self.k = nn.Parameter(torch.ones(1)*k)
        
        head_dim = int(inner_channels/num_heads)//8
        self.dim = part_dim

        # self-attention heads in Lo-Fi
        self.l_heads = int(num_heads)
        # token dimension in Lo-Fi
        self.l_dim = self.l_heads * head_dim
        self.scale = head_dim ** -0.5
        self.layernum = 4
        self.sr = nn.AvgPool2d(kernel_size=8, stride=8)
        self.l_q = nn.Linear(self.dim, self.dim, bias=True)
        self.l_kv = nn.Linear(self.dim, self.dim * 2, bias=True)
        self.l_proj = nn.Linear(self.l_dim, self.l_dim, bias=True)
        # self.pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)
        # self.att_s = CAttentionLayer(32)
        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      self.layernum, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(
            inner_channels, inner_channels//self.layernum, 3, padding=1, bias=bias)

        self.bin = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      self.layernum, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//self.layernum),
            nn.ReLU(inplace=True),)
        
        self.smooth = nn.Sequential(
            nn.Conv2d(inner_channels//self.layernum, inner_channels //
                      self.layernum, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//self.layernum),
            nn.GELU())
        
        self.binarize = nn.Sequential(
            nn.ConvTranspose2d(inner_channels//self.layernum, inner_channels//self.layernum, 2, 2),
            BatchNorm2d(inner_channels//self.layernum),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//self.layernum, 1, 2, 2),
            nn.Sigmoid())
        # self.spatial_wise.apply(self.weights_init)
        self.smooth.apply(self.weights_init)
        self.bin.apply(self.weights_init)
        self.binarize.apply(self.weights_init)
        self.thr  = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),)
        self.thresh = nn.Sequential(
            nn.Conv2d(inner_channels//4, 1, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear'))
        
        self.thr.apply(self.weights_init)
        self.thresh.apply(self.weights_init)
        
        
        self.adaptive = adaptive

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
    def att(self, x,y):
        B, C, H, W = x.shape
        # print(x.permute(0,2,3,1).shape,self.dim)
        q = self.l_q(x.permute(0,2,3,1))
        q = q.reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)

        # if self.ws > 1:
        y = self.sr(y).reshape(B, C, -1).permute(0, 2, 1)
        kv = self.l_kv(y).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        # kv=self.l_kv(x).reyshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # print(q.shape, k.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        x = self.l_proj(x).permute(0,3,1,2)
        return x
    def forward(self, features, gt=None, masks=None, training=False,):
        # print(size_balance,1111)
        # raise
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = self.up5(in5) + in4  # 1/16
        out3 = self.up4(in4) + in3  # 1/8
        out2 = self.up3(out3) + in2  # 1/4

        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)

        fuse = torch.cat((p5, p4, p3, p2), 1)
        
        seg_fuse = self.bin(fuse)
        gauss_fuse = self.thr(fuse)

        att_temp = self.att(seg_fuse[:, :self.dim, :, :],gauss_fuse[:, :self.dim, :, :])
        # print(seg_fuse[:, :self.dim, :, :].shape, att_temp.shape)
        att_fuse = torch.cat((seg_fuse[:, :self.dim, :, :] + att_temp,
                              seg_fuse[:, self.dim:, :, :]),  dim=1)

        att_fuse = self.smooth(att_fuse)
        binary = self.binarize(att_fuse)
        if self.training:
            result = OrderedDict(binary=binary)
        else:
            return binary
       
        if self.adaptive and self.training:
            size = self.thresh(gauss_fuse)
            result.update(size=size)
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
