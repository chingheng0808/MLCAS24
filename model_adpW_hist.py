import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self,
        conv,
        n_feat,
        kernel_size,
        reduction,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * (1 + self.use_affine_level)),
        )

    def forward(self, x, text_embed):
        text_embed = text_embed.permute(1, 0, 2).float()
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.MLP(text_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            # print(gamma.shape, beta.shape)
            x = (1 + gamma) * x + beta
        return x


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks
    ):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv,
                n_feat,
                kernel_size,
                reduction,
                bias=True,
                bn=False,
                act=nn.ReLU(True),
                res_scale=1,
            )
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)
        self.prompt_guidance = FeatureWiseAffine(in_channels=512, out_channels=n_feat)

    def forward(self, x, prompt):
        x = self.prompt_guidance(x, prompt) + x
        res = self.body(x)
        res += x
        return res

class SpectralWiseAttention(nn.Module):
    def __init__(self, in_dim, gamma=1.0, reduction=8):
        super(SpectralWiseAttention, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.conv_qk = nn.Conv2d(in_dim, 2 * in_dim // self.reduction, kernel_size=1, padding=0, stride=1)
        self.conv_v = nn.Conv2d(in_dim, in_dim, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        b, c, h, w = x.shape

        q, k = self.conv_qk(x).chunk(2, dim=1)
        v = self.conv_v(x)
        q = q.view(b, -1, h * w).permute(0, 2, 1)
        k = k.view(b, -1, h * w)
        v = v.view(b, -1, h * w)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = torch.bmm(q, k)  # b, hw, hw,
        attn = attn.softmax(dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, h, w)
        return out * self.gamma + x

class Adapter(nn.Module):
    def __init__(self, in_dim, act=nn.ReLU()):
        super(Adapter, self).__init__()
        self.linear_dw = nn.Linear(in_dim, in_dim//8)
        self.act = act
        self.linear_up = nn.Linear(in_dim//8, in_dim)

    def forward(self, x):
        res = x
        x = self.linear_dw(x)
        x = self.act(x)
        x = self.act(self.linear_up(x) + res)
        return x
    
class SimpleAttention(nn.Module):
    def __init__(self, in_dim):
        super(SimpleAttention, self).__init__()

        self.q = nn.Linear(in_dim, in_dim, bias=False)
        self.k = nn.Linear(in_dim, in_dim, bias=False)
        self.v = nn.Linear(in_dim, in_dim, bias=False)

        self.scale = 1.0 / (in_dim ** 0.5)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attention_scores = torch.matmul(q.unsqueeze(1), k.unsqueeze(1).transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v.unsqueeze(1))
        attention_output = attention_output.squeeze(1)
        return attention_output
    
class model_LSTM(nn.Module):
    def __init__(self, input_size=12*32, hidden_size=128, num_layers=5, bias=True):
        super(model_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first=True)
        self.fc = nn.Linear(128, 64)

    def forward(self, seq):
        # print(seq.shape)
        seq = seq.view(seq.shape[0], seq.shape[1], -1)
        y, _ = self.lstm(seq) # y: (b, tp==6, 128)
        y = y[:,-1,:] # get last layer's hidden state

        y = self.fc(y)

        return y.squeeze(1) # b, 64

class model_Regression_add(nn.Module):
    def __init__(self, in_dim, n_feats=32, add_hidden=64, n_resblocks=2, conv=default_conv, img_size=(20, 10), device='cuda'):
        super(model_Regression_add, self).__init__()
        
        self.device = device
        reduction = 16
        kernel_size = 3
        act = nn.GELU()
        self.n_feats = n_feats
        self.n_resblocks = n_resblocks
        self.img_saize = img_size
        self.add_hidden = add_hidden

        # define head module
        modules_head = [conv(in_dim, n_feats, kernel_size)]

        self.add_block = nn.Sequential( 
            nn.Linear(3+in_dim, self.add_hidden*2, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.add_hidden*2, self.add_hidden, bias=True),
        )
        # self.add_adp = Adapter(self.add_hidden*2)

        self.head = nn.Sequential(*modules_head) # b, n_feat, 20, 10

        self.rg1 = ResidualGroup(
            conv, n_feats, kernel_size, reduction, act, 1, n_resblocks
        )
        self.s_att1 = SpectralWiseAttention(n_feats, gamma=1.0, reduction=reduction)
        self.conv_dwn1 = self.depwiseSepConv(n_feats, n_feats*2**1, 3) # b, n_feat*2, 10, 5
        self.rg2 = ResidualGroup(
            conv, n_feats*2**1, kernel_size, reduction, act, 1, n_resblocks
        )
        self.s_att2 = SpectralWiseAttention(n_feats*2**1, gamma=1.0, reduction=reduction)
        self.conv_dwn2 = self.depwiseSepConv(n_feats*2**1, n_feats*2**2, 3) # b, n_feat*2, 5, 3

        self.conv_df = nn.Conv2d(n_feats*2**2, n_feats, 1, 1, 0)
        
        self.spec_att_d = SpectralWiseAttention(n_feats)

        self.lrelu = nn.LeakyReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.mlp1 = Mlp(
            n_feats*5*3 + add_hidden + 64 + add_hidden,
            n_feats*2,
            1,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.add_weight1 = torch.nn.Parameter(torch.ones(1))
        self.add_weight2 = torch.nn.Parameter(torch.ones(1))
        
        self.adapter = Adapter(512)
        self.param_lin = nn.Sequential(nn.Linear(6, 24), 
                                        nn.Sigmoid(),
                                        nn.Linear(24, 6),
                                        )
        self.params = torch.nn.Parameter(torch.ones(1, 6))

        self.lstm = model_LSTM()
        self.simple_att1 = SimpleAttention(n_feats*5*3 + add_hidden)
        self.simple_att2 = SimpleAttention(64 + self.add_hidden)

    def forward(self, x, prompt, ai, hist):
        non_zeros = (x.view(x.shape[0], x.shape[1], -1).mean(dim=-1) != 0).float()
        # print(x.shape)
        params = torch.nn.functional.normalize(self.param_lin(self.params.repeat(x.shape[0], 1)), p=1, eps=0.1) + self.params.repeat(x.shape[0], 1)
        x = torch.sum(x, dim=1)
        x = x * (1 / torch.sum(params * non_zeros, dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))

        stat = torch.mean(x.view(x.shape[0], x.shape[1], -1), dim=2)
        prompt = self.adapter(prompt)
        y = self.lstm(hist)

        x = self.head(x)
        con1 = x
        x = self.rg1(x, prompt)
        x = self.s_att1(x) + con1
        x = self.conv_dwn1(x)
        con2 = x
        x = self.rg2(x, prompt)
        x = self.s_att2(x) + con2
        x = self.conv_dwn2(x)

        x = self.lrelu(self.conv_df(x))
        x = self.spec_att_d(x)

        ai = self.add_block(torch.concat((ai[:, [0,1,3]], stat), dim=1))
        
        x = torch.concat((self.flatten(x), self.add_weight1*ai), 1)
        x = self.simple_att1(x)

        y = torch.concat((y, self.add_weight2*ai), 1)
        y = self.simple_att2(y)

        x = self.mlp1(torch.concat((x, y), 1))

        # print(x)
        return x

    def depwiseSepConv(self, in_dim, out_dim, ker_sz):
        depwiseConv = nn.Conv2d(
            in_dim, in_dim, ker_sz, 2, ker_sz // 2, groups=in_dim // 8
        )
        ptwiseConv = nn.Conv2d(in_dim, out_dim, 1, 1, 0)
        return nn.Sequential(depwiseConv, ptwiseConv)