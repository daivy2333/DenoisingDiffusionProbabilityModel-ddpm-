
   
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class Swish(nn.Module):  # 定义一个名为Swish的类，继承自nn.Module，用于实现Swish激活函数
    def forward(self, x):  # 定义前向传播函数，接受输入x
        return x * torch.sigmoid(x)  # 返回输入x与其sigmoid函数值的乘积，即Swish激活函数的计算公式


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        # 确保d_model是偶数
        assert d_model % 2 == 0
        super().__init__()
        # 创建一个从0到d_model-1，步长为2的序列，用于计算位置编码
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        # 计算e的负指数，用于缩放
        emb = torch.exp(-emb)
        # 创建一个从0到T-1的序列，表示时间步
        pos = torch.arange(T).float()
        # 计算位置编码
        emb = pos[:, None] * emb[None, :]
        # 确保emb的形状为[T, d_model // 2]
        assert list(emb.shape) == [T, d_model // 2]
        # 使用正弦和余弦函数生成位置编码
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        # 确保emb的形状为[T, d_model // 2, 2]
        assert list(emb.shape) == [T, d_model // 2, 2]
        # 将emb的形状调整为[T, d_model]
        emb = emb.view(T, d_model)

        # 定义时间嵌入的网络结构
        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),  # 使用预训练的位置编码
            nn.Linear(d_model, dim),  # 线性变换
            Swish(),  # 激活函数
            nn.Linear(dim, dim),  # 再次线性变换
        )
        self.initialize()  # 初始化网络参数

    def initialize(self):
        # 遍历所有子模块，初始化线性层的权重和偏置
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)  # 使用Xavier均匀分布初始化权重
                init.zeros_(module.bias)  # 将偏置初始化为0

    def forward(self, t):
        # 前向传播，获取时间嵌入
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    # 定义一个下采样类，继承自nn.Module
    def __init__(self, in_ch):
        # 构造函数，初始化下采样层
        super().__init__()
        # 调用父类的构造函数
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        # 定义一个二维卷积层，输入通道数和输出通道数均为in_ch，卷积核大小为3x3，步长为2，填充为1
        self.initialize()

        # 调用初始化函数
    def initialize(self):
        # 初始化函数，用于初始化卷积层的权重和偏置
        init.xavier_uniform_(self.main.weight)
        # 使用Xavier均匀分布初始化卷积层的权重
        init.zeros_(self.main.bias)

        # 将卷积层的偏置初始化为0
    def forward(self, x, temb):
        # 前向传播函数，输入为x和temb
        x = self.main(x)
        # 对输入x进行卷积操作
        return x


class UpSample(nn.Module):
    # 定义一个上采样类，继承自nn.Module
    def __init__(self, in_ch):

        # 构造函数，初始化上采样层
        super().__init__()  # 调用父类的构造函数
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)

        # 定义一个二维卷积层，输入通道数和输出通道数均为in_ch，卷积核大小为3x3，步长为1，填充为1
        self.initialize()  # 调用初始化函数

    def initialize(self):
        # 初始化函数，用于初始化卷积层的权重和偏置
        init.xavier_uniform_(self.main.weight)
        # 使用Xavier均匀分布初始化卷积层的权重
        init.zeros_(self.main.bias)

        # 将卷积层的偏置初始化为0
    def forward(self, x, temb):
        # 前向传播函数，输入为x和temb（虽然temb在当前代码中未使用）
        _, _, H, W = x.shape
        # 获取输入张量x的形状，_表示忽略的维度，H和W分别表示高度和宽度
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        # 使用最近邻插值法将输入张量x的尺寸放大两倍
        x = self.main(x)
        # 将上采样后的张量通过卷积层进行处理
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        # 初始化父类
        super().__init__()
        # 创建GroupNorm层，用于规范化输入特征
        self.group_norm = nn.GroupNorm(32, in_ch)
        # 创建卷积层，用于生成查询矩阵Q
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        # 创建卷积层，用于生成键矩阵K
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        # 创建卷积层，用于生成值矩阵V
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        # 创建卷积层，用于生成最终的输出
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        # 初始化权重
        self.initialize()

    def initialize(self):
        # 对卷积层权重进行Xavier均匀初始化，偏置初始化为0
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        # 对最后一个卷积层的权重进行Xavier均匀初始化，但增益设置为1e-5
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        # 获取输入的形状
        B, C, H, W = x.shape
        # 对输入进行GroupNorm规范化
        h = self.group_norm(x)
        # 生成查询矩阵Q
        q = self.proj_q(h)
        # 生成键矩阵K
        k = self.proj_k(h)
        # 生成值矩阵V
        v = self.proj_v(h)

        # 将Q的维度从[B, C, H, W]转换为[B, H*W, C]
        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        # 将K的维度从[B, C, H, W]转换为[B, C, H*W]
        k = k.view(B, C, H * W)
        # 计算注意力权重，并进行缩放
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        # 确保权重矩阵的形状为[B, H*W, H*W]
        assert list(w.shape) == [B, H * W, H * W]
        # 对权重矩阵进行softmax归一化
        w = F.softmax(w, dim=-1)

        # 将V的维度从[B, C, H, W]转换为[B, H*W, C]
        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        # 计算加权和
        h = torch.bmm(w, v)
        # 确保加权和的形状为[B, H*W, C]
        assert list(h.shape) == [B, H * W, C]
        # 将加权和的维度从[B, H*W, C]转换回[B, C, H, W]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        # 通过卷积层生成最终输出
        h = self.proj(h)

        # 返回输入和输出的加权和，实现残差连接
        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        # 初始化父类
        super().__init__()
        # 第一个块：包含GroupNorm、Swish激活函数和卷积层
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),  # 对输入通道进行归一化
            Swish(),  # Swish激活函数
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),  # 3x3卷积层
        )
        # 时间嵌入投影：包含Swish激活函数和线性层
        self.temb_proj = nn.Sequential(
            Swish(),  # Swish激活函数
            nn.Linear(tdim, out_ch),  # 线性层，将时间嵌入维度映射到输出通道数
        )
        # 第二个块：包含GroupNorm、Swish激活函数、Dropout和卷积层
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),  # 对输出通道进行归一化
            Swish(),  # Swish激活函数
            nn.Dropout(dropout),  # Dropout层
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),  # 3x3卷积层
        )
        # 如果输入通道数和输出通道数不同，则使用1x1卷积层进行维度变换
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()  # 如果相同，则直接传递
        # 如果需要注意力机制，则使用AttnBlock，否则使用Identity
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        # 初始化权重
        self.initialize()

    def initialize(self):
        # 遍历所有模块，对卷积层和线性层进行权重初始化
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)  # 使用Xavier均匀分布初始化权重
                init.zeros_(module.bias)  # 偏置初始化为0
        # 对第二个块的最后一个卷积层进行特殊初始化
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        # 第一个块处理输入
        h = self.block1(x)
        # 将时间嵌入投影加到特征图上
        h += self.temb_proj(temb)[:, :, None, None]
        # 第二个块处理特征图
        h = self.block2(h)

        # 残差连接：将输入直接加到输出上
        h = h + self.shortcut(x)
        # 应用注意力机制
        h = self.attn(h)
        return h


class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h


if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)
    print(y.shape)

