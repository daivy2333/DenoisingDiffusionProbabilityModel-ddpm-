
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    # 获取t所在的设备（CPU或GPU）
    device = t.device
    # 使用torch.gather函数从v中按照t指定的索引提取元素
    # v: 输入的张量，t: 指定的索引，dim=0表示沿着第0维进行索引
    # .float()将提取出的元素转换为浮点数
    # .to(device)将结果移动到与t相同的设备上
    out = torch.gather(v, index=t, dim=0).float().to(device)
    # 将提取出的结果重新形状为[t.shape[0]] + [1] * (len(x_shape) - 1)
    # t.shape[0]表示batch_size，[1] * (len(x_shape) - 1)表示在除了batch_size之外的维度上添加1
    # 这样做的目的是为了方便后续的广播操作
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        # 初始化模型
        self.model = model
        # 设置扩散步数
        self.T = T

        # 注册缓冲区，存储从beta_1到beta_T的线性空间，用于扩散过程
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        # 计算alphas，即1减去betas
        alphas = 1. - self.betas
        # 计算alphas的累积乘积，用于后续计算
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        # 生成一个随机的时间步t，范围在0到self.T之间，大小与x_0的第一个维度相同，设备与x_0相同
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        # 生成与x_0形状相同的正态分布噪声
        noise = torch.randn_like(x_0)
        # 计算在时间步t的噪声数据x_t
        # 使用extract函数从self.sqrt_alphas_bar中提取对应时间步t的值，并与x_0相乘
        # 使用extract函数从self.sqrt_one_minus_alphas_bar中提取对应时间步t的值，并与噪声相乘
        # 将上述两个结果相加得到x_t
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        # 计算模型预测的噪声与真实噪声之间的均方误差损失
        # self.model(x_t, t)表示模型在时间步t对x_t的预测
        # noise表示真实的噪声
        # reduction='none'表示不进行任何形式的损失聚合，返回每个样本的损失
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        # 返回计算得到的损失
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        # 初始化模型、时间步数T
        self.model = model
        self.T = T

        # 注册缓冲区，存储从beta_1到beta_T的线性空间
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        # 计算alphas，即1减去betas
        alphas = 1. - self.betas
        # 计算累积乘积alphas_bar
        alphas_bar = torch.cumprod(alphas, dim=0)
        # 计算前一个时间步的alphas_bar
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        coeff1 = torch.sqrt(1. / alphas)
        coeff2 = coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar)
        # 计算系数coeff1和coeff2
        self.register_buffer('coeff1', coeff1)
        self.register_buffer('coeff2', coeff2)

        # 计算后验方差
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        # 确保x_t和eps的形状相同
        assert x_t.shape == eps.shape
        # 根据公式计算x_{t-1}的均值
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        
        print("NaN in x_t:", torch.isnan(x_t).sum().item())

        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        # 初始化x_t为输入的x_T
        x_t = x_T
        # 从最后一个时间步开始，倒序遍历到第一个时间步
        for time_step in reversed(range(self.T)):
            # 打印当前时间步
            print(time_step)
            # 创建一个与x_T形状相同的张量，所有元素初始化为当前时间步的值
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            # 计算当前时间步的均值和方差
            mean, var= self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            if time_step == 999:
                print("==> time_step: ", time_step)
                print("var min:", var.min().item(), "var max:", var.max().item())
                print("NaN in var:", torch.isnan(var).sum().item())
                print("NaN in mean:", torch.isnan(mean).sum().item())
           
            eps = 1e-5  # 或 1e-6，根据实际需要调整
            x_t = mean + torch.sqrt(var + eps) * noise

            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        

        return torch.clip(x_0, -1, 1)   


