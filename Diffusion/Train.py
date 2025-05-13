
import os
from typing import Dict
from torch.utils.tensorboard import SummaryWriter
import torchvision
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
import torch.backends.cudnn
from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler
from torch import amp
# 以后import统一放上面
def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    torch.backends.cudnn.benchmark = True # 使得卷积操作更快
    # dataset
    writer = SummaryWriter(log_dir=modelConfig.get("log_dir", "./logs"))

    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    scaler = amp.GradScaler("cuda")
    best_loss = float('inf')
    start_epoch = modelConfig.get("resume_epoch", 0)
    # 断点功能
    if modelConfig["training_load_weight"] is not None:
        checkpoint = torch.load(
            os.path.join(modelConfig["save_weight_dir"], 
                         modelConfig["training_load_weight"]), 
                         map_location=device,
                         weights_only=False
                         )
        
        
        net_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        warmUpScheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"✅ Resuming training from epoch {start_epoch}")
    
    for e in range(start_epoch, modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                with amp.autocast("cuda"):
                    loss = trainer(x_0).sum() / 1000.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
                # TensorBoard 记录
        writer.add_scalar("Loss/train", loss.item(), e)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], e)
        if modelConfig.get("log_images", True):
            # 每 N 个 epoch 执行一次图像采样和写入
            if (e + 1) % modelConfig.get("log_image_every", 20) == 0:
                net_model.eval()
                try:
                    with torch.no_grad():
                        noisyImage = torch.randn(
                            size=[modelConfig["nrow"], 3, 32, 32], device=device)
                        sampledImgs = trainer(noisyImage)
                        sampledImgs = sampledImgs * 0.5 + 0.5  # 反归一化
                        writer.add_images("Sampled Images", sampledImgs, e)
                finally:
                    net_model.train()


        # 保存模型每 50 个 epoch
        if (e + 1) % 50 == 0:
            torch.save({
                'epoch': e,
                'model_state_dict': net_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': warmUpScheduler.state_dict()
            }, os.path.join(modelConfig["save_weight_dir"], f'ckpt_{e}_.pt'))

            
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(net_model.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], 'best_model.pt'))
            
    writer.close()


def denoise_from_xt(model, x_t, t, modelConfig):
    model.eval()
    device = x_t.device
    x = x_t.clone()

    # 构造 beta, alpha, alpha_bar
    betas = torch.linspace(modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
    alphas = 1. - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    for time_step in reversed(range(t)):
        t_batch = torch.full((x.size(0),), time_step, device=device, dtype=torch.long)
        noise_pred = model(x, t_batch)
        
        alpha = alphas[time_step]
        alpha_bar = alpha_bars[time_step]
        beta = betas[time_step]

        # 逆过程计算 x_{t-1}，简单采样
        x = (1. / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * noise_pred)

        # 加一点随机噪声（除最后一步）
        if time_step > 0:
            noise = torch.randn_like(x)
            x += torch.sqrt(beta) * noise

    return x


def denoise_real_image(model, sampler, image_path, modelConfig):
    device = torch.device(modelConfig["device"])
    
    # 加载并处理图像
    transform = Compose([
        Resize((32, 32)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = Image.open(image_path).convert('RGB')
    x_0 = transform(img).unsqueeze(0).to(device)  # (1, 3, 32, 32)

    # 添加噪声到第 t 步
    t = 50 # 可以尝试 100, 250, 500, 越大越模糊
    noise = torch.randn_like(x_0)
    alpha_bar = torch.cumprod(torch.linspace(1 - modelConfig["beta_1"], 1 - modelConfig["beta_T"], modelConfig["T"]), dim=0)
    x_t = torch.sqrt(alpha_bar[t]) * x_0 + torch.sqrt(1 - alpha_bar[t]) * noise

    # 去噪过程x_recon = sampler(x_t)。旧的，没用
    x_recon = denoise_from_xt(model, x_t, t, modelConfig)  # 用新的去噪函数

    # 检测范围
    print("x_0 range:", x_0.min().item(), x_0.max().item())
    print("x_t range:", x_t.min().item(), x_t.max().item())
    print("x_recon range:", x_recon.min().item(), x_recon.max().item())

    # 保存原图、加噪图和还原图
    from torchvision.utils import save_image
    output_dir = modelConfig["sampled_dir"]
    save_image(x_0 * 0.5 + 0.5, os.path.join(output_dir, "real_input.png"))
    save_image(x_t * 0.5 + 0.5, os.path.join(output_dir, f"noisy_t{t}.png"))
    save_image(x_recon * 0.5 + 0.5, os.path.join(output_dir, f"recon_from_t{t}.png"))
    print(f"✅ Denoised image saved from t={t}")

"""
def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, 32, 32], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(
            modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save_image(sampledImgs, os.path.join(
            modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])
        image_path = modelConfig.get("image_path", None)
        denoise_real_image(model, sampler, image_path, modelConfig)
"""


def denoise_from_real_image(modelConfig: Dict):
    """
    从真实图像去噪。
    """
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        model.eval()

        sampler = GaussianDiffusionSampler(model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

        image_path = modelConfig.get("image_path", None)
        if image_path is None:
            raise ValueError("image_path not specified in modelConfig.")
        denoise_real_image(model, sampler, image_path, modelConfig)

def sample_from_noise(modelConfig: Dict):
    """
    随机噪声生成图像并保存。
    """
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        model.eval()

        sampler = GaussianDiffusionSampler(model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

        noisyImage = torch.randn(size=[modelConfig["batch_size"], 3, 32, 32], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])

        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5
        save_image(sampledImgs, os.path.join(modelConfig["sampled_dir"], modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])
        print("✅ Sampled images from noise saved.")
