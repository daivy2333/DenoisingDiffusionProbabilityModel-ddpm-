from Diffusion.Train import train, denoise_from_real_image, sample_from_noise


# 先跑400个epoch，再跑600
def main(model_config = None):
    modelConfig = {
        "state": "eval", # or eval
        "epoch": 1200, # 1000
        "batch_size": 128, # 128
        "T": 1000,
        "channel": 64, # 64 128
        "channel_mult": [1, 2], #1 2 3 4
        "attn": [], # [2]
        "num_res_blocks": 1, # 2
        "dropout": 0.1, # 0.15
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": "ckpt_1199_.pt" ,### 指定加载模型用于重启中断 "ckpt_133_.pt" 重启训练用到
        "save_weight_dir": "DenoisingDiffusionProbabilityModel-ddpm-/Checkpoints/",
        "test_load_weight": "ckpt_1199_.pt",### 指定加载模型用于测试 "best_model.pt"
        "sampled_dir": "DenoisingDiffusionProbabilityModel-ddpm-/SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8,
        "resume_epoch": 1199, ### 重启起点，重启用到
        "log_images": True, # 采样开关
        "log_image_every": 20,### 图像采样间隔，受限于本人的显存，设置为20
        "image_path":"DenoisingDiffusionProbabilityModel-ddpm-\SampledImgs\image.png", #修改输入图片用到
        "eval_model":"sample" # or "sample"  # 选择评估模型的类型
        }
    
    """
    训练关键参数："state": "train"
    "epoch": 400, # 400
    "training_load_weight": "ckpt_79_.pt"
    "resume_epoch": 79

    评估关键参数："state": "eval"
    "eval_model":"denoise" # or "sample"
    "test_load_weight": "best_model.pt"
    "image_path":""
    """

    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        if modelConfig["eval_model"] == "denoise":
            denoise_from_real_image(modelConfig)
        else:
            
            sample_from_noise(modelConfig)


if __name__ == '__main__':
    main()
