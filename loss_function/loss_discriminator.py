import torch
import torch.nn as nn

# 判别器：BCE with logits
bce_crit = nn.BCEWithLogitsLoss()

def discriminator_loss(d_real, d_fake):
    """
    x_input: 真实的条件输入 [B, T_in, H, W, C]
    x_real:  真实目标帧       [B, T_out, H, W, C]
    x_fake:  生成网络预测帧   [B, T_out, H, W, C]

    discriminator(...) -> [B, 1, H/16, W/16]
    """

    # 1) 判别器对 "真实数据" 期望输出 1
    label_real = torch.ones_like(d_real)      # 与 d_real 同形状
    loss_real = bce_crit(d_real, label_real)

    if isinstance(d_fake, torch.Tensor):
        # 2) 判别器对 "生成数据" 期望输出 0
        label_fake = torch.zeros_like(d_fake)
        loss_fake = bce_crit(d_fake, label_fake)
    else:
        loss_fake = 0
        for d_fake_one in d_fake:
            loss_fake += bce_crit(d_fake_one, torch.zeros_like(d_fake_one))
        loss_fake = loss_fake / len(d_fake)

    # 判别器总损失
    loss_disc = loss_real + loss_fake
    return loss_disc
