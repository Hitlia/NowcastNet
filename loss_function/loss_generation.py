import torch.nn.functional as F
import torch
import torch.nn as nn

bce_crit = nn.BCEWithLogitsLoss()

def weight_func(x, value_lim):
    M = value_lim[1]
    m = value_lim[0]
    return torch.clamp(1.0 + (x-m)/(M-m)*128, max=24.0)

def wdis_l1(pred, gt, reg_loss, value_lim):
    """
    论文(5)式中的加权 L1 距离:
    L_{wdis}(x, x') = || x - x' ||_1 ⊙ w(x)
    pred, gt: [B, T, H, W]
    """
    w = weight_func(gt, value_lim)
    diff_w = torch.abs(pred - gt) * w
    if reg_loss:
        loss_n = torch.sum(diff_w, (-1, -2)) / torch.sum(w, (-1, -2))
        loss = torch.sum(loss_n, -1).mean()
    else:
        loss = diff_w.mean() * gt.shape[1]
    return loss


def pool_regularization(x_real, x_fakes, kernel_size=5, stride=2, reg_loss=True, value_lim=[0,65]):
    """
    x_real: [B, T, H, W, C], 真实场
    x_fakes: list or tensor of shape (k, [B, T, H, W, C]) or something
             代表 k 个ensemble预测

    对每个预测 x_fakes[i] 做一次 pool, 跟 x_real 做比较。
    pool方式：这里示例 2D max-pool
    (实际可针对T个帧做更复杂的 3D pool, 视论文需要)

    Return: scalar, J_pool
    """
    # device = x_real.device
    B, T, C, H, W = x_real.shape

    # x_real = x_real.squeeze(2)
    # if isinstance(x_fakes, torch.Tensor):
    #     x_fakes = x_fakes.squeeze(2)
    # else:
    #     x_fakes = [xx.squeeze(2) for xx in x_fakes]

    # [B, T, C, H, W] => [B*T*C, 1, H, W], 方便 2D 池化
    # print(x_real.shape)
    real_reshaped = x_real.reshape(B *T * C, H, W)
    # print(real_reshaped.shape)

    # 做 max-pool
    real_pool = F.max_pool2d(real_reshaped, kernel_size, stride=stride)
    # real_pool => [B*T*C, 1, H', W']

    # 对多个预测做同样操作并累加 wdis
    total_loss = 0.0
    k = len(x_fakes)

    x_fakes_ = []
    for i in range(k):
        fake_i = x_fakes[i]  # [B, T, H, W, C]
        fake_reshaped = fake_i.reshape(B*T * C,1, H, W)
        fake_pool = F.max_pool2d(fake_reshaped, kernel_size, stride=stride)
        # shape [B*T*C, 1, H', W']
        x_fakes_.append(fake_pool)
    x_fakes_mean = torch.mean(torch.cat(x_fakes_, dim=1),dim=1)
    # 计算 wdis
    if reg_loss:
        wloss = wdis_l1(x_fakes_mean, real_pool, reg_loss, value_lim)/B
    else:
        wloss = wdis_l1(x_fakes_mean, real_pool, reg_loss, value_lim)/x_real.shape[1]
    total_loss += wloss

    # 这里不做平均也可，看您是否要 sum or avg
    return total_loss


def adversarial_loss(d_fake):
    """
    生成网络总损失 = J_adv + J_pool

    x_input: [B, T_in,  H, W, C]
    x_fake:  [B, T_out, H, W, C] - 这次生成的某一次(单次)预测
    x_real:  [B, T_out, H, W, C] - 真实值(可选, 仅在J_pool里会用)
    ensemble_fakes: list of [B, T_out, H, W, C], 代表多个潜在向量 z_i 对应的生成输出
    """
    # --- 1) 对抗损失 (J_adv) ---
    if isinstance(d_fake, torch.Tensor):
        # 2) 判别器对 "生成数据" 期望输出 0
        label_fake = torch.ones_like(d_fake)
        loss_adv = bce_crit(d_fake, label_fake)
    else:
        loss_adv = 0
        for d_fake_one in d_fake:
            loss_adv += bce_crit(d_fake_one, torch.ones_like(d_fake_one))
        loss_adv = loss_adv / len(d_fake)


    # 总损失
    # 论文 (12) 形式: J_generative = J_adv + β J_disc + γ J_pool
    # 这里示例: J_adv + J_pool 即可
    return loss_adv
