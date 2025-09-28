import torch
import torch.nn.functional as F


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
        loss = torch.sum(loss_n, -2).mean()
    else:
        loss = diff_w.mean() * gt.shape[1]
    return loss


def sobel_filter_2d(x):
    """ 对 2D 张量 x (B, H, W) 做 Sobel 操作, 返回梯度近似. """
    # 这里使用简化方式: 分别对 x 做 x方向, y方向 的 Sobel
    # 你也可以自己写更灵活的 2D 卷积
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=x.device).view(1,1,3,3)

    # x shape: [B, H, W], reshape -> [B,1,H,W] 方便 conv2d
    x_ = x.unsqueeze(1)
    gx = F.conv2d(x_, sobel_x, padding=1)  # [B,1,H,W]
    gy = F.conv2d(x_, sobel_y, padding=1)
    # 返回 [B,H,W], 两个分量可合成为范数,或分别返回
    gx = gx.squeeze(1)
    gy = gy.squeeze(1)
    return gx, gy


def motion_reg(v, x, reg_loss=True, value_lim=[0,65]):
    """
    v: [B, T, 2, H, W]  -> motion field
    x: [B, T, H, W]     -> real frames (for weighting)
    论文 (6) 式:
    J_motion = Σ_t || ∂v^1_t/∂(x,y)*w(x_t) ||^2 +  || ∂v^2_t/∂(x,y)*w(x_t) ||^2
    """
    B, T, _, H, W = v.shape

    total_reg = 0.0
    for t in range(T):
        # v[:, t, 0] => vx; v[:, t, 1] => vy
        vx = v[:, t, 0]
        vy = v[:, t, 1]
        w = weight_func(x[:, t], value_lim)  # [B,H,W]

        # 计算 vx, vy 的梯度
        gx_vx, gy_vx = sobel_filter_2d(vx)
        gx_vy, gy_vy = sobel_filter_2d(vy)
        if reg_loss:
            # |∇vx|^2 + |∇vy|^2, 并乘上 w(x)
            reg_vx = torch.sum((gx_vx**2 + gy_vx**2), (-1, -2)) / torch.sum(w, (-1, -2))
            reg_vy = torch.sum((gx_vy**2 + gy_vy**2), (-1, -2)) / torch.sum(w, (-1, -2))
            total_reg += reg_vx.mean() + reg_vy.mean()
        else:
            reg_vx = (gx_vx**2 + gy_vx**2) * w
            reg_vy = (gx_vy**2 + gy_vy**2) * w
            total_reg += reg_vx.mean() + reg_vy.mean()

    return total_reg


def accumulation_loss(pred_final, pred_bili, real, reg_loss=True, value_lim=[0,1]):
    """
    pred_final: x''_t (nearest+intensity)
    pred_bili : x'_t^{bili} (bilinear+intensity) - 可选, 若不需要就传 None
    real      : ground truth, shape [B,T,H,W]
    motion    : [B,T,2,H,W]  (vx, vy)
    lam       : λ
    """
    # 1) 累积损失
    #   包含 wdis( real, pred_bili ) + wdis( real, pred_final )
    #   若 pred_bili 不存在, 只对 pred_final 做

    accum_loss = 0.0
    if reg_loss:
        accum_loss += wdis_l1(pred_final, real, reg_loss, value_lim)
        if pred_bili is not None:
            accum_loss += wdis_l1(pred_bili, real, reg_loss, value_lim)
            accum_loss /= 2
        accum_loss = accum_loss
    else:
        accum_loss += wdis_l1(pred_final, real, reg_loss, value_lim).mean()
        if pred_bili is not None:
            accum_loss += wdis_l1(pred_bili, real, reg_loss, value_lim).mean()
            accum_loss /= 2
        accum_loss = accum_loss * real.shape[1]

    # 2) 运动正则
    # motion_loss = motion_reg(motion, real)

    # 总损失
    # loss_evo = accum_loss + lam * motion_loss
    return accum_loss
