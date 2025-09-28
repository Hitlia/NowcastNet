import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
import torch
import glob
from tqdm import tqdm
import time
from utils import warp, make_grid
from evaluator import save_plots
# from submodels import Generative_Encoder, Generative_Decoder, Evolution_Network,Noise_Projector
from model_make import *
from dataset import *
from loss_function.loss_evolution import *
from loss_function.loss_discriminator import *
from loss_function.loss_generation import *
from params import get_args

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@torch.no_grad()
def valid_one_epoch(epoch, args, vali_loader, conv_merge, generator, discriminator, vali_losses):
    pbar = tqdm(vali_loader, total=len(vali_loader), desc=f"Epoch [{epoch}/{args.num_epochs}]")
    generator.evo_net.eval()
    generator.gen_enc.eval()
    generator.gen_dec.eval()
    generator.proj.eval()
    discriminator.eval()

    # 记录本轮测试的指标
    vali_evo_loss = 0.0
    vali_accum_loss = 0.0
    vali_motion_loss = 0.0
    vali_disc_loss = 0.0
    vali_gen_loss = 0.0
    vali_adv_loss = 0.0
    vali_pool_loss = 0.0
    vali_count = 0

    # 开始验证循环(不需要梯度)
    for batch_id, test_ims in enumerate(pbar):
        input_frames, target_frames = test_ims # B, t, H, W, 3
        batch, T, height, width, channels = input_frames.shape
        last_frames = input_frames[:, -1:, :, :, 0].to(args.device) # B, 1, H, W
        input_frames = input_frames.to(args.device)
        target_frames = target_frames[:,:,:,:,0].to(args.device) # B, t, H, W
        
        input_frames = input_frames.permute(0,4,2,3,1) #bs, t, h, w, c -> bs, c, h, w, t
        input_frames = conv_merge(input_frames)
        input_frames = input_frames.permute(0,4,2,3,1).view(batch, -1, height, width) # bs, t, h, w

        # ============ 演化网络前向 + 逐帧梯度截断 ============
        intensity, motion = generator.evo_net(input_frames)
        motion_ = motion.reshape(batch, args.pred_length, 2, height, width)
        intensity_ = intensity.reshape(batch, args.pred_length, 1, height, width)

        series = []
        series_bili = []

        sample_tensor = torch.zeros(1, 1, args.img_height, args.img_width).to(args.device)
        grid = make_grid(sample_tensor)
        grid = grid.repeat(batch, 1, 1, 1)

        # 多步演化, 每帧截断梯度
        for i in range(args.pred_length):
            last_frames = last_frames.detach()

            last_frames_bili = warp(last_frames, motion_[:, i], grid, mode="bilinear",
                                    padding_mode="border")
            last_frames_ = warp(last_frames, motion_[:, i], grid, mode="nearest", padding_mode="border")
            last_frames_ = last_frames_ + intensity_[:, i]
            last_frames_ = last_frames_.detach()

            last_frames = last_frames_
            series.append(last_frames)
            series_bili.append(last_frames_bili)

        evo_result = torch.cat(series, dim=1) # B, T, H, W
        evo_result_bili = torch.cat(series_bili, dim=1)

        # ============ 演化网络损失及更新 ============
        loss_motion = motion_reg(motion_, target_frames)
        loss_accum = accumulation_loss(
            pred_final=evo_result,
            pred_bili=evo_result_bili,
            real=target_frames,  # [B, T_out, H, W]
        )
        loss_evo = loss_accum + args.alpha * loss_motion

        # detach 演化输出, 避免对抗梯度回传
        evo_result_detach = evo_result.detach() / 65

        # ============ 生成网络 + 判别器 ============

        # 1) 生成器前向
        evo_feature = generator.gen_enc(torch.cat([input_frames, evo_result_detach], dim=1)) # B, T_IN+T_OUT, H, W

        gen_result_list = []
        dis_result_pre_list = []
        for _ in range(args.pool_loss_k):
            noise = torch.randn(batch, args.ngf, height // 32, width // 32).to(args.device)
            noise_feature = generator.proj(noise)
            noise_feature = noise_feature.reshape(
                batch, -1, 4, 4, 8, 8
            ).permute(0, 1, 4, 5, 2, 3).reshape(batch, -1, height // 8, width // 8)

            feature = torch.cat([evo_feature, noise_feature], dim=1)
            gen_result = generator.gen_dec(feature, evo_result_detach)
            # shape => [B, T_out, H, W], 视实际而定

            gen_result = gen_result.unsqueeze(2)  # => [B,1,H,W] => or [B,T=1,H,W]
            gen_result_list.append(gen_result)

            # 判别器对 fake
            dis_result_pre = discriminator(gen_result, input_frames.unsqueeze(2))
            dis_result_pre_list.append(dis_result_pre)

        # ============ (a) 生成器更新 ============
        loss_adv = adversarial_loss(dis_result_pre_list)  # 生成器骗判别器
        loss_pool = pool_regularization(target_frames.unsqueeze(2), gen_result_list)
        loss_generative = args.beta * loss_adv + args.gamma * loss_pool


        # 判别器对 real
        dis_result_GT = discriminator(target_frames.unsqueeze(2), input_frames.unsqueeze(2))
        dis_result_pre = discriminator(gen_result.detach(), input_frames.unsqueeze(2))
        # ============ (b) 判别器更新 ============
        loss_disc = discriminator_loss(dis_result_GT, dis_result_pre)

        # 累加loss
        vali_evo_loss += (loss_evo.item())
        vali_accum_loss += (loss_accum.item())
        vali_motion_loss += (loss_motion.item())
        vali_disc_loss += (loss_disc.item())
        vali_gen_loss += (loss_generative.item())
        vali_adv_loss += (loss_adv.item())
        vali_pool_loss += (loss_pool.item())
        vali_count += 1

    # 计算平均loss
    vali_evo_loss /= vali_count
    vali_accum_loss /= vali_count
    vali_motion_loss /= vali_count
    vali_disc_loss /= vali_count
    vali_gen_loss /= vali_count
    vali_adv_loss /= vali_count
    vali_pool_loss /= vali_count

    # 保存到 vali_losses
    vali_losses.append({
        'epoch': epoch,
        'loss_evo': vali_evo_loss,
        'loss_accum': vali_accum_loss,
        'loss_motion': vali_motion_loss,
        'loss_disc': vali_disc_loss,
        'loss_gen': vali_gen_loss,
        'loss_adv': vali_adv_loss,
        'loss_pool': vali_pool_loss,
    })

    print(
        f"=============================================> Validation on epoch {epoch}: evo_loss={vali_evo_loss:.4f}, acc={vali_accum_loss:.4f}, mot={vali_motion_loss:.4f}, disc_loss={vali_disc_loss:.4f}, gen_loss={vali_gen_loss:.4f}, adv={vali_adv_loss:.4f}, pool={vali_pool_loss:.4f}")

def train_one_epoch(epoch, args, train_loader, conv_merge, generator, discriminator, optim_evo, optim_gen, optim_disc, train_losses):
    pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch}/{args.num_epochs}]")

    if epoch == args.lr_decrease_epoch:
        optim_evo = torch.optim.Adam(generator.evo_net.parameters(), lr=args.lr_evo_de, betas=(0.5, 0.999))
        optim_gen = torch.optim.Adam(
            list(generator.gen_enc.parameters())
            + list(generator.gen_dec.parameters())
            + list(generator.proj.parameters()),
            lr=args.lr_gen_de, betas=(0.5, 0.999)
        )

    train_evo_loss = 0.0
    train_accum_loss = 0.0
    train_motion_loss = 0.0
    train_disc_loss = 0.0
    train_gen_loss = 0.0
    train_adv_loss = 0.0
    train_pool_loss = 0.0
    train_count = 0

    t_dataload = time.time()
    for batch_id, test_ims in enumerate(pbar):

        generator.evo_net.train()
        generator.gen_enc.train()
        generator.gen_dec.train()
        generator.proj.train()
        discriminator.train()

        input_frames, target_frames = test_ims
        batch, T, height, width, channels = input_frames.shape
        last_frames = input_frames[:, -1:, :, :, 0].to(args.device)
        input_frames = input_frames.to(args.device)
        target_frames = target_frames[:,:,:,:,0].to(args.device)
        
        input_frames = input_frames.permute(0,4,2,3,1) #bs, t, h, w, c -> bs, c, h, w, t
        input_frames = conv_merge(input_frames)
        input_frames = input_frames.permute(0,4,2,3,1).view(batch, -1, height, width) # bs, t, h, w

        t_dataload = time.time() - t_dataload
            # ============ 演化网络前向 + 逐帧梯度截断 ============
        t_forward = time.time()
        intensity, motion = generator.evo_net(input_frames)
        motion_ = motion.reshape(batch, args.pred_length, 2, height, width)
        intensity_ = intensity.reshape(batch, args.pred_length, 1, height, width)

        series = []
        series_bili = []

        sample_tensor = torch.zeros(1, 1, args.img_height, args.img_width).to(args.device)
        grid = make_grid(sample_tensor)
        grid = grid.repeat(batch, 1, 1, 1)

        t_forward = time.time() - t_forward
        t_evo = time.time()

        # EvolutionNet.inc.double_conv[0].weight.grad
        # 多步演化, 每帧截断梯度
        for i in range(args.pred_length):
            x_t = last_frames.detach()

            x_t_dot_bili = warp(x_t, motion_[:, i], grid, mode="bilinear", padding_mode="border")
            x_t_dot = warp(x_t, motion_[:, i], grid, mode="nearest", padding_mode="border")
            x_t_dot_dot = x_t_dot.detach() + intensity_[:, i]
            # last_frames_ = last_frames_

            last_frames = x_t_dot_dot
            series.append(x_t_dot_dot)
            series_bili.append(x_t_dot_bili)

        evo_result = torch.cat(series, dim=1)
        evo_result_bili = torch.cat(series_bili, dim=1)

        t_evo = time.time() - t_evo
        t_backward = time.time()

        # ============ 演化网络损失及更新 ============
        loss_motion = motion_reg(motion_, target_frames)
        loss_accum = accumulation_loss(
            pred_final=evo_result,
            pred_bili=evo_result_bili,
            real=target_frames,  # [B, T_out, H, W]
        )
        loss_evo = loss_accum + args.alpha * loss_motion
        if epoch < args.lr_stop_epoch:
            optim_evo.zero_grad()
            loss_evo.backward(retain_graph=True)
            optim_evo.step()

        gen_result = evo_result
        t_backward = time.time() - t_backward


        t_gen_forward = time.time()
        # detach 演化输出, 避免对抗梯度回传
        evo_result_detach = evo_result.detach() / 65

        # ============ 生成网络 + 判别器 ============

        # 1) 生成器前向
        evo_feature = generator.gen_enc(torch.cat([input_frames, evo_result_detach], dim=1))

        gen_result_list = []
        dis_result_pre_list = []
        for _ in range(args.pool_loss_k):
            noise = torch.randn(batch, args.ngf, height // 32, width // 32).to(args.device)
            noise_feature = generator.proj(noise)
            noise_feature = noise_feature.reshape(
                batch, -1, 4, 4, 8, 8
            ).permute(0, 1, 4, 5, 2, 3).reshape(batch, -1, height // 8, width // 8)

            feature = torch.cat([evo_feature, noise_feature], dim=1)
            gen_result = generator.gen_dec(feature, evo_result_detach)
            # shape => [B, T_out, H, W], 视实际而定

            gen_result = gen_result.unsqueeze(2)  # => [B,1,H,W] => or [B,T=1,H,W]
            gen_result_list.append(gen_result)

            # 判别器对 fake
            dis_result_pre = discriminator(gen_result, input_frames.unsqueeze(2))
            dis_result_pre_list.append(dis_result_pre)

        t_gen_forward = time.time() - t_gen_forward
        t_gen_backward = time.time()
        # ============ (a) 生成器更新 ============
        loss_adv = adversarial_loss(dis_result_pre_list)  # 生成器骗判别器
        loss_pool = pool_regularization(target_frames.unsqueeze(2), gen_result_list)
        loss_generative = args.beta * loss_adv + args.gamma * loss_pool

        optim_gen.zero_grad()
        loss_generative.backward(retain_graph=True)
        optim_gen.step()

        t_gen_backward = time.time() - t_gen_backward
        t_dis_forward = time.time()

        # 判别器对 real
        dis_result_GT = discriminator(target_frames.unsqueeze(2), input_frames.unsqueeze(2))
        dis_result_pre = discriminator(gen_result.detach(), input_frames.unsqueeze(2))

        t_dis_forward = time.time() - t_dis_forward
        t_dis_backward = time.time()
        # ============ (b) 判别器更新 ============
        loss_disc = discriminator_loss(dis_result_GT, dis_result_pre)
        optim_disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        optim_disc.step()

        t_dis_backward = time.time() - t_dis_backward

        # 打印步骤时间
        # print(f'\ndata_load={t_dataload:.2f}, evo_f={t_forward:.2f}, evo_phy={t_evo:.2f}, evo_b={t_backward:.2f}, gen_f={t_gen_forward:.2f}, gen_b={t_gen_backward:.2f}, dis_f={t_dis_forward:.2f}, dis_b={t_dis_backward:.2f}')
        t_dataload = time.time()


        # ============ 打印与 tqdm 显示 ============
        pbar.set_postfix({
            'loss_evo': f"{loss_evo.item():.4f}",
            'acc': f"{loss_accum.item():.4f}",
            'mot': f"{loss_motion.item():.4f}",
            'loss_disc': f"{loss_disc.item():.4f}",
            'loss_gen': f"{loss_generative.item():.4f}",
            'adv': f"{loss_adv.item():.4f}",
            'pool': f"{loss_pool.item():.4f}",
        })

        train_evo_loss += loss_evo.item()
        train_accum_loss += loss_accum.item()
        train_motion_loss += loss_motion.item()
        train_disc_loss += loss_disc.item()
        train_gen_loss += loss_generative.item()
        train_adv_loss += loss_adv.item()
        train_pool_loss += loss_pool.item()
        train_count += 1
        
        if batch_id % 80 == 0:
            with open(args.gen_frm_dir + '/log.output','a') as file:
                print("Epoch:{},Batch:{}".format(epoch,batch_id),file=file)
                print("train_evo_loss:{},train_disc_loss:{},train_gen_loss:{},train_adv_loss:{},train_pool_loss:{}".format(
                    train_evo_loss/train_count,train_disc_loss/train_count,train_gen_loss/train_count,train_adv_loss/train_count,train_pool_loss/train_count),file=file)
                print("----------------------------------------",file=file)
        if batch_id % 80 == 0:
            target_frames = target_frames.cpu()
            temp_plot = gen_result_list[0][0,0,0]
            save_plots(input_frames[0,0],epoch,batch_id,os.path.join(args.gen_frm_dir, 'X'))
            save_plots(target_frames[0,0],epoch,batch_id,os.path.join(args.gen_frm_dir, 'Y'))
            save_plots(temp_plot,epoch,batch_id,os.path.join(args.gen_frm_dir, 'Result'))
    
    train_losses = {
        'epoch': epoch,
        'loss_evo': train_evo_loss/train_count,
        'loss_accum': train_accum_loss/train_count,
        'loss_motion': train_motion_loss/train_count,
        'loss_disc': train_disc_loss/train_count,
        'loss_gen': train_gen_loss/train_count,
        'loss_adv': train_adv_loss/train_count,
        'loss_pool': train_pool_loss/train_count,
    }
    
    return train_losses
    
def train(args):
    # 创建输出文件夹（若已存在则删除重建）
    args.gen_frm_dir = os.path.join(args.gen_frm_dir, args.experiment)
    if os.path.exists(args.gen_frm_dir):
        args.gen_frm_dir = args.gen_frm_dir + "_new"
        # shutil.rmtree(args.gen_frm_dir)
    os.makedirs(args.gen_frm_dir)
    
    result_dir = os.path.join(args.gen_frm_dir, 'Result')
    os.mkdir(result_dir)
    x_dir = os.path.join(args.gen_frm_dir, 'X')
    os.mkdir(x_dir)
    y_dir = os.path.join(args.gen_frm_dir, 'Y')
    os.mkdir(y_dir)

    # model
    print('>>> 初始化并加载模型 ...')
    conv_merge = nn.Conv3d(in_channels=3, out_channels=1, kernel_size=1).to(args.device)
    generator = NowcastNet(args).to(args.device)
    discriminator = Temporal_Discriminator(args).to(args.device)
        
    # optimizer
    optim_evo = torch.optim.Adam(generator.evo_net.parameters(), lr=args.lr_evo, betas=(0.5,0.999))
    optim_gen = torch.optim.Adam(
        list(generator.gen_enc.parameters())
        + list(generator.gen_dec.parameters())
        + list(generator.proj.parameters()),
        lr=args.lr_gen, betas=(0.5,0.999)
    )
    optim_disc = torch.optim.Adam(discriminator.parameters(), lr=args.lr_disc, betas=(0.5,0.999))
    
    # >>> 新增: 创建 / 加载 checkpoint 逻辑 <<<
    checkpoint_dir = os.path.join(args.checkpoint_path, args.experiment)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 尝试搜集已有 ckpt
    ckpts = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    ckpts.sort()  # 根据名称排序，最后一个视作最新

    start_epoch = 0
    train_losses = []
    vali_losses = []

    if len(ckpts) > 0:
        latest_ckpt = ckpts[-1]
        print(f"[INFO] Found checkpoint: {latest_ckpt}, now loading...")
        checkpoint = torch.load(latest_ckpt, map_location=args.device)

        # 加载模型参数
        conv_merge.load_state_dict(checkpoint['ConvMerge'])
        generator.evo_net.load_state_dict(checkpoint['EvolutionNet'])
        generator.gen_enc.load_state_dict(checkpoint['GenerativeEncoder'])
        generator.gen_dec.load_state_dict(checkpoint['GenerativeDecoder'])
        generator.proj.load_state_dict(checkpoint['NoiseProjector'])
        discriminator.load_state_dict(checkpoint['TemporalDiscriminator'])

        # 加载优化器参数
        optim_evo.load_state_dict(checkpoint['optim_evo'])
        optim_gen.load_state_dict(checkpoint['optim_gen'])
        optim_disc.load_state_dict(checkpoint['optim_disc'])

        # 恢复 epoch 及迭代数
        start_epoch = checkpoint.get('epoch', 0)
        train_losses = checkpoint.get('train_losses', [])
        vali_losses = checkpoint.get('vali_losses', [])
        print(f"[INFO] Loaded checkpoint successfully! start_epoch={start_epoch}")

    else:
        print("[INFO] No existing checkpoint found, starting from scratch...")
        latest_ckpt = None
        
    # dataloaders
    [train_loader, vali_loader] = make_dataloaders(args, splits=['train','vali'])
    
    # training
    for epoch in range(start_epoch, args.num_epochs):
        # train
        train_losses_epoch = train_one_epoch(epoch, args, train_loader, conv_merge, generator, discriminator, optim_evo, optim_gen, optim_disc, train_losses)
        train_losses.append(train_losses_epoch)
        # vali
        vali_losses_epoch = valid_one_epoch(epoch, args, vali_loader, conv_merge, generator, discriminator, vali_losses)
        vali_losses.append(vali_losses_epoch)
        
        # 存储checkpoints
        checkpoint_dir = os.path.join(args.checkpoint_path, args.experiment)
        ckpt_path = os.path.join(checkpoint_dir, f"ckpt_epoch{epoch:03d}.pth")
        torch.save({
            'epoch': epoch,
            'train_losses': train_losses,
            'vali_losses': vali_losses,
            'ConvMerge': conv_merge.state_dict(),
            'EvolutionNet': generator.evo_net.state_dict(),
            'GenerativeEncoder': generator.gen_enc.state_dict(),
            'GenerativeDecoder': generator.gen_dec.state_dict(),
            'NoiseProjector': generator.proj.state_dict(),
            'TemporalDiscriminator': discriminator.state_dict(),
            'optim_evo': optim_evo.state_dict(),
            'optim_gen': optim_gen.state_dict(),
            'optim_disc': optim_disc.state_dict(),
        }, ckpt_path)
        print(f"[INFO] Saved checkpoint to {ckpt_path}")
        
if __name__ == '__main__':
    args = get_args()
    train(args)