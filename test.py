import os
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
def test_one_epoch(epoch, args, test_loader, conv_merge, generator, discriminator):
    pbar = tqdm(test_loader, total=len(test_loader), desc=f"Epoch [{1}/{1}]")
    generator.evo_net.eval()
    generator.gen_enc.eval()
    generator.gen_dec.eval()
    generator.proj.eval()
    discriminator.eval()

    # 记录本轮测试的指标
    test_evo_loss = 0.0
    test_accum_loss = 0.0
    test_motion_loss = 0.0
    test_disc_loss = 0.0
    test_gen_loss = 0.0
    test_adv_loss = 0.0
    test_pool_loss = 0.0
    test_count = 0

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
        test_evo_loss += (loss_evo.item())
        test_accum_loss += (loss_accum.item())
        test_motion_loss += (loss_motion.item())
        test_disc_loss += (loss_disc.item())
        test_gen_loss += (loss_generative.item())
        test_adv_loss += (loss_adv.item())
        test_pool_loss += (loss_pool.item())
        test_count += 1
        
        if batch_id % 10 == 0:
            with open(args.test_results_dir + '/log.output','a') as file:
                print("Epoch:{},Batch:{}".format(epoch,batch_id),file=file)
                print("train_evo_loss:{},train_disc_loss:{},train_gen_loss:{},train_adv_loss:{},train_pool_loss:{}".format(
                    test_evo_loss/test_count,test_disc_loss/test_count,test_gen_loss/test_count,test_adv_loss/test_count,test_pool_loss/test_count),file=file)
                print("----------------------------------------",file=file)
        if batch_id % 80 == 0:
            target_frames = target_frames.cpu()
            temp_plot = gen_result_list[0][0,0,0]
            save_plots(input_frames[0,0],epoch,batch_id,os.path.join(args.test_results_dir, 'X'))
            save_plots(target_frames[0,0],epoch,batch_id,os.path.join(args.test_results_dir, 'Y'))
            save_plots(temp_plot,epoch,batch_id,os.path.join(args.test_results_dir, 'Result'))

    # 计算平均loss
    test_evo_loss /= test_count
    test_accum_loss /= test_count
    test_motion_loss /= test_count
    test_disc_loss /= test_count
    test_gen_loss /= test_count
    test_adv_loss /= test_count
    test_pool_loss /= test_count

    # 保存到 vali_losses
    test_losses = {
        'epoch': epoch,
        'loss_evo': test_evo_loss,
        'loss_accum': test_accum_loss,
        'loss_motion': test_motion_loss,
        'loss_disc': test_disc_loss,
        'loss_gen': test_gen_loss,
        'loss_adv': test_adv_loss,
        'loss_pool': test_pool_loss,
    }

    print(
        f"=============================================> Validation on epoch {epoch}: evo_loss={test_evo_loss:.4f}, acc={test_accum_loss:.4f}, mot={test_motion_loss:.4f}, disc_loss={test_disc_loss:.4f}, gen_loss={test_gen_loss:.4f}, adv={test_adv_loss:.4f}, pool={test_pool_loss:.4f}")    
    
    return test_losses
    
def test(args):
    # 创建输出文件夹（若已存在则删除重建）
    args.test_results_dir = os.path.join(args.test_results_dir, args.experiment)
    if os.path.exists(args.test_results_dir):
        shutil.rmtree(args.test_results_dir)
    os.makedirs(args.test_results_dir)
    
    result_dir = os.path.join(args.test_results_dir, 'Result')
    os.mkdir(result_dir)
    x_dir = os.path.join(args.test_results_dir, 'X')
    os.mkdir(x_dir)
    y_dir = os.path.join(args.test_results_dir, 'Y')
    os.mkdir(y_dir)

    # model
    print('>>> 初始化并加载模型 ...')
    conv_merge = nn.Conv3d(in_channels=3, out_channels=1, kernel_size=1)
    generator = NowcastNet(args).to(args.device)
    discriminator = Temporal_Discriminator(args).to(args.device)
    
    # >>> 新增: 创建 / 加载 checkpoint 逻辑 <<<
    checkpoint_dir = os.path.join(args.checkpoint_path, args.experiment)
    # 尝试搜集已有 ckpt
    ckpts = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    ckpts.sort()  # 根据名称排序，最后一个视作最新

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
        
    # dataloaders
    [test_loader] = make_dataloaders(args, splits=['test'])
    
    # train
    test_losses = test_one_epoch(args, test_loader, conv_merge, generator, discriminator)
    return test_losses
        
if __name__ == '__main__':
    args = get_args()
    test_losses = test(args)
    test_loss_dir = args.test_results_dir + 'test_loss.npy'
    np.save(test_loss_dir, test_losses)
    # test_loss = np.load(test_loss_dir, allow_pickle = True).item()