import argparse

def get_args():
    parser = argparse.ArgumentParser('NowcastNet', add_help=False)
    # data params
    parser.add_argument('--data_path', type=str, default='/data/NJU-CPOL')
    parser.add_argument('--input_data_type', type=str, default='float32')
    
    # model params
    parser.add_argument('--input_length', type=int, default=10)
    parser.add_argument('--pred_length', type=int, default=10)
    parser.add_argument('--total_length', type=int, default=10)
    parser.add_argument('--img_height', type=int, default=256)
    parser.add_argument('--img_width', type=int, default=256)
    parser.add_argument('--input_channel', type=int, default=1)
    parser.add_argument('--evo_ic', type=int, default=20)
    parser.add_argument('--ic_feature', type=int, default=320)
    parser.add_argument('--gen_oc', type=int, default=10)
    
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=6.0)
    parser.add_argument('--gamma', type=float, default=20.0)
    parser.add_argument('--pool_loss_k', type=int, default=2)
    parser.add_argument('--ngf', type=int, default=32)   
    # optimizer params
    parser.add_argument('--lr_evo', type=float, default=1e-3)
    parser.add_argument('--lr_gen', type=float, default=3e-5)
    parser.add_argument('--lr_disc', type=float, default=3e-5)
    parser.add_argument('--lr_decrease_epoch', type=int, default=6)
    parser.add_argument('--lr_evo_de', type=float, default=10e-5)
    parser.add_argument('--lr_gen_de', type=float, default=1e-5)
    parser.add_argument('--lr_stop_epoch', type=int, default=18)
    # train params
    parser.add_argument('--num_epochs', type=int, default=100)
    
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/')
    parser.add_argument('--gen_frm_dir', type=str, default='./output/')
    parser.add_argument('--test_results_dir', type=str, default='./test_output/')
    
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--experiment', type=str, default='test0910')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--cpu_worker', type=int, default=0)
    
    return parser.parse_args()
