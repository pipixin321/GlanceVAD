import os
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser("Official Pytorch Implementation of GLAD")
    #misc 
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--hyp', type=str, default='ucf_i3d')
    parser.add_argument('--seed', type=int, default=0, help='random seed (-1 for no manual seed)')
    parser.add_argument('--workers', type=int, default=0, help=' number of workers in dataloader')
    #path setting
    parser.add_argument('--ckpt_path', type=str, default="./ckpt")
    parser.add_argument('--model', type=str, default='', help='model path for inference')
    parser.add_argument('--consume', type=bool, default=False)
    #model
    parser.add_argument('--num_abn_mem', type=int, default=60, help=' number of abnormal memory')
    parser.add_argument('--num_nor_mem', type=int, default=60, help=' number of normal memory')
    parser.add_argument('--early_fusion', type=bool, default=False)
    args = parser.parse_args()

    with open('./configs/{}.yaml'.format(args.hyp)) as f:
        hyp_dict = yaml.load(f, Loader=yaml.FullLoader)
    for key, value in hyp_dict.items():
        setattr(args, key, value)

    return init_args(args)
    
def init_args(args):
    args.log_path = os.path.join(args.ckpt_path, args.dataset, 'log', args.run_info)
    args.model_path = os.path.join(args.ckpt_path, args.dataset, 'model', args.run_info)
    args.output_path = os.path.join(args.ckpt_path, args.dataset, 'output', args.run_info)
    for dir in [args.log_path, args.model_path, args.output_path]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    return args

    

    
