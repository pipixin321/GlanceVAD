import os
import yaml
import time
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboard_logger import Logger
from tqdm import tqdm
import datetime


from utils import set_seed, save_best_record, color
from options import parse_args
from dataset import Dataset
from models import URDMU
from engine import train, inference


def main():
    # >> Load args
    args = parse_args()
    set_seed(args.seed)
    args.device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    print('Using Device:'+color(args.device))
    # >> Model Initialization
    model = URDMU(args.feature_size, flag = "Train", a_nums = args.num_abn_mem, n_nums = args.num_nor_mem)
    model = model.to(args.device)
    if args.consume:
        ckpt = os.path.join(args.model_path, 'best_ckpt.pth')
        model.load_state_dict(torch.load(ckpt))
        print('>>> Checkpoint {} loaded.'.format(ckpt))
    # >> Train/Test
    if args.mode == 'train':
        # >> Datasets
        # >> normal videos for the training set
        train_regular_set = Dataset(args, is_normal=True, test_mode=False)
        # >> abnormal videos for the training set
        train_anomaly_set = Dataset(args, is_normal=False, test_mode=False)
        # >> videos for the testing set
        test_set = Dataset(args, test_mode=True)

        # >> DataLoader
        train_regular_loader = DataLoader(train_regular_set, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.workers, pin_memory=False, drop_last=True,)
        train_anomaly_loader = DataLoader(train_anomaly_set, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.workers, pin_memory=False, drop_last=True,)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                                num_workers=args.workers, pin_memory=False, )
        
        # >> Optimizer and Test info
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=args.wd)
        test_info = {'epoch': [], 'elapsed': [], 'now': [], 'train_loss': [], 'test_{metrics}'.format(metrics=args.metrics): [], 'ANO':[], 'FAR':[]}
        tb_logger = Logger(args.log_path)
        

        # >> Caculate initial result
        # score, a_score, far = 0, 0, 0
        score, a_score, far = inference(0, args, test_loader, model, tb_logger)
        sys_info = """
        {title}
            - dataset:\t {dataset}
            - description:\t {descr}
            - initial {metric} score: {score:.3f} %
            - initial ANO score: {a_score:.3f} %
            - initial FAR: {FAR:.3f} %
            - initial learning rate: {lr:.4f}
        """.format(
            title=color('Video Anomaly Detection', 'magenta'),
            dataset=color(args.dataset, 'white', attrs=['bold', 'underline']),
            descr=color(args.run_info),
            metric=args.metrics,
            score=score * 100,
            a_score = a_score * 100,
            FAR=far * 100,
            lr=args.lr
        )
        print(sys_info)

        # >> Define score_log file
        log_filepath = os.path.join(args.log_path, '{}.score'.format(args.dataset))
        if os.path.exists(log_filepath):
            os.remove(log_filepath)

        # >> Write log file
        with open(log_filepath, 'w') as f:
            f.write('\n{sep}\n{info}\n\n\n{env}\n{sep}\n'.format(sep = '*' * 10,
                info=yaml.dump(args, sort_keys=False, default_flow_style=False),
                env=''))
            f.write('\n{sep}\n{info}\n{sep}\n'.format(sep = '=' * 10, info=model))
            f.write('\n{}\n'.format(sys_info))

            title = '| {:^6s} | {:^8s} | {:^8s} | {:^8s} | {:^15s} | {:^30s} | {:^30s} |'.format(
                'Step', args.metrics, 'ANO', 'FAR', 'Training loss', 'Elapsed time', 'Now')
            f.write('+{sep}+\n'.format(sep = '-'*(len(title)-2)))
            f.write('{}\n'.format(title))
            f.write('{sep}\n'.format(sep = '-'*len(title)))

        # >> Training process start
        start_time = time.time()
        best_result = 0
        p_bar = tqdm(range(1, args.max_epoch + 1))
        for step in p_bar:
            if (step - 1) % len(train_regular_loader) == 0:
                loadern_iter = iter(train_regular_loader)
            if (step - 1) % len(train_anomaly_loader) == 0:
                loadera_iter = iter(train_anomaly_loader)
            
            # >> training
            loss, loss_dict = train(step, args, tb_logger, loadern_iter, loadera_iter, model, optimizer)
            
            # >> testing
            if step % args.evaluate_freq == 0 and step > args.evaluate_min_step:
                score, a_score, far = inference(step, args, test_loader, model, tb_logger, cache=False)
                metric = 'test_{metric}'.format(metric=args.metrics)
                test_info["epoch"].append(step)
                test_info[metric].append(score)
                test_info["ANO"].append(a_score)
                test_info["FAR"].append(far)
                test_info["train_loss"].append(loss)
                test_info["elapsed"].append(str(datetime.timedelta(seconds = time.time() - start_time)))
                test_info["now"].append(str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                if score > best_result:
                    best_result = score
                    torch.save(model.state_dict(), os.path.join(args.model_path, 'best.pth'))
                    save_best_record(test_info, log_filepath, metric)
                torch.save(model.state_dict(), os.path.join(args.model_path, 'ckpt.pth'))
                tb_logger.log_value('acc/best', best_result, step)
                print('[iter-{iter}]:{metric}:{score:.2f}'.format(iter=step, metric=args.metrics, score=score*100))

            # >> update progress bar
            p_bar.set_description('[{dataset}]:loss: {loss:.3f}, {metric}: {score:.2f}, best: {best:.2f}' \
                                  .format(dataset=color(args.dataset), loss = loss, metric=args.metrics,\
                                           score = score * 100, best=best_result*100))

        with open(log_filepath, 'a') as f:
            f.write('+{sep}+\n'.format(sep = '-'*(len(title)-2)))


    elif args.mode == 'test':
        checkpoint_path = os.path.join(args.model_path, 'best.pth')
        if os.path.exists(args.model):
            checkpoint_path = args.model
        model.load_state_dict(torch.load(checkpoint_path))
        print('>>> Checkpoint {} loaded.'.format(checkpoint_path))

        train_regular_set = Dataset(args, is_normal=True, test_mode=False)
        train_anomaly_set = Dataset(args, is_normal=False, test_mode=False)
        test_set = Dataset(args, test_mode=True)
        train_regular_loader = DataLoader(train_regular_set, batch_size=1, shuffle=False,
                                        num_workers=args.workers, pin_memory=False, drop_last=False)
        train_anomaly_loader = DataLoader(train_anomaly_set, batch_size=1, shuffle=False,
                                        num_workers=args.workers, pin_memory=False, drop_last=False)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=False)

        score, a_score, far = inference(0, args, test_loader, model, cal_FAR=True, cache=True)
        print(score, a_score, far)

        
if __name__ == '__main__':
    main()