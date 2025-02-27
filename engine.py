import torch
import torch.nn as nn
import numpy as np
import os, json
from sklearn.metrics import auc, roc_curve, precision_recall_curve, confusion_matrix, average_precision_score
from utils import gaussian_kernel_mining, temporal_gaussian_splatting


def cal_false_alarm(gt, preds, threshold=0.5):
    preds = list(preds.cpu().detach().numpy())
    gt = list(gt.cpu().detach().numpy())
    preds = np.repeat(preds, 16)
    preds[preds < threshold] = 0
    preds[preds >= threshold] = 1
    tn, fp, fn, tp = confusion_matrix(gt, preds, labels=[0, 1]).ravel()
    far = fp / (fp + tn)
    return far

    
def train(step, args, tb_logger, regular_loader, anomaly_loader, model, loss_fx, optimizer, scheduler=None):
    with torch.set_grad_enabled(True):
        model.train()
        model.flag = 'Train'
        regular_sample = next(regular_loader)
        anomaly_sample= next(anomaly_loader)
        regular_video, regular_label = regular_sample['features'], regular_sample['label']
        anomaly_video, anomaly_label = anomaly_sample['features'], anomaly_sample['label']
        point_label = anomaly_sample['point_label']
        
        video = torch.cat((regular_video, anomaly_video), 0).to(args.device)            #[B*2, N, T, dim_in]
        outputs = model(video)
        cost, loss_dict = model.criterion(args, outputs, regular_label, anomaly_label, point_label, anomaly_sample)

        # >> GlanceVAD
        abn_scores = outputs['video_scores'][args.batch_size:]
        #abn_seqlen = torch.sum(torch.max(torch.abs(anomaly_video), dim=2)[0] > 0, 1) #B/2
        loss_abn = loss_fx(abn_scores, point_lael, step) #,abn_seqlen
        
        # Guassian Mining
        #loss_abn = 0
        #bce_criterion = nn.BCELoss()
        #abn_score = outputs['video_scores'][args.batch_size:]
        #abn_kernel = gaussian_kernel_mining(args, abn_scores.detach().cpu(), point_label)
        ## Temporal Gaussian Splatting
        #sigma = args.sigma
        #if step < args.min_mining_step:
        #    rendered_score = temporal_gaussian_splatting(point_label, 'normal', params={'sigma':sigma})
        #else:
        #    rendered_score = temporal_gaussian_splatting(abn_kernel, 'normal', params={'sigma':sigma})
        #rendered_score = rendered_score.to(args.device)
        #loss_abn = bce_criterion(abn_scores, rendered_score.to(args.device)).mean()
        
        loss_dict['loss_abn'] = loss_abn
        cost = cost + loss_abn

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
            tb_logger.log_value('loss/lr', optimizer.param_groups[0]['lr'], step)

        for item in loss_dict.keys():
            tb_logger.log_value('loss/'+item, loss_dict[item], step)
        return cost, loss_dict

        
def inference(step, args, test_loader, model, tb_logger=None, 
            cal_FAR=True, cache=False):
    with torch.no_grad():
        model.eval()
        model.flag = "Test"
        
        #------------------------------Test set-------------------------------#
        # prepare global prediction and label
        gt = test_loader.dataset.ground_truths
        all_preds = torch.zeros(0).to(args.device)
        abnormal_preds = torch.zeros(0).to(args.device)
        abnormal_labels = torch.zeros(0).to(args.device)
        normal_preds = torch.zeros(0).to(args.device)
        normal_labels = torch.zeros(0).to(args.device)
        gt_tmp = torch.tensor(gt.copy()).to(args.device)
        video_list = test_loader.dataset.video_list
        if cache:
            result_dict = dict()

        temp_pred = torch.zeros(0).to(args.device)
        for i, sample in enumerate(test_loader):
            #>> prepare inputs
            video = sample['features']
            video = video.to(args.device)                                           #[B, T, C]
            seq_len = torch.sum(torch.max(torch.abs(video), dim=-1)[0] > 0, -1)     #[B]
            #>> parser outputs
            outputs = model(video)
            pred_crop = outputs['video_scores']            #[B,T]

            #>> fusion of N-crop prediction
            temp_pred = torch.cat((temp_pred, pred_crop))
            if (i+1) % args.n_crop == 0:
                # split the testing set to 2 part: Normal & Abnormal
                labels = gt_tmp[: seq_len[0] * 16]
                gt_tmp = gt_tmp[seq_len[0] * 16:]
                pred = torch.mean(temp_pred, 0)
                temp_pred = torch.zeros(0).to(args.device)
                all_preds = torch.cat((all_preds, pred))

                if torch.sum(labels) == 0:
                    normal_labels = torch.cat((normal_labels, labels))
                    normal_preds = torch.cat((normal_preds, pred))
                else:
                    abnormal_labels = torch.cat((abnormal_labels, labels))
                    abnormal_preds = torch.cat((abnormal_preds, pred))
                
                # write output 
                if cache:
                    video_id = video_list[(i+1) // args.n_crop - 1]
                    result_dict[video_id] = dict()
                    result_dict[video_id]['score'] = [float(s) for s in list(pred.cpu().detach().numpy())]
        
        #>> get frame prediction
        all_preds = list(all_preds.cpu().detach().numpy())
        all_preds = np.repeat(all_preds, 16)                #[n_seg] -> [n_frames] (1 segment = 16 frames)
        abnormal_preds = list(abnormal_preds.cpu().detach().numpy())
        abnormal_preds = np.repeat(abnormal_preds, 16)
        abnormal_labels = list(abnormal_labels.cpu().detach().numpy())

        
        #>> evaluation
        if args.metrics == 'AUC':
            fpr, tpr, th = roc_curve(list(gt), all_preds)
            rec_auc = auc(fpr, tpr)
            score = rec_auc
            a_score = 0
            if cal_FAR:
                a_fpr, a_tpr, _ = roc_curve(abnormal_labels, abnormal_preds)
                a_rec_auc = auc(a_fpr, a_tpr)
                a_score = a_rec_auc

        elif args.metrics == 'AP':
            precision, recall, th = precision_recall_curve(list(gt), all_preds)
            pr_auc = auc(recall, precision)
            score = pr_auc
            a_score = 0
            if cal_FAR:
                a_precision, a_recall, _ = precision_recall_curve(abnormal_labels, abnormal_preds)
                a_pr_auc = auc(a_recall, a_precision)
                a_score = a_pr_auc

        far = 0
        if cal_FAR:
            far = cal_false_alarm(normal_labels, normal_preds)
        if args.mode == 'test':
            print('Score:{}, Ano:{}, FAR:{}'.format(score, a_score, far))
        if tb_logger is not None:
            tb_logger.log_value('acc/current', score, step)
            tb_logger.log_value('acc/ANO', a_score, step)
            tb_logger.log_value('acc/FAR', far, step)

        #>> save output for further analysis 
        if cache:
            save_dir =  os.path.join(args.output_path, 'snippet_pred.json')
            with open(save_dir, 'w') as f:
                json.dump(result_dict, f)
        return score, a_score, far
