import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.collect_env import get_pretty_env_info
import numpy as np
from termcolor import colored


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def color(text, txt_color='green', attrs=['bold']):
    return colored(text, txt_color, attrs=attrs)

def norm(data):
    l2 = torch.norm(data, p = 2, dim = -1, keepdim = True)
    return torch.div(data, l2)

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True



def gaussian_kernel_mining(args, score, point_label):
    abn_snippet = point_label.clone().detach()
    abn_ratio = args.alpha
    for b in range(point_label.shape[0]):
        abn_idx = torch.nonzero(point_label[b]).squeeze(1)
        if len(abn_idx) == 0: continue

        # most left
        if abn_idx[0] > 0:
            '''pseudo abnormal'''
            for j in range(abn_idx[0]-1, -1, -1):
                abn_thresh = abn_ratio * score[b, abn_idx[0]]
                if score[b, j] >= abn_thresh:
                    abn_snippet[b, j] = 1
                else: break

        # most right
        if abn_idx[-1] < (point_label.shape[1]-1):
            '''pseudo abnormal'''
            for j in range(abn_idx[-1]+1, point_label.shape[1]-1):
                abn_thresh = abn_ratio * score[b, abn_idx[-1]]
                if score[b, j] >= abn_thresh:
                    abn_snippet[b, j] = 1
                else: break
            
        # between
        for i in range(len(abn_idx)-1):
            if abn_idx[i+1] - abn_idx[i] <= 1:
                continue
            '''pseudo abnormal'''
            for j in range(abn_idx[i]+1, abn_idx[i+1]):
                abn_thresh = abn_ratio * score[b, abn_idx[i]]
                if score[b, j] >= abn_thresh:
                    abn_snippet[b, j] = 1
                else: break
            for j in range(abn_idx[i+1]-1, abn_idx[i], -1):
                abn_thresh = abn_ratio * score[b, abn_idx[i+1]]
                if score[b, j] >= abn_thresh:
                    abn_snippet[b, j] = 1
                else: break
    return abn_snippet

def temporal_gaussian_splatting(point_label, distribution='normal', params=None):
    """
    Calculate weights splatted by different gaussian kernels.
    Args:
    - point_label: Input point labels
    - distribution: Distribution type, options are 'normal', 'cauchy', 'laplace', 'exponential', 'lognormal'
    - params: Distribution parameters, a dictionary
    """

    distribution_weight = torch.zeros_like(point_label)
    N = distribution_weight.shape[1]

    for b in range(point_label.shape[0]):
        abn_idx = torch.nonzero(point_label[b]).squeeze(1)
        if len(abn_idx) == 0:
            continue

        temp_weight = torch.zeros([len(abn_idx), N])

        for i, point in enumerate(abn_idx):
            i_arr = torch.arange(N, dtype=torch.float32)
            h_i = 2 * (i_arr - 1) / (N - 1) - 1
            h_p = 2 * (point - 1) / (N - 1) - 1

            if distribution == 'normal':
                weight = torch.exp(-(h_i - h_p) ** 2 / (2 * params['sigma']**2)) / (params['sigma'] * (2 * np.pi)**0.5)
            elif distribution == 'cauchy':
                weight = 1 / (1 + ((h_i - h_p) / params['gamma'])**2) / (np.pi * params['gamma'])
            elif distribution == 'laplace':
                weight = 0.5 * torch.exp(-torch.abs(h_i - h_p) / params['b']) / params['b']
            else:
                raise ValueError("Unsupported distribution type")

            weight = (weight - torch.min(weight)) / (torch.max(weight) - torch.min(weight))
            temp_weight[i, :] = weight

        temp_weight = torch.max(temp_weight, dim=0)[0]
        temp_weight = (temp_weight - torch.min(temp_weight)) / (torch.max(temp_weight) - torch.min(temp_weight))
        distribution_weight[b, :] = temp_weight

    return distribution_weight


class GlanceLoss(nn.Module):
    def __init__(self, args): 
        super().__init__()
        
        self.dvc = args.device
        self.abn_ratio = args.alpha
        self.sigma = torch.tensor(args.sigma)
        self.min_mining_step = args.min_mining_step

        self.bce = nn.BCELoss(reduction='none')  # 4flexibility
        
    def gaussian_kernel_mining(self, score, point_label, seqlen=None):
        abn_snippet = point_label.clone().detach()
        bs, max_len = point_label.shape
        
        for b in range(bs):
            valid_len = max_len
            if seqlen is not None:
                valid_len = min(seqlen[b].item(), max_len)
            
            abn_idx = torch.nonzero(point_label[b]).squeeze(1)
            if len(abn_idx.shape) == 0 and abn_idx.numel() > 0:  # Handle single index case
                abn_idx = abn_idx.unsqueeze(0)
            if abn_idx.numel() == 0: continue
            #if len(abn_idx) == 0: continue   
            
            # Process most left boundary
            if abn_idx[0] > 0:
                for j in range(abn_idx[0]-1, -1, -1):
                    abn_thresh = self.abn_ratio * score[b, abn_idx[0]]
                    if score[b, j] >= abn_thresh:
                        abn_snippet[b, j] = 1
                    else: break

            # Process most right boundary (respect sequence length)
            if abn_idx[-1] < (valid_len-1):
                for j in range(abn_idx[-1]+1, valid_len):
                    abn_thresh = self.abn_ratio * score[b, abn_idx[-1]]
                    if score[b, j] >= abn_thresh:
                        abn_snippet[b, j] = 1
                    else: break
                
            # Process between abnormal points
            for i in range(len(abn_idx)-1):
                if abn_idx[i+1] - abn_idx[i] <= 1:
                    continue
                    
                # Forward pass
                for j in range(abn_idx[i]+1, abn_idx[i+1]):
                    abn_thresh = self.abn_ratio * score[b, abn_idx[i]]
                    if score[b, j] >= abn_thresh:
                        abn_snippet[b, j] = 1
                    else: break
                        
                # Backward pass
                for j in range(abn_idx[i+1]-1, abn_idx[i], -1):
                    abnloss_fx_thresh = self.abn_ratio * score[b, abn_idx[i+1]]
                    if score[b, j] >= abn_thresh:
                        abn_snippet[b, j] = 1
                    else: break
                        
        return abn_snippet

    def temporal_gaussian_splatting(self, point_label, distribution='normal', params=None, seqlen=None):
        point_label = point_label.clone().detach().cpu()
        bs, max_len = point_label.shape
        distribution_weight = torch.zeros_like(point_label)
        
        for b in range(bs):
            N = max_len
            if seqlen is not None:
                N = min(seqlen[b].item(), max_len)
                if N == 0: continue
            
            #point_label = point_label[b, :N]
            abn_idx = torch.nonzero(point_label[b]).squeeze(1)
            if len(abn_idx.shape) == 0 and abn_idx.numel() > 0:  # Handle single index case
                abn_idx = abn_idx.unsqueeze(0)
            if abn_idx.numel() == 0: continue

            temp_weight = torch.zeros([len(abn_idx), N])
            for i, point in enumerate(abn_idx):
                # Normalize to [-1, 1] range within valid sequence
                i_arr = torch.arange(N, dtype=torch.float32)
                if N > 1:
                    h_i = 2 * (i_arr - 1) / (N - 1) - 1
                    h_p = 2 * (point - 1) / (N - 1) - 1
                else:
                    h_i = h_p = torch.zeros(1)

                # Calculate weights based on distribution
                if distribution == 'normal':
                    weight = torch.exp(-(h_i - h_p) ** 2 / (2 * params['sigma']**2)) / (params['sigma'] * (2 * torch.pi)**0.5)
                elif distribution == 'cauchy':
                    weight = 1 / (1 + ((h_i - h_p) / params['gamma'])**2) / (torch.pi * params['gamma'])
                elif distribution == 'laplace':
                    weight = 0.5 * torch.exp(-torch.abs(h_i - h_p) / params['b']) / params['b']
                else:
                    raise ValueError(f"Unsupported distribution: {distribution}")

                if torch.min(weight) != torch.max(weight):
                    weight = (weight - torch.min(weight)) / (torch.max(weight) - torch.min(weight))
                else:
                    weight = torch.ones_like(weight)
                    
                temp_weight[i, :] = weight

            temp_weight = torch.max(temp_weight, dim=0)[0]
            
            if torch.min(temp_weight) != torch.max(temp_weight):
                temp_weight = (temp_weight - torch.min(temp_weight)) / (torch.max(temp_weight) - torch.min(temp_weight))
            
            distribution_weight[b, :N] = temp_weight

        return distribution_weight

    def forward(self, abn_scores, abn_pnt_lbl, step, abn_seqlen = None):
        
        # Gaussian Mining
        abn_kernel = self.gaussian_kernel_mining(
            abn_scores.detach().cpu(), 
            abn_pnt_lbl
        )
        # Temporal Gaussian Splatting
        if step < self.min_mining_step:
            rendered_score = self.temporal_gaussian_splatting(
                abn_pnt_lbl, 
                'normal', 
                params={'sigma': self.sigma}
            )
        else:
            rendered_score = self.temporal_gaussian_splatting(
                abn_kernel, 
                'normal', 
                params={'sigma': self.sigma}
            )
        loss = self.bce(abn_scores, rendered_score.to(self.dvc)).mean()
        return loss


def save_best_record(test_info, file_path, metric):
    with open(file_path, 'a') as f:
        f.write('| {:^6s} | {:^8s} | {:^8s} | {:^8s} | {:^15s} | {:^30s} | {:^30s} | \n'.format(
            str(test_info["epoch"][-1]),
            '{:.3f}'.format(test_info[metric][-1] * 100.),
            '{:.3f}'.format(test_info['ANO'][-1] * 100),
            '{:.3f}'.format(test_info['FAR'][-1] * 100),
            '{:.3f}'.format(test_info['train_loss'][-1]),
            test_info['elapsed'][-1],
            test_info['now'][-1],
        ))



