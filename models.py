import torch
import torch.nn as nn
from torch.nn.modules.module import Module

from utils import norm
import math
from encoder_layer.translayer import Transformer


class Memory_Unit(Module):
    def __init__(self, nums, dim):
        super().__init__()
        self.dim = dim
        self.nums = nums
        self.memory_block = nn.Parameter(torch.empty(nums, dim))
        self.sig = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.memory_block.size(1))
        self.memory_block.data.uniform_(-stdv, stdv)
        if self.memory_block is not None:
            self.memory_block.data.uniform_(-stdv, stdv)
    
    def forward(self, data):  ####data size---> B,T,D       K,V size--->K,D
        attention = self.sig(torch.einsum('btd,kd->btk', data, self.memory_block) / (self.dim**0.5))   #### Att---> B,T,K
        temporal_att = torch.topk(attention, self.nums//16+1, dim = -1)[0].mean(-1)
        augment = torch.einsum('btk,kd->btd', attention, self.memory_block)                   #### feature_aug B,T,D
        return temporal_att, augment
    

class Temporal(Module):
    def __init__(self, input_size, out_size):
        super(Temporal, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3,
                    stride=1, padding=1),
            nn.ReLU(),
        )
    def forward(self, x):  
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = x.permute(0, 2, 1)
        return x

class ADCLS_head(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim,128), nn.ReLU(), nn.Linear(128,out_dim), nn.Sigmoid())
    def forward(self, x):
        return self.mlp(x)

class URDMU(Module):
    def __init__(self, input_size, flag, a_nums, n_nums):
        super().__init__()
        self.flag = flag
        self.a_nums = a_nums
        self.n_nums = n_nums
        self.dim_out = 512

        self.embedding = Temporal(input_size,self.dim_out)
        self.self_attn = Transformer(self.dim_out, 2, 4, 128, self.dim_out, dropout = 0.5)

        self.triplet = nn.TripletMarginLoss(margin=1)
        self.cls_head = ADCLS_head(2*self.dim_out, 1)
        self.Amemory = Memory_Unit(nums=a_nums, dim=self.dim_out)
        self.Nmemory = Memory_Unit(nums=n_nums, dim=self.dim_out)

        self.encoder_mu = nn.Sequential(nn.Linear(self.dim_out, self.dim_out))
        self.encoder_var = nn.Sequential(nn.Linear(self.dim_out, self.dim_out))
        self.relu = nn.ReLU()

        self.bce = nn.BCELoss()

    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def latent_loss(self, mu, var):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = 1))
        return kl_loss

    def forward(self, x):
        if len(x.size()) == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1

        x = self.embedding(x)                   #[B,T,D]
        x = self.self_attn(x)                   #[B,T,D]

        if self.flag == "Train":
            N_x = x[:b*n//2]                   #### Normal part
            A_x = x[b*n//2:]                   #### Abnormal part
            A_att, A_aug = self.Amemory(A_x)   ###bt,btd,   anomaly video --->>>>> Anomaly memeory  at least 1 [1,0,0,...,1]
            N_Aatt, N_Aaug = self.Nmemory(A_x) ###bt,btd,   anomaly video --->>>>> Normal memeory   at least 0 [0,1,1,...,1]

            A_Natt, A_Naug = self.Amemory(N_x) ###bt,btd,   normal video --->>>>> Anomaly memeory   all 0 [0,0,0,0,0,...,0]
            N_att, N_aug = self.Nmemory(N_x)   ###bt,btd,   normal video --->>>>> Normal memeory    all 1 [1,1,1,1,1,...,1]
    
            
            # origin
            _, A_index = torch.topk(A_att, t//16 + 1, dim=-1)
            negative_ax = torch.gather(A_x, 1, A_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,n,-1).mean(1)

            _, N_index = torch.topk(N_att, t//16 + 1, dim=-1)
            anchor_nx=torch.gather(N_x, 1, N_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,n,-1).mean(1)

            _, P_index = torch.topk(N_Aatt, t//16 + 1, dim=-1)
            positivte_nx = torch.gather(A_x, 1, P_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,n,-1).mean(1)

            triplet_margin_loss = self.triplet(norm(anchor_nx), norm(positivte_nx), norm(negative_ax))

            N_aug_mu = self.encoder_mu(N_aug)
            N_aug_var = self.encoder_var(N_aug)
            N_aug_new = self._reparameterize(N_aug_mu, N_aug_var)
            
            anchor_nx_new = torch.gather(N_aug_new, 1, N_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,n,-1).mean(1)

            A_aug_new = self.encoder_mu(A_aug)
            negative_ax_new = torch.gather(A_aug_new, 1, A_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,n,-1).mean(1)

            kl_loss = self.latent_loss(N_aug_mu, N_aug_var)
            A_Naug = self.encoder_mu(A_Naug)
            N_Aaug = self.encoder_mu(N_Aaug)

            distance = torch.relu(100 - torch.norm(negative_ax_new, p=2, dim=-1) + torch.norm(anchor_nx_new, p=2, dim=-1)).mean()
            x = torch.cat((x, (torch.cat([N_aug_new + A_Naug, A_aug_new + N_Aaug], dim=0))), dim=-1)
            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)

            return {
                    "video_scores": pre_att,
                    'triplet_margin': triplet_margin_loss,
                    'kl_loss': kl_loss, 
                    'distance': distance,
                    'A_att': A_att.reshape((b//2, n, -1)).mean(1),
                    "N_att": N_att.reshape((b//2, n, -1)).mean(1),
                    "A_Natt": A_Natt.reshape((b//2, n, -1)).mean(1),
                    "N_Aatt": N_Aatt.reshape((b//2, n, -1)).mean(1),
                }
        else:           
            _, A_aug = self.Amemory(x)
            _, N_aug = self.Nmemory(x)  

            A_aug = self.encoder_mu(A_aug)
            N_aug = self.encoder_mu(N_aug)

            x = torch.cat([x, A_aug + N_aug], dim=-1)
            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)

            return {"video_scores":pre_att}

    
    def criterion(self, args, outputs, regular_label, anomaly_label, point_label=None, anomaly_sample=None):
        loss = {}
        _label = torch.cat((regular_label, anomaly_label), 0).to(args.device)
        _label = _label.float()
        _label = torch.cat((regular_label, anomaly_label), 0).to(args.device)

        triplet = outputs["triplet_margin"]
        att = outputs['video_scores']
        A_att = outputs["A_att"]
        N_att = outputs["N_att"]
        A_Natt = outputs["A_Natt"]
        N_Aatt = outputs["N_Aatt"]
        kl_loss = outputs["kl_loss"]
        distance = outputs["distance"]
        b = _label.size(0)//2
        t = att.size(1)

        # mil loss  
        abn_att = att[args.batch_size:]
        anomaly = torch.topk(abn_att, t//16 + 1, dim=-1)[0].mean(-1)
        anomaly_loss = self.bce(anomaly, anomaly_label.to(args.device))
        
        nor_attn = att[:args.batch_size]
        normaly = torch.topk(nor_attn, t//16 + 1, dim=-1)[0].mean(-1)
        normaly_loss = self.bce(normaly, regular_label.to(args.device))

        # abnormal video attention
        panomaly = torch.topk(1 - N_Aatt, t//16 + 1, dim=-1)[0].mean(-1)
        panomaly_loss = self.bce(panomaly, torch.ones((b)).to(args.device))
        A_att_k = torch.topk(A_att, t//16 + 1, dim = -1)[0].mean(-1)
        A_loss = self.bce(A_att_k, torch.ones((b)).to(args.device))

        # normal video attention
        N_loss = self.bce(N_att, torch.ones_like((N_att)).to(args.device))    
        A_Nloss = self.bce(A_Natt, torch.zeros_like((A_Natt)).to(args.device))

        cost = 0 * anomaly_loss + 1.0 * normaly_loss + 0.1 * (A_loss + panomaly_loss + N_loss + A_Nloss) + 0.1 * triplet + 0.001 * kl_loss + 0.0001 * distance
        loss['att_loss'] = anomaly_loss
        loss['N_Aatt'] = panomaly_loss
        loss['A_loss'] = A_loss
        loss['N_loss'] = N_loss
        loss['A_Nloss'] = A_Nloss
        loss["triplet"] = triplet
        loss['kl_loss'] = kl_loss

        return cost, loss
