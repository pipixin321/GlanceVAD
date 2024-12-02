import os
import json
import torch
import numpy as np
import pandas as pd
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, test_mode=False):
        self.args = args
        self.is_normal = is_normal
        self.test_mode = test_mode
        self.dataset = args.dataset
        self.backbone = args.backbone
        self.n_crop = args.n_crop
        self.feature_size = args.feature_size
        self.quantize_size = args.quantize_size
        
        self.subset = 'test' if test_mode else 'train'
        self.ann_file = args.ann_file
        self.feature_path = args.feature_path
        self.data_file = args.data_file_test if test_mode else args.data_file_train

        # >> load video list
        video_list = pd.read_csv(self.data_file)
        video_list = video_list['video-id'].values[:]
        self._prepare_data(video_list)

        # >> load glance anotations
        assert os.path.exists(args.glance_file)
        self.glance_annotations =  pd.read_csv(args.glance_file)



    def _prepare_data(self, video_list):
        if self.test_mode is False:
            if 'ucf-crime' in self.dataset: index = 810
            elif 'xd-violence' in self.dataset: index = 1905
            self.video_list = video_list[index:] if self.is_normal else video_list[:index]
        else:
            self.video_list = video_list
            self.ground_truths = self._prepare_frame_level_labels(video_list)

        # N crop Data Argumentation
        if 'i3d' in self.backbone:
            self.feat_crop_src = dict()
            crop_num = self.n_crop
            feat_list = []
            for vid in self.video_list:
                for i in range(crop_num):
                    if 'ucf-crime' in self.dataset and i==0:
                        feat_crop = vid
                    else:
                        feat_crop = vid + '__{}'.format(i)
                    feat_list.append(feat_crop)
                    self.feat_crop_src[feat_crop] = vid
            self.feat_list = feat_list
        else:
            self.feat_list = self.video_list

    def _prepare_frame_level_labels(self, video_list):
        with open(self.ann_file, 'r') as fin:
            db = json.load(fin)
        ground_truths = list()
        for video_id in video_list:
            labels = db[video_id]['labels']
            ground_truths.append(labels)
        ground_truths = np.concatenate(ground_truths)
        return ground_truths
    
    def get_label(self):
        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)
        return label
    
    def __getitem__(self, index):
        feat_id = self.feat_list[index]
        feat_path = os.path.join(self.feature_path, self.subset, '{}.npy'.format(feat_id))
        features = np.load(feat_path, allow_pickle=True).astype(np.float32)                 #[T_in,dim_in]

        label = self.get_label()
        vid_name = self.feat_crop_src[feat_id] if 'i3d' in self.backbone else self.video_list[index]

        if self.test_mode:
            sample = dict()
            sample['vid_name'] = vid_name
            sample['features'] = features
            return sample
        
        else:
            points = []
            points = self.glance_annotations.loc[self.glance_annotations['video-id'] == vid_name, 'glance'].values
            features, point_label = self.process_feat(features, points, self.quantize_size)
            sample = dict()
            sample['vid_name'] = vid_name
            sample['features'] = features
            sample['label'] = label
            sample['point_label'] = point_label
            return sample

    def process_feat(self, feat, points, length):
        #[T,D]
        if feat.ndim == 2: 
            num_seg = feat.shape[0]
            new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
            r = np.linspace(0, len(feat), length + 1, dtype = np.int)
            for i in range(length):
                if r[i] != r[i+1]:
                    new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
                else:
                    new_feat[i:i+1,:] = feat[r[i]:r[i]+1,:]
        #[N,T,D]
        else: 
            feat_all = []
            for n in range(feat.shape[0]):
                feat_n = feat[n]
                new_feat = np.zeros((length, feat_n.shape[1])).astype(np.float32)
                r = np.linspace(0, len(feat_n), length + 1, dtype = np.int)
                for i in range(length):
                    if r[i] != r[i+1]:
                        new_feat[i,:] = np.mean(feat_n[r[i]:r[i+1],:], 0)
                    else:
                        new_feat[i:i+1,:] = feat_n[r[i]:r[i]+1,:]
                feat_all.append(np.expand_dims(new_feat, axis=0))
            new_feat = np.concatenate(feat_all, axis=0)


        # points
        num_seg = feat.shape[-2]
        temp_point_anno = np.zeros([length], dtype=np.float32)
        if len(points)>0 and points[0]>0:
            for p in points:
                idx_seg = int((p/16)/num_seg*length)
                temp_point_anno[idx_seg] = 1


        return new_feat, temp_point_anno
    
    def shift_array(self, array, N):
        shifted_array = np.zeros_like(array)  
        indices = np.where(array == 1)[0]  

        for index in indices:
            if index > N and index < len(array)-1-N:
                direction = np.random.choice([-1, 1])  
                shifted_index = index + direction * N 
                shifted_index = np.clip(shifted_index, 0, len(array)-1)  
                shifted_array[shifted_index] = 1
            else:
                shifted_array[index] = 1

        return shifted_array     
    
    
    def __len__(self):
        return len(self.feat_list)
        
