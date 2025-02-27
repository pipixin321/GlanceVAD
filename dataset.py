import os
import json
import torch
import numpy as np
import pandas as pd
import torch.utils.data as data


class SelOrPad:
    def __init__(self, seg_len, random_extract=True):
        self.seg_len = seg_len
        self.random_extract = random_extract

    def process(self, feat):
        if len(feat) > self.seg_len:
            return self._extract(feat)
        return self._pad(feat)

    def _extract(self, feat):
        if self.random_extract:
            return self._random_extract(feat)
        return self._uniform_extract(feat)

    def _random_extract(self, feat):
        start = np.random.randint(len(feat) - self.seg_len)
        return feat[start:start + self.seg_len]

    def _uniform_extract(self, feat):
        indices = np.linspace(0, len(feat)-1, self.seg_len, dtype=np.uint16)
        return feat[indices]

    def _pad(self, feat):
        if feat.shape[0] < self.seg_len:
            pad_size = self.seg_len - feat.shape[0]
            return np.pad(feat, ((0, pad_size), (0, 0)), mode='constant')
        return feat

    def __call__(self, feat):
        return self.process(feat)

class Interpolate:
    def __init__(self, seg_len):
        self.seg_len = seg_len

    def pool(self, feat):
        if feat.ndim == 2: return self._pool_single(feat), None
        elif feat.ndim == 3: return self._pool_batch(feat), None
        else: raise ValueError("Input features must be 2D (T, D) or 3D (N, T, D)")

    def _pool_single(self, feat):
        new_feat = np.zeros((self.seg_len, feat.shape[1]), dtype=np.float32)
        r = np.linspace(0, len(feat), self.seg_len + 1, dtype=int)
        for i in range(self.seg_len):
            if r[i] != r[i+1]:
                new_feat[i] = np.mean(feat[r[i]:r[i+1]], axis=0)
            else:
                new_feat[i] = feat[r[i]]
        return new_feat

    def _pool_batch(self, feat):
        pooled_features = []
        for n in range(feat.shape[0]):
            pooled = self._pool_single(feat[n])
            pooled_features.append(pooled[np.newaxis, ...])
        return np.concatenate(pooled_features, axis=0)

    def __call__(self, feat):
        return self.pool(feat)


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
        
        if args.seg == 'itp': self.process_feat = Interpolate(self.quantize_size)
        elif args.seg == 'seq': 
            raise NotImplementedError
            self.process_feat = SelOrPad(self.quantize_size)
        
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
            features = self.process_feat(features, points, self.quantize_size)
            point_label == self.get_point_label(points, features.shape[-2])
            sample = dict()
            sample['vid_name'] = vid_name
            sample['features'] = features
            sample['label'] = label
            sample['point_label'] = point_label
            return sample
            
    def get_point_label(points, feat_len):
        ## if selorpad and feat_len > self.quantize_size 
        #   use only points present in chosen feat segments/idxs
        temp_point_anno = np.zeros([self.quantize_size], dtype=np.float32)
        if len(points)>0 and points[0]>0:
            for p in points:
                idx_seg = int((p/16)/feat_len*self.quantize_size)
                temp_point_anno[idx_seg] = 1
        return temp_point_anno
    
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
        
