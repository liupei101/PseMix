"""
Class for bag-style dataloader
"""
from typing import Union
import os.path as osp
import torch
import numpy as np
from torch.utils.data import Dataset
import torch_geometric

from utils.io import retrieve_from_table_clf
from utils.io import retrieve_from_table_surv
from utils.io import read_patch_data, read_patch_coord
from utils.func import sampling_data, random_mask_instance


class WSIPatchClf(Dataset):
    r"""A patch dataset class for classification tasks (slide-level in general).
    
    Args:
        patient_ids (list): A list of patients (string) to be included in dataset.
        patch_path (string): The root path of WSI patch features. 
        table_path (string): The path of table with dataset labels, which has to be included. 
        mode (string): 'patch', 'cluster', or 'graph'.
        read_format (string): The suffix name or format of the file storing patch feature.
    """
    def __init__(self, patient_ids: list, patch_path: str, table_path: str, label_path:Union[None,str]=None,
        read_format:str='pt', ratio_sampling:Union[None,float,int]=None, ratio_mask=None, mode='patch', **kws):
        super(WSIPatchClf, self).__init__()
        if ratio_sampling is not None:
            assert ratio_sampling > 0 and ratio_sampling < 1.0
            print("[dataset] Patient-level sampling with ratio_sampling = {}".format(ratio_sampling))
            patient_ids, pid_left = sampling_data(patient_ids, ratio_sampling)
            print("[dataset] Sampled {} patients, left {} patients".format(len(patient_ids), len(pid_left)))
        if ratio_mask is not None and ratio_mask > 1e-5:
            assert ratio_mask <= 1, 'The argument ratio_mask must be not greater than 1.'
            assert mode == 'patch', 'Only support a patch mode for instance masking.'
            self.ratio_mask = ratio_mask
            print("[dataset] Masking instances with ratio_mask = {}".format(ratio_mask))
        else:
            self.ratio_mask = None
        # used for randomly choosing slides to perform slide-level augmentation
        self.patch_path_random = kws['random_patch_path']
        self.patch_path_choice = ['feat-x20-RN50-B-color_norm-vflip', 'feat-x20-RN50-B-color_hed_light']

        self.read_path = patch_path
        self.label_path = label_path
        self.has_patch_label = (label_path is not None) and len(label_path) > 0
        
        info = ['sid', 'sid2pid', 'sid2label']
        self.sids, self.sid2pid, self.sid2label = retrieve_from_table_clf(
            patient_ids, table_path, ret=info, level='slide')
        self.uid = self.sids
        
        assert mode in ['patch', 'cluster', 'graph']
        self.mode = mode
        self.read_format = read_format
        self.kws = kws
        self.new_sid2label = None
        self.flag_use_corrupted_label = False
        if self.mode == 'cluster':
            assert 'cluster_path' in kws
        if self.mode == 'patch':
            assert 'coord_path' in kws
        if self.mode == 'graph':
            assert 'graph_path' in kws
        self.summary()

    def summary(self):
        print(f"Dataset WSIPatchClf for {self.mode}: avaiable WSIs count {self.__len__()}")
        if self.patch_path_random:
            print("[info] randomly load patch features.")
        if not self.has_patch_label:
            print("[note] the patch-level label is not avaiable, derived by slide label.")

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, index):
        sid   = self.sids[index]
        pid   = self.sid2pid[sid]
        label = self.sid2label[sid] if not self.flag_use_corrupted_label else self.new_sid2label[sid]
        # get patches from one slide
        index = torch.Tensor([index]).to(torch.int)
        label = torch.Tensor([label]).to(torch.long)

        if self.mode == 'patch':
            if self.patch_path_random:
                prob = np.random.rand()
                if prob <= 0.5:
                    cur_read_path = self.read_path
                else:
                    if prob <= 0.75:
                        cur_sub_path = self.patch_path_choice[0]
                    elif prob <= 1.00:
                        cur_sub_path = self.patch_path_choice[1]
                    else:
                        cur_sub_path = ""
                    temp_paths = self.read_path.split('/')
                    temp_paths[-2] = cur_sub_path
                    cur_read_path = "/".join(temp_paths)
                full_path = osp.join(cur_read_path, sid + '.' + self.read_format)
            else:
                full_path = osp.join(self.read_path, sid + '.' + self.read_format)
            feats = read_patch_data(full_path, dtype='torch').to(torch.float)
            # if masking patches
            if self.ratio_mask:
                feats = random_mask_instance(feats, self.ratio_mask, scale=1, mask_way='mask_zero')
            full_coord = osp.join(self.kws['coord_path'],  sid + '.h5')
            coors = read_patch_coord(full_coord, dtype='torch')
            if self.has_patch_label:
                path = osp.join(self.label_path, sid + '.npy')
                patch_label = read_patch_data(path, dtype='torch', key='label').to(torch.long)
            else:
                patch_label = label * torch.ones(feats.shape[0]).to(torch.long)
            assert patch_label.shape[0] == feats.shape[0]
            assert coors.shape[0] == feats.shape[0]
            return index, (feats, coors), (label, patch_label)
        else:
            pass
            return None

    def corrupt_labels(self, corrupt_prob):
        """
        The implementation roughly follows https://github.com/pluskid/fitting-random-labels/blob/master/cifar10_data.py
        """
        labels = np.array([self.sid2label[_sid] for _sid in self.sids])
        mask = np.random.rand(len(labels)) <= corrupt_prob
        labels[mask] = np.random.choice(labels.max() + 1, mask.sum()) # replaced with random labels
        # change the label
        cnt_wlabel = 0
        self.new_sid2label = dict()
        for i, _sid in enumerate(self.sids):
            if labels[i] != self.sid2label[_sid]:
                cnt_wlabel += 1
            self.new_sid2label[_sid] = int(labels[i])
        self.flag_use_corrupted_label = True
        print("[info] {:.2f}% corrupted labels with corrupt_prob = {}".format(cnt_wlabel / len(labels) * 100, corrupt_prob))

    def resume_labels(self):
        if self.flag_use_corrupted_label:
            self.flag_use_corrupted_label = False
            print("[info] the corrupted labels have been resumed.")


class WSIProtoPatchClf(Dataset):
    r"""A patch dataset class for classification tasks (slide-level in general).
    
    Args:
        patient_ids (list): A list of patients (string) to be included in dataset.
        patch_path (string): The root path of WSI patch features. 
        table_path (string): The path of table with dataset labels, which has to be included. 
        mode (string): 'patch', 'cluster', or 'graph'.
        read_format (string): The suffix name or format of the file storing patch feature.
    """
    def __init__(self, patient_ids: list, patch_path: str, table_path: str, **kws):
        super(WSIProtoPatchClf, self).__init__()
        self.read_path = patch_path
        
        info = ['sid', 'sid2pid', 'sid2label']
        self.sids, self.sid2pid, self.sid2label = retrieve_from_table_clf(
            patient_ids, table_path, ret=info, level='slide')

        self.uid = self.sids
        self.kws = kws
        self.summary()

    def summary(self):
        print(f"Dataset WSIProtoPatchClf: avaiable WSIs count {self.__len__()}")

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, index):
        sid   = self.sids[index]
        pid   = self.sid2pid[sid]
        label = self.sid2label[sid]
        # get patches from one slide
        index = torch.Tensor([index]).to(torch.int)
        label = torch.Tensor([label]).to(torch.long)

        # load bag prototypes, i.e., reduced-bag
        cur_path = osp.join(self.read_path, sid + '_bag_feats_proto_8.npy') # [n_cluster, dim_feat]
        feats = read_patch_data(cur_path, dtype='torch').to(torch.float)
        # load semantic shift vectors
        cur_path = osp.join(self.read_path, sid + '_bag_shift_proto_8.npy') # [n_cluster, n_shift_vector, dim_feat]
        semantic_shifts = read_patch_data(cur_path, dtype='torch').to(torch.float)
        # load patch labels (here only used as a placeholder)
        patch_label = label * torch.ones(feats.shape[0]).to(torch.long)
        assert feats.shape[0] == semantic_shifts.shape[0]
        return index, (feats, semantic_shifts), (label, patch_label)
    
    def resume_labels(self):
        pass


class WSIPatchSurv(Dataset):
    r"""A patch dataset class for classification tasks (patient-level in general).

    Args:
        patient_ids (list): A list of patients (string) to be included in dataset.
        patch_path (string): The root path of WSI patch features. 
        table_path (string): The path of table with dataset labels, which has to be included. 
        mode (string): 'patch', 'cluster', or 'graph'.
        read_format (string): The suffix name or format of the file storing patch feature.
    """
    def __init__(self, patient_ids: list, patch_path: str, table_path: str, mode:str,
        read_format:str='pt', time_format='ratio', time_bins=4, ratio_sampling:Union[None,float,int]=None, **kws):
        super(WSIPatchSurv, self).__init__()
        if ratio_sampling is not None:
            print("[dataset] Patient-level sampling with ratio_sampling = {}".format(ratio_sampling))
            patient_ids, pid_left = sampling_data(patient_ids, ratio_sampling)
            print("[dataset] Sampled {} patients, left {} patients".format(len(patient_ids), len(pid_left)))

        info = ['pid', 'pid2sid', 'pid2label']
        self.pids, self.pid2sid, self.pid2label = retrieve_from_table_surv(
            patient_ids, table_path, ret=info, time_format=time_format, time_bins=time_bins)
        self.uid = self.pids
        self.read_path = patch_path
        assert mode in ['patch', 'cluster', 'graph']
        self.mode = mode
        self.read_format = read_format
        self.kws = kws
        if self.mode == 'cluster':
            assert 'cluster_path' in kws
        if self.mode == 'patch':
            assert 'coord_path' in kws
        if self.mode == 'graph':
            assert 'graph_path' in kws
        self.summary()

    def summary(self):
        print(f"Dataset WSIPatchSurv for {self.mode}: avaiable patients count {self.__len__()}")

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, index):
        pid   = self.pids[index]
        sids  = self.pid2sid[pid]
        label = self.pid2label[pid]
        # get all data from one patient
        index = torch.Tensor([index]).to(torch.int)
        label = torch.Tensor(label).to(torch.float)

        if self.mode == 'patch':
            feats = []
            for sid in sids:
                full_path = osp.join(self.read_path, sid + '.' + self.read_format)
                feats.append(read_patch_data(full_path, dtype='torch'))

            feats = torch.cat(feats, dim=0).to(torch.float)
            return index, (feats, torch.Tensor([0])), label

        elif self.mode == 'cluster':
            cids = np.load(osp.join(self.kws['cluster_path'], '{}.npy'.format(pid)))
            feats = []
            for sid in sids:
                full_path = osp.join(self.read_path, sid + '.' + self.read_format)
                feats.append(read_patch_data(full_path, dtype='torch'))
            feats = torch.cat(feats, dim=0).to(torch.float)
            cids = torch.Tensor(cids)
            assert cids.shape[0] == feats.shape[0]
            return index, (feats, cids), label

        elif self.mode == 'graph':
            feats, graphs = [], []
            from .GraphBatchWSI import GraphBatch
            for sid in sids:
                full_path = osp.join(self.read_path, sid + '.' + self.read_format)
                feats.append(read_patch_data(full_path, dtype='torch'))
                full_graph = osp.join(self.kws['graph_path'],  sid + '.pt')
                graphs.append(torch.load(full_graph))
            feats = torch.cat(feats, dim=0).to(torch.float)
            graphs = GraphBatch.from_data_list(graphs, update_cat_dims={'edge_latent': 1})
            assert isinstance(graphs, torch_geometric.data.Batch)
            return index, (feats, graphs), label

        else:
            pass
            return None
