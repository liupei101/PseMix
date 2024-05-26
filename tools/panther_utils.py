"""PANTHER util functions

Code is borrowed from PANTHER (https://github.com/mahmoodlab/PANTHER)
"""
from __future__ import print_function, division
import sys
import os
from os.path import join as j_
import os.path as osp
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import h5py
from tqdm import tqdm
import time
from sklearn.cluster import KMeans


####################################################
#               Util functions
####################################################
def seed_torch(seed=7):
    import random
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def df_sdir(dataroot: str, cols=['fpath', 'fname', 'slide_id']):
    """
    Returns pd.DataFrame of the file paths and fnames of contents in dataroot.

    Args:
        dataroot (str): path to files.

    Returns:
        (pandas.Dataframe): pd.DataFrame of the file paths and fnames of contents in dataroot (make default cols: ['fpath', 'fname_ext', 'fname_noext']?)
    """
    return pd.DataFrame([(e.path, e.name, os.path.splitext(e.name)[0]) for e in sdir_(dataroot)], columns=cols)

# TODO: Fix doc + also make function for ldir_diff
def series_diff(s1, s2, dtype='O'):
    r"""
    Returns set difference of two pd.Series.
    """
    return pd.Series(list(set(s1).difference(set(s2))), dtype=dtype)

def infer_columns_for_splitting(available_columns):
    columns_with_key_words = ('train', 'test', 'val')

    ret_columns = []
    for c in columns_with_key_words:
        target_c = None
        for a_c in available_columns:
            if c in a_c:
                target_c = a_c
        ret_columns.append(target_c)

    train_col, test_col, val_col = ret_columns
    if test_col is None:
        test_col = val_col
        val_col  = None

    assert train_col is not None, "The column corresponding to `train` is not found."
    assert test_col is not None, "The column corresponding to `test` is not found."
    
    return train_col, test_col, val_col

def read_file_data_splitting(path: str):
    _, ext = osp.splitext(path)

    data_split = dict()
    if ext == '.npz':
        data_npz = np.load(path)
        column_train, column_test, column_validation = infer_columns_for_splitting([_ for _ in data_npz.keys()])
        
        pids_train = [str(s) for s in data_npz[column_train]]
        data_split['train'] = pids_train
        print(f"[data split] there are {len(pids_train)} cases for train.")
        
        pids_test = [str(s) for s in data_npz[column_test]]
        data_split['test'] = pids_test
        print(f"[data split] there are {len(pids_test)} cases for test.")

        if column_validation is not None:
            pids_val = [str(s) for s in data_npz[column_validation]]
            data_split['validation'] = pids_val
            print(f"[data split] there are {len(pids_val)} cases for validation.")

    elif ext == '.csv':
        data_csv = pd.read_csv(path)
        column_train, column_test, column_validation = infer_columns_for_splitting([_ for _ in data_csv.columns])

        pids_train = [str(s) for s in data_csv[column_train].dropna()]
        data_split['train'] = pids_train
        print(f"[data split] there are {len(pids_train)} cases for train.")

        pids_test = [str(s) for s in data_csv[column_test].dropna()]
        data_split['test'] = pids_test
        print(f"[data split] there are {len(pids_test)} cases for test.")

        if column_validation is not None:
            pids_val = [str(s) for s in data_csv[column_validation].dropna()]
            data_split['validation'] = pids_val
            print(f"[data split] there are {len(pids_val)} cases for validation.")

    return data_split

def read_csv_df(patient_ids, label_path, at_column='patient_id'):
    df = pd.read_csv(label_path, dtype={at_column: str})
    ret_idx = []
    for i in df.index:
        if df.loc[i, at_column] in patient_ids:
            ret_idx.append(i)
    # assert len(ret_idx) == len(patient_ids)
    print(f"read {len(ret_idx)} rows.")
    return df.loc[ret_idx, :]

def save_pkl(filename, save_object):
    with open(filename,'wb') as f:
        pickle.dump(save_object, f)


####################################################
#               Clustering function
####################################################
def cluster(data_loader, n_proto, n_iter, n_init=5, feature_dim=1024, n_proto_patches=50000, mode='kmeans', use_cuda=False):
    """
    K-Means clustering on embedding space

    For further details on FAISS,
    https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization
    """
    n_patches = 0

    n_total = n_proto * n_proto_patches

    # Sample equal number of patch features from each WSI
    try:
        n_patches_per_batch = (n_total + len(data_loader) - 1) // len(data_loader)
    except:
        n_patches_per_batch = 1000

    print(f"Sampling maximum of {n_proto * n_proto_patches} patches: {n_patches_per_batch} each from {len(data_loader)}")

    patches = torch.Tensor(n_total, feature_dim)

    for batch in tqdm(data_loader):
        if n_patches >= n_total:
            continue

        data = batch['img'] # (n_batch, n_instances, instance_dim)

        with torch.no_grad():
            out = data.reshape(-1, data.shape[-1])[:n_patches_per_batch]  # Remove batch dim

        size = out.size(0)
        if n_patches + size > n_total:
            size = n_total - n_patches
            out = out[:size]
        patches[n_patches: n_patches + size] = out
        n_patches += size

    print(f"\nTotal of {n_patches} patches aggregated")

    s = time.time()
    if mode == 'kmeans':
        print("\nUsing Kmeans for clustering...")
        print(f"\n\tNum of clusters {n_proto}, num of iter {n_iter}")
        kmeans = KMeans(n_clusters=n_proto, max_iter=n_iter)
        kmeans.fit(patches[:n_patches].cpu())
        weight = kmeans.cluster_centers_[np.newaxis, ...]

    elif mode == 'faiss':
        assert use_cuda, f"FAISS requires access to GPU. Please enable use_cuda"
        try:
            import faiss
        except ImportError:
            print("FAISS not installed. Please use KMeans option!")
            raise
        
        numOfGPUs = torch.cuda.device_count()
        print(f"\nUsing Faiss Kmeans for clustering with {numOfGPUs} GPUs...")
        print(f"\tNum of clusters {n_proto}, num of iter {n_iter}")

        kmeans = faiss.Kmeans(patches.shape[1], 
                              n_proto, 
                              niter=n_iter, 
                              nredo=n_init,
                              verbose=True, 
                              max_points_per_centroid=n_proto_patches,
                              gpu=numOfGPUs)
        kmeans.train(patches)
        weight = kmeans.centroids[np.newaxis, ...]

    else:
        raise NotImplementedError(f"Clustering not implemented for {mode}!")

    e = time.time()
    print(f"\nClustering took {e-s} seconds!")

    return n_patches, weight


####################################################
#               Dataset class
####################################################
class WSIProtoDataset(Dataset):
    """WSI Custer Dataset."""

    def __init__(self,
                 df,
                 data_source,
                 sample_col='slide_id',
                 slide_col='slide_id',
                 use_h5=False):
        """
        Args:
        """
        self.use_h5 = use_h5
        self.ext = '.h5' if use_h5 else '.pt'

        self.data_source = []
        for src in data_source:
            if self.ext == '.h5':
                assert 'h5' in os.path.basename(src), f"{src} has to end with the directory feats_h5"
            elif self.ext == '.pt':
                assert 'pt' in os.path.basename(src), f"{src} has to end with the directory feats_pt"
            self.data_source.append(src)

        self.data_df = df
        assert 'Unnamed: 0' not in df.columns
        self.sample_col = sample_col
        self.slide_col = slide_col
        self.data_df[sample_col] = self.data_df[sample_col].astype(str)
        self.data_df[slide_col] = self.data_df[slide_col].astype(str)
        self.X = None
        self.y = None

        self.idx2sample_df = pd.DataFrame({'sample_id': self.data_df[sample_col].astype(str).unique()})
        self.set_feat_paths_in_df()
        self.data_df.index = self.data_df[sample_col].astype(str)
        self.data_df.index.name = 'sample_id'

    def __len__(self):
        return len(self.idx2sample_df)

    def set_feat_paths_in_df(self):
        """
        Sets the feature path (for each slide id) in self.data_df. At the same time, checks that all slides 
        specified in the split (or slides for the cases specified in the split) exist within data source.
        """
        self.feats_df = pd.concat([df_sdir(feats_dir, cols=['fpath', 'fname', self.slide_col]) for feats_dir in self.data_source]).drop(['fname'], axis=1).reset_index(drop=True)
        missing_feats_in_split = series_diff(self.data_df[self.slide_col], self.feats_df[self.slide_col])

        ### Assertion to make sure that there are not any missing slides that were specified in your split csv file
        try:
            assert len(missing_feats_in_split) == 0
        except:
            print(f"Missing Features in Split:\n{missing_feats_in_split}")
            sys.exit()

        ### Assertion to make sure that all slide ids to feature paths have a one-to-one mapping (no duplicated features).
        try:
            self.data_df = self.data_df.merge(self.feats_df, how='left', on=self.slide_col, validate='1:1')
            assert self.feats_df[self.slide_col].duplicated().sum() == 0
        except:
            print("Features duplicated in data source(s). List of duplicated features (and their paths):")
            print(self.feats_df[self.feats_df[self.slide_col].duplicated()].to_string())
            sys.exit()

        self.data_df = self.data_df[list(self.data_df.columns[-1:]) + list(self.data_df.columns[:-1])]

    def get_sample_id(self, idx):
        return self.idx2sample_df.loc[idx]['sample_id']

    def get_feat_paths(self, idx):
        feat_paths = self.data_df.loc[self.get_sample_id(idx), 'fpath']
        if isinstance(feat_paths, str):
            feat_paths = [feat_paths]
        return feat_paths

    def __getitem__(self, idx):
        feat_paths = self.get_feat_paths(idx)

        # Read features (and coordinates, Optional) from pt/h5 file
        all_features = []
        all_coords = []
        for feat_path in feat_paths:
            if self.use_h5:
                with h5py.File(feat_path, 'r') as f:
                    features = f['features'][:]
                    coords = f['coords'][:]
                all_coords.append(coords)
            else:
                features = torch.load(feat_path)

            if len(features.shape) > 2:
                assert features.shape[0] == 1, f'{features.shape} is not compatible! It has to be (1, numOffeats, feat_dim) or (numOffeats, feat_dim)'
                features = np.squeeze(features, axis=0)

            all_features.append(features)
        all_features = torch.from_numpy(np.concatenate(all_features, axis=0))
        if len(all_coords) > 0:
            all_coords = np.concatenate(all_coords, axis=0)

        out = {'img': all_features,
               'coords': all_coords}

        return out
