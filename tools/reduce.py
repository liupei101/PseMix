# Part of ReMix implementation.
# Copied from https://github.com/Jiawei-Yang/ReMix/blob/master/tools/clustering.py
import argparse
import os
import os.path as osp

import numpy as np
from tqdm import tqdm
import torch
import h5py

from sklearn.cluster import KMeans
#from clustering import Kmeans # clustering error when using fassi.clustering


def reduce(args, train_list):
    os.makedirs(args.dir_save, exist_ok=True)
    for feat_pth in tqdm(train_list):
        is_pt_file = False
        if feat_pth.endswith('.pt'):
            is_pt_file = True
        sid = osp.split(feat_pth)[-1].split('.pt')[0] if is_pt_file else osp.split(feat_pth)[-1].split('.h5')[0]
        path_save_centroids = osp.join(args.dir_save, "{}_bag_feats_proto_{}.npy".format(sid, args.num_prototypes))
        path_save_shifts = osp.join(args.dir_save, "{}_bag_shift_proto_{}.npy".format(sid, args.num_prototypes))
        
        if osp.exists(path_save_shifts) and osp.exists(path_save_centroids):
            print("having been generated, so skipping {}".format(sid))
            continue
        
        if is_pt_file:
            feats = torch.load(feat_pth, map_location=torch.device('cpu')).numpy()
        else:
            with h5py.File(feat_pth, 'r') as hf:
                feats = hf['features'][:]
        feats = np.ascontiguousarray(feats, dtype=np.float32)
        kmeans = KMeans(n_clusters=args.num_prototypes, init='random', random_state=66).fit(feats)
        assignments = kmeans.labels_.astype(np.int64)
        # compute the centroids and covariance matrix for each cluster
        centroids, covariances = [], []
        for i in range(args.num_prototypes):
            feat_iclus = feats[assignments == i]
            c = np.mean(feat_iclus, axis=0)
            centroids.append(c)
            cov = np.cov(feat_iclus.T)
            if np.isnan(cov).any():
                covariances.append(np.eye(feat_iclus.shape[1]) * 1e-6)
            else:
                covariances.append(cov)
            # print(feat_iclus.shape)

        centroids = np.array(centroids)
        covariances = np.array(covariances)
        # the semantic shift vectors are enough.
        semantic_shift_vectors = []
        for cov in covariances:
            # sample shift vector from zero-mean multivariate Gaussian distritbuion N(0, cov)
            try:
                semantic_shift = np.random.multivariate_normal(np.zeros(cov.shape[0]), cov, size=args.num_shift_vectors)
            except np.linalg.LinAlgError:
                semantic_shift = np.random.multivariate_normal(np.zeros(cov.shape[0]), np.eye(feat_iclus.shape[1]) * 1e-6, size=args.num_shift_vectors)
            semantic_shift_vectors.append(semantic_shift)
        
        semantic_shift_vectors = np.array(semantic_shift_vectors)
        # save the prototypes and shift_vectors per bag
        np.save(path_save_centroids, centroids)
        np.save(path_save_shifts, semantic_shift_vectors)

# python3 reduce.py --dir_read /NAS02/ExpData/tcga_brca/feat-x20-RN50-B-color_norm/pt_files 
# --dir_save /home/liup/data/tcga_brca/feat-x20-RN50-B-color_norm-remix --num_prototypes 8
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='base dictionary construction')
    #parser.add_argument('--dataset', type=str, default='Camelyon16')
    parser.add_argument('--dir_read', type=str)
    parser.add_argument('--dir_save', type=str)
    parser.add_argument('--num_prototypes', type=int, default=1)
    parser.add_argument('--num_shift_vectors', type=int, default=200)
    args = parser.parse_args()
    train_list = []
    for filename in os.listdir(args.dir_read):
        if filename.endswith('.pt') or filename.endswith('.h5'):
            train_list.append(osp.join(args.dir_read, filename))
    reduce(args, train_list)