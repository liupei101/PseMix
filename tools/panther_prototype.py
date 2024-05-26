"""This file is adapted from PANTHER (https://github.com/mahmoodlab/PANTHER)

This will perform K-means clustering on the training data.

Good reference for clustering
https://github.com/facebookresearch/faiss/wiki/FAQ#questions-about-training
"""

from __future__ import print_function

import argparse
import os
from os.path import join as j_
import torch
from torch.utils.data import DataLoader

from .panther_prototype import WSIProtoDataset
from .panther_prototype import seed_torch, read_file_data_splitting
from .panther_prototype import read_csv_df, save_pkl, cluster


def build_datasets(csv_splits, label_csv, batch_size=1, num_workers=2, train_kwargs={}):
    dataset_splits = {}
    train_kwargs['sample_col'] = 'pathology_id'
    train_kwargs['slide_col']  = 'pathology_id'
    for k in csv_splits.keys(): # ['train']
        patient_ids = csv_splits[k]
        df = read_csv_df(patient_ids, label_csv, at_column='patient_id')
        dataset_kwargs = train_kwargs.copy()
        dataset = WSIProtoDataset(df, **dataset_kwargs)

        batch_size = 1
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        dataset_splits[k] = dataloader
        print(f'split: {k}, n: {len(dataset)}')

    return dataset_splits


def main(args):
    
    train_kwargs = dict(data_source=args.data_source,
                        use_h5=args.use_h5)
       
    seed_torch(args.seed)
    csv_splits = read_file_data_splitting(args.split_path)
    csv_splits = {k: csv_splits[k] for k in args.split_names}
    print('\nsuccessfully read splits for: ', list(csv_splits.keys()))

    dataset_splits = build_datasets(csv_splits,
                                    args.label_path,
                                    batch_size=1,
                                    num_workers=args.num_workers,
                                    train_kwargs=train_kwargs)

    print('\nInit Datasets...', end=' ')

    loader_train = dataset_splits['train']
    
    _, weights = cluster(loader_train,
                            n_proto=args.n_proto,
                            n_iter=args.n_iter,
                            n_init=args.n_init,
                            feature_dim=args.in_dim,
                            mode=args.mode,
                            n_proto_patches=args.n_proto_patches,
                            use_cuda=True if torch.cuda.is_available() else False)
    
    save_name = f"fold{args.split_fold}_train"
    save_fpath = j_(args.save_dir,
                    f"prototypes_{save_name}_c{args.n_proto}_{args.data_source[0].split('/')[-2]}_{args.mode}_num_{args.n_proto_patches:.1e}.pkl")

    save_pkl(save_fpath, {'prototypes': weights})


# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
# model / loss fn args ###
parser.add_argument('--n_proto', type=int, help='Number of prototypes')
parser.add_argument('--n_proto_patches', type=int, default=10000,
                    help='Number of patches per prototype to use. Total patches = n_proto * n_proto_patches')
parser.add_argument('--n_init', type=int, default=5,
                    help='Number of different KMeans initialization (for FAISS)')
parser.add_argument('--n_iter', type=int, default=50,
                    help='Number of iterations for Kmeans clustering')
parser.add_argument('--in_dim', type=int)
parser.add_argument('--mode', type=str, choices=['kmeans', 'faiss'], default='kmeans')

# dataset / split args ###
parser.add_argument('--data_source', type=str, default=None,
                    help='manually specify the data source')
parser.add_argument('--split_path', type=str, default=None,
                    help='manually specify the set of splits to use')
parser.add_argument('--split_fold', type=int, default=0,
                    help='fold index')
parser.add_argument('--label_path', type=str, default=None,
                    help='manually specify the label csv file to use')
parser.add_argument('--split_names', type=str, default='train',
                    help='delimited list for specifying names within each split')
parser.add_argument('--save_dir', type=str, default=None,
                    help='directory for saving initial prototypes.')
parser.add_argument('--use_h5', action='store_true', default=False)
parser.add_argument('--num_workers', type=int, default=8)

args = parser.parse_args()

if __name__ == "__main__":
    args.split_path = args.split_path.format(args.split_fold)
    print('data_source: ', args.data_source)
    print('label_path: ', args.label_path)
    print('split_path: ', args.split_path)
    print('split_names: ', args.split_names)

    args.data_source = [src for src in args.data_source.split(',')]
    args.split_names = [name for name in args.split_names.split(',')]
    results = main(args)