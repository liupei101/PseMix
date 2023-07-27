# PseMix

[[arXiv preprint](https://arxiv.org/abs/2306.16180)] Pseudo-Bag Mixup Augmentation for Multiple Instance Learning-Based Whole Slide Image Classification. 

## Key features

Applying PseMix (as a data augmentation method) to multiple instance learning networks, e.g., ABMIL, DSMIL, and TransMIL, could 
- **improve network performance** with minimal extra computational costs, without introducing any complicated techniques
- help the network obtain **better generalization and robustness** (against patch occlusion and noisy label learning)


## Applying PseMix to your training pipeline

Minimal code example (pseudo-code):
```python
# generate_pseudo_bags: function for dividing WSI bags into pseudo-bags
# ALPHA: the hyper-parameter of Beta distribution
# N: the number of pseudo-bags in each WSI bag
# PROB_MIXUP: random mixing parameter for determining the proportion of mixed bags. 
for (X, y) in loader: # load a minibatch 
    n_batch = X.shape[0] # with `n_batch` WSI bags (samples)

    # 1. dividing each bag into `N` pseudo-bags
    X = generate_pseudo_bags(X)

    new_idxs = torch.randperm(n_batch)
    # draw a mixing scale from Beta distribution
    lam = numpy.random.beta(ALPHA, ALPHA) 
    lam = min(lam, 1.0 - 1e-5) # avoid numerical overflow when transforming it into discrete ones
    lam_discrete = int(lam * (N + 1)) # transform into discrete values

    # 2. pseudo-bag-level Mixup generates samples (new_X, new_y)
    new_X, new_y = [], []
    for i in range(n_batch):
    	# randomly select pseudo-bags according to `lam_discrete`
        masked_bag_A = select_pseudo_bags(X[i], lam_discrete) # select `lam_discrete` pseudo-bags
        masked_bag_B = select_pseudo_bags(X[new_idxs[i]], N - lam_discrete) # select `n-lam_discrete` pseudo-bags

        # random-mixing mechanism for two purposes: more data diversity and efficient learning on mixed samples.
        if np.random.rand() <= PROB_MIXUP:
            mixed_bag = torch.cat([masked_bag_A, masked_bag_B], dim=0) # instance-axis concat
            new_X.append(mixed_bag)
            mix_ratio = lam_discrete / N
        else:
            masked_bag = masked_bag_A 
            new_X.append(masked_bag)
            mix_ratio = 1.0

        # target-level mixing
        new_y.append(mix_ratio * y[i] + (1 - mix_ratio) * y[new_idxs[i]]) 

    # 3. minibatch training
    minibatch_training(new_X, new_y)
```

NOTE that we actually use a weighted loss for target mixing, following [Mixup implementation](https://github.com/facebookresearch/mixup-cifar10). Details could be found at [the weighted loss](https://github.com/liupei101/PseMix/blob/main/model/clf_handler.py#L407).

Our implementation roughly follows the pseudo-codes above. More details could be found by the following codes:

- [generate_pseudo_bags](https://github.com/liupei101/PseMix/blob/main/utils/core.py#L146C13-L146C13).
- [pseudo-bag-level Mixup](https://github.com/liupei101/PseMix/blob/main/utils/core.py#L13C10-L13C10).
- [training with mixed labels](https://github.com/liupei101/PseMix/blob/main/model/clf_handler.py#L381).


## AUC performance

| Network | BRCA | NSCLC | RCC | Average                                  |
|---------|------|-------|-----|------------------------------------------|
| ABMIL   | 87.05 | 92.23 | 97.36 | 92.21 |
| ABMIL **w/ PseMix**   | 89.49 | 93.01 | 98.02 | 93.51 |
| DSMIL   | 87.73 | 92.99 | 97.65 | 92.79 |
| DSMIL **w/ PseMix**   | 89.65 | 93.92 | 97.89 | 93.82 |
| TransMIL   | 88.83 | 92.14 | 97.88 | 92.95 |
| TransMIL **w/ PseMix**   | 90.40 | 93.47 | 97.76 | 93.88 |

## Training with PseMix

Training curves (training and test AUC, exported from wandb) are listed as follows. Solid lines indicate training with PseMix, and dashed ones are those vanilla models without PseMix.   

| Model                                                                  | Wandb training curves                                  |
|------------------------------------------------------------------------|--------------------------------------------------------|
| [ABMIL](https://proceedings.mlr.press/v80/ilse18a.html)                |![](docs/wandb-abmil-train.png)          |
| [DSMIL](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_CVPR_2021_paper.pdf)         | ![](docs/wandb-dsmil-train.png)   |
| [TransMIL](https://openreview.net/forum?id=LKUfuWxajHc)     |![](docs/wandb-transmil-train.png)        |

## Citation

If you find our code helpful for your research, please using the following bibtex to cite this paper:
```txt
@misc{liu2023pseudobag,
      title={Pseudo-Bag Mixup Augmentation for Multiple Instance Learning-Based Whole Slide Image Classification}, 
      author={Pei Liu and Luping Ji and Xinyu Zhang and Feng Ye},
      year={2023},
      eprint={2306.16180},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
