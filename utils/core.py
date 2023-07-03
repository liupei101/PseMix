"""
This file contains the core implementation of PseMix, as well as other data mixing strategies. 
"""
from typing import Optional
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def augment_bag(bags, labels, alpha=1.0, method='sebmix', **kws):
    n_batch = len(bags)

    if alpha > .0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    idxs = torch.randperm(n_batch)

    if n_batch == 1:
        return bags, labels, labels, 1.0, idxs

    if method == 'psebmix':
        n_pseb = kws['psebmix_n'] # pseudo-bag number
        ind_pseb = kws['psebmix_ind'] # indictor of pseudo-bag partitions in each bag
        prob_mixup = kws['psebmix_prob'] # prob to perform psebmix, otherwise performing simple pseudo-bag sampling
        mixup_lam_from = kws['mixup_lam_from']
        # recalculate the factor for mixup
        # to uniformly normalize lam to the set of [0, 1,..., n]
        # the proba of mixup is (n - 2) / n, 0.33, 0.5, 0.6, 0.66 when n = 3, 4, 5, 6         
        lam_temp = lam if lam != 1.0 else lam - 1e-5
        lam_temp = int(lam_temp * (n_pseb + 1))
        # mixup pseudo-bags: phenotype-based cutmix
        mixed_bags = []
        mixed_ratios = []
        for i in range(n_batch):
            bag_a  = fetch_pseudo_bags(bags[i], ind_pseb[i], n_pseb, lam_temp)
            bag_b  = fetch_pseudo_bags(bags[idxs[i]], ind_pseb[idxs[i]], n_pseb, n_pseb - lam_temp)
            if bag_a is None or len(bag_a) == 0:
                bag_ab = bag_b
                area_ratio = 0.0
                cont_ratio = lam_temp / n_pseb
            elif bag_b is None or len(bag_b) == 0:
                bag_ab = bag_a
                area_ratio = 1.0
                cont_ratio = lam_temp / n_pseb
            else:
                if np.random.rand() <= prob_mixup:
                    bag_ab = torch.cat([bag_a, bag_b], dim=0) # instance-axis concat
                    area_ratio = len(bag_a) / len(bag_ab)
                    cont_ratio = lam_temp / n_pseb
                else:
                    bag_ab = bag_a # instance-axis concat
                    area_ratio = 1.0
                    cont_ratio = 1.0
            mixed_bags.append(bag_ab.unsqueeze(0))
            
            if mixup_lam_from is not None:
                if mixup_lam_from == 'area':
                    temp_mix_ratio = area_ratio
                elif mixup_lam_from == 'content':
                    temp_mix_ratio = cont_ratio
                else:
                    temp_mix_ratio = lam
            else:
                temp_mix_ratio = lam
            mixed_ratios.append(temp_mix_ratio)
        
        lam_new = mixed_ratios
        labels_a, labels_b = labels, [labels[idxs[i]] for i in range(n_batch)]

    elif method == 'insmix':
        # mixup instances: naive cutmix
        mixed_bags = []
        mixed_ratios = []
        for i in range(n_batch):
            bag_a  = fetch_instances(bags[i], lam)
            bag_b  = fetch_instances(bags[idxs[i]], 1 - lam)
            if bag_a is None:
                bag_ab = bag_b
                area_ratio = 0.0
            elif bag_b is None:
                bag_ab = bag_a
                area_ratio = 1.0
            else:
                bag_ab = torch.cat([bag_a, bag_b], dim=0) # instance-axis concat
                area_ratio = len(bag_a) / len(bag_ab)
            mixed_bags.append(bag_ab.unsqueeze(0))
            mixed_ratios.append(area_ratio)

        labels_a, labels_b = labels, [labels[idxs[i]] for i in range(n_batch)]
        lam_new = mixed_ratios
    else:
        # NO any mixup
        mixed_bags = bags
        labels_a, labels_b = labels, labels
        lam_new = 1.0

    return mixed_bags, labels_a, labels_b, lam_new, idxs

def fetch_instances(X, r=1.0):
    if len(X.shape) > 2:
        X = X.squeeze(0)
    
    N = len(X)
    
    if isinstance(r, float):
        num = int(N * r)
    elif isinstance(r, int):
        num = r
    else:
        pass

    if num == 0:
        return None
    idxs = torch.randperm(N)[:num].to(X.device)
    X_fetched = X[idxs]

    return X_fetched

def fetch_pseudo_bags(X, ind_X, n:int, n_parts:int):
    """
    X: bag features, usually with a shape of [N, d]
    ind_X: pseudo-bag indicator, usually with a shape of [N, ]
    n: pseudo-bag number, int
    n_parts: the pseudo-bag number to fetch, int
    """
    if len(X.shape) > 2:
        X = X.squeeze(0)
    assert n_parts <= n, 'the pseudo-bag number to fetch is invalid.'
    if n_parts == 0:
        return None

    ind_fetched = torch.randperm(n)[:n_parts]
    X_fetched = torch.cat([X[ind_X == ind] for ind in ind_fetched], dim=0)

    return X_fetched


#############################################################################################
#                     Different ways of dividing pseudo-bags.
#############################################################################################
class PseudoBag(object):
    """General class for generating pseudo-bags.

    Args
        n (int): pseudo-bag number to divide.
        l (int): phenotype number to consider.
    """
    def __init__(self, n: int, l: int, proto_method: str = 'mean', pheno_cut_method: str = 'quantile', 
        iter_fine_tuning: int = 0):
        assert proto_method in ['max', 'mean']
        assert pheno_cut_method in ['uniform', 'quantile']
        self.n = n
        self.l = l
        self.proto_method = proto_method
        self.iter_fine_tuning = iter_fine_tuning
        self.pheno_cut_method = pheno_cut_method
        self.norm_X = None
        print("Prototype-based pseudo-bag dividing: n = {}, l = {}".format(n, l))

    def divide(self, bag: Tensor, ptype: Optional[Tensor] = None, ret_pseudo_bag: bool = False):
        """
        bag (Tensor): bag of multiple instances with shape of [N, d].
        ptype (Optional[None,Tensor]): bag prototype with shape of [*, d].
        """
        if len(bag.shape) > 2:
            bag = bag.squeeze(0)
        # process prototype
        if ptype is not None:
            self.ptype = ptype
        elif self.proto_method == 'mean':
            self.ptype = torch.mean(bag, dim=0)
        elif self.proto_method == 'max':
            self.ptype, _ = torch.max(bag, dim=0)
        else:
            self.ptype = None
        self.norm_ptype = F.normalize(self.ptype, p=2, dim=-1)
        
        # calculate distances
        dis, limits = self.measure_distance_from_prototype(bag) # [N, ], tuple
        
        # stratified random dividing
        # 1. cut into phenotype clusters
        if self.pheno_cut_method == 'uniform':
            step = (limits[1] - limits[0]) / self.l
            label_phe = ((dis - limits[0]) / step).long().clamp(0, self.l - 1)
        elif self.pheno_cut_method == 'quantile':
            data = dis.cpu().numpy()
            bins = np.quantile(data, [i/self.l for i in range(self.l+1)])
            bins[0], bins[-1] = bins[0] - 1e-5, bins[-1] + 1e-5
            label_phe = np.digitize(data, bins) - 1
            label_phe = torch.LongTensor(label_phe).to(dis.device)
        else:
            pass
        # if fine-tuning phenotype clusters
        if self.iter_fine_tuning > 0:
            label_phe = self.fine_tune_cluster(bag, label_phe)

        # 2. sample from each phenotype cluster
        label_psebag = torch.zeros_like(label_phe)
        for c in range(self.l):
            c_size = (label_phe == c).sum().item()
            label_psebag[label_phe == c] = self.uniform_assign(c_size, self.n).to(label_psebag.device)

        if ret_pseudo_bag:
            pseudo_bags = [bag[label_psebag == i] for i in range(self.n)]
            return label_psebag, pseudo_bags
        
        return label_psebag

    def measure_distance_from_prototype(self, X, metric='cosine'):
        assert X.shape[-1] == self.norm_ptype.shape[-1]
        if metric == 'cosine':
            norm_X = F.normalize(X, p=2, dim=-1)
            dis = torch.mm(norm_X, self.norm_ptype.view(-1, 1)).squeeze() # [N, 1] -> [N, ]
            limits = (-1, 1)
            self.norm_X = norm_X
        else:
            raise NotImplementedError("cannot recognize {}".format(metric))
        return dis, limits

    def fine_tune_cluster(self, X, ind_cluster, metric='cosine'):
        """
        X (Tensor): bag features with a shape of [N, d]
        ind_cluster (Tensor): indicator of cluster assignment with a shape of [N, ]
        """
        assert self.iter_fine_tuning > 0
        for i in range(self.iter_fine_tuning):
            centroids, _ = self.mean_by_label(X, ind_cluster)
            if metric == 'cosine':
                norm_centroids = F.normalize(centroids, p=2, dim=-1) # [l, d]
                dis = torch.mm(self.norm_X, norm_centroids.T) # [N, d] x [d, l] -> [N, l]
            else:
                raise NotImplementedError("cannot recognize {}".format(metric))
            _, new_ind = torch.max(dis, dim=1)
            ind_cluster = new_ind
        return new_ind

    @staticmethod
    def uniform_assign(N, num_label):
        L = torch.randperm(N) % num_label
        rlab = torch.randperm(num_label)
        res = rlab[L]
        return res

    @staticmethod
    def mean_by_label(samples, labels):
        '''
        select mean(samples), count() from samples group by labels, ordered by labels ASC.
        '''
        weight = torch.zeros(labels.max()+1, samples.shape[0]).to(samples.device) # #class, N
        weight[labels, torch.arange(samples.shape[0])] = 1
        label_count = weight.sum(dim=1)
        weight = F.normalize(weight, p=1, dim=1) # l1 normalization
        mean = torch.mm(weight, samples) # #class, F
        index = torch.arange(mean.shape[0])[label_count > 0]
        return mean[index], label_count[index]


class PseudoBag_Kmeans(object):
    """Class for generating pseudo-bags via Kmeans.

    Args
        n (int): pseudo-bag number to divide.
        l (int): phenotype number to consider, or cluster number.
    """
    def __init__(self, n: int, l: int):
        self.n = n
        self.l = l
        print("Kmeans-based pseudo-bag dividing: n = {}, l = {}".format(n, l))

    def divide(self, bag: Tensor, ret_pseudo_bag: bool = False):
        """
        bag (Tensor): bag of multiple instances with shape of [N, d].
        """
        if len(bag.shape) > 2:
            bag = bag.squeeze(0)

        # stratified random dividing
        # 1. cut into phenotype clusters
        feats = np.ascontiguousarray(bag.cpu().numpy(), dtype=np.float32)
        kmeans = KMeans(n_clusters=self.l, random_state=0).fit(feats)
        label_phe = kmeans.labels_.astype(np.int64)
        label_phe = torch.LongTensor(label_phe).to(bag.device)

        # 2. sample from each phenotype cluster
        label_psebag = torch.zeros_like(label_phe)
        for c in range(self.l):
            c_size = (label_phe == c).sum().item()
            label_psebag[label_phe == c] = self.uniform_assign(c_size, self.n).to(label_psebag.device)

        if ret_pseudo_bag:
            pseudo_bags = [bag[label_psebag == i] for i in range(self.n)]
            return label_psebag, pseudo_bags
        
        return label_psebag

    @staticmethod
    def uniform_assign(N, num_label):
        L = torch.randperm(N) % num_label
        rlab = torch.randperm(num_label)
        res = rlab[L]
        return res


class PseudoBag_Random(object):
    """Class for generating pseudo-bags via random sampling.

    Args
        n (int): pseudo-bag number to divide.
    """
    def __init__(self, n: int):
        self.n = n
        print("Kmeans-based pseudo-bag dividing: n = {}".format(n))

    def divide(self, bag: Tensor, ret_pseudo_bag: bool = False):
        """
        bag (Tensor): bag of multiple instances with shape of [N, d].
        """
        if len(bag.shape) > 2:
            bag = bag.squeeze(0)

        # sample from all instances
        c_size = bag.shape[0]
        label_psebag = self.uniform_assign(c_size, self.n).to(bag.device)

        if ret_pseudo_bag:
            pseudo_bags = [bag[label_psebag == i] for i in range(self.n)]
            return label_psebag, pseudo_bags
        
        return label_psebag

    @staticmethod
    def uniform_assign(N, num_label):
        L = torch.randperm(N) % num_label
        rlab = torch.randperm(num_label)
        res = rlab[L]
        return res

#############################################################################################
# - ReMix: A General and Efficient Framework for MIL-based WSI Classification, MICCAI 2022.
# - Following the official implementation, https://github.com/Jiawei-Yang/ReMix
#############################################################################################
DIM_PATCH_FEAT = 1024
def mix_aug(src_feats, tgt_feats, mode='replace', rate=0.3, strength=0.5, shift=None):
    assert mode in ['replace', 'append', 'interpolate', 'cov', 'joint']
    auged_feats = [_ for _ in src_feats.reshape(-1, DIM_PATCH_FEAT)]
    tgt_feats = tgt_feats.reshape(-1, DIM_PATCH_FEAT)
    closest_idxs = np.argmin(cdist(src_feats.reshape(-1, DIM_PATCH_FEAT), tgt_feats), axis=1)
    if mode != 'joint':
        for ix in range(len(src_feats)):
            if np.random.rand() <= rate:
                if mode == 'replace':
                    auged_feats[ix] = tgt_feats[closest_idxs[ix]]
                elif mode == 'append':
                    auged_feats.append(tgt_feats[closest_idxs[ix]])
                elif mode == 'interpolate':
                    generated = (1 - strength) * auged_feats[ix] + strength * tgt_feats[closest_idxs[ix]]
                    auged_feats.append(generated)
                elif mode == 'cov':
                    generated = auged_feats[ix][np.newaxis, :] + strength * shift[closest_idxs[ix]][np.random.choice(200, 1)]
                    auged_feats.append(generated.flatten())
                else:
                    raise NotImplementedError
    else:
        for ix in range(len(src_feats)):
            if np.random.rand() <= rate:
                # replace
                auged_feats[ix] = tgt_feats[closest_idxs[ix]]
            if np.random.rand() <= rate:
                # append
                auged_feats.append(tgt_feats[closest_idxs[ix]])
            if np.random.rand() <= rate:
                # interpolate
                generated = (1 - strength) * auged_feats[ix] + strength * tgt_feats[closest_idxs[ix]]
                auged_feats.append(generated)
            if np.random.rand() <= rate:
                # covary
                generated = auged_feats[ix][np.newaxis, :] + strength * shift[closest_idxs[ix]][np.random.choice(200, 1)]
                auged_feats.append(generated.flatten())
    return np.array(auged_feats)


def remix_bag(bags, labels, mode='joint', semantic_shifts=None):
    N = len(bags)
    list_labels = [l.item() for l in labels]
    ret_bags = []
    for idx_bag, cur_bag in enumerate(bags):
        # randomly select one bag from the same class
        candidate_idxs = [_i for _i in range(N) if list_labels[_i] == list_labels[idx_bag]]
        selected_id = np.random.choice(candidate_idxs)
        # lambda parameter
        strength = np.random.uniform(0, 1)
        new_bag = mix_aug(
            cur_bag.cpu().numpy(), bags[selected_id].cpu().numpy(),
            shift=semantic_shifts[selected_id] if mode in ['joint', 'cov'] else None,
            strength=strength, mode=mode
        )
        new_bag = torch.Tensor([new_bag]).reshape(-1, DIM_PATCH_FEAT).unsqueeze(0).cuda()
        ret_bags.append(new_bag)
    
    return ret_bags, labels

#########################################################
#              ProtoDiv's Implementation
#########################################################
def generate_pseudo_bags(bags, n_pseb, ind_pseb):
    """
    n_pseb: pseudo-bag number
    ind_pseb: indictor of pseudo-bag partitions in each bag
    """
    n_batch = len(bags)
    # sampling one pseudo-bag from each bag
    sampled_bags = []
    for i in range(n_batch):
        new_bag  = fetch_pseudo_bags(bags[i], ind_pseb[i], n_pseb, 1)
        sampled_bags.append(new_bag.unsqueeze(0))

    return sampled_bags


######################################################################
#               Mixup's Implementation
# The instance number of two bags are aligned by cropping the larger
# - RankMix (CVPR, 2023):
#     Given instance scores, sizes are aligned by Ranking + Cropping 
#     the lowest instances from the larger.
# - Vallina Mixup:
#     Without instance scores, sizes are aligned by random cropping 
#     the larger.
######################################################################
def rank_instances(bag, ins_score):
    bag = bag.squeeze(0)
    ins_score = ins_score.squeeze()
    assert len(bag) == len(ins_score), "Sizes are not aligned."

    _, idx_rank = torch.sort(ins_score, descending=True)
    bag = bag[idx_rank] # place the instances with largest scores at first

    return bag

def mixup_bag(bags, labels, scores=None, alpha=1.0):
    n_batch = len(bags)

    if alpha > .0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    idxs = torch.randperm(n_batch)

    if n_batch == 1:
        return bags, labels, labels, 1.0, idxs

    # mixup two bags
    mixed_bags = []
    mixed_ratios = []
    for i in range(n_batch):
        bag_a = bags[i].squeeze(0)
        bag_b = bags[idxs[i]].squeeze(0)
        num_a, num_b = bag_a.shape[0], bag_b.shape[0]
        num_min = min(num_a, num_b)

        if scores is not None:
            bag_a = rank_instances(bag_a, scores[i])
            bag_b = rank_instances(bag_b, scores[idxs[i]])
            bag_ab = lam * bag_a[:num_min] + (1.0 - lam) * bag_b[:num_min]
        else:
            if num_a <= num_b:
                bag_b = fetch_instances(bag_b, num_min)
            else:
                bag_a = fetch_instances(bag_a, num_min)
            bag_a = bag_a[torch.randperm(num_min), :]
            bag_b = bag_b[torch.randperm(num_min), :]
            bag_ab = lam * bag_a + (1.0 - lam) * bag_b

        mixed_bags.append(bag_ab.unsqueeze(0))
        mixed_ratios.append(lam)

    labels_a, labels_b = labels, [labels[idxs[i]] for i in range(n_batch)]
    lam_new = mixed_ratios

    return mixed_bags, labels_a, labels_b, lam_new, idxs
