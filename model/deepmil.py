import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer_utils import *


class DeepMIL(nn.Module):
    """
    Deep Multiple Instance Learning for Bag-level Task.

    Args:
        dim_in: input instance dimension.
        dim_emb: instance embedding dimension.
        num_cls: the number of class to predict.
        pooling: the type of MIL pooling, one of 'mean', 'max', and 'attention', default by attention pooling.
    """
    def __init__(self, dim_in=1024, dim_hid=1024, num_cls=2, use_feat_proj=True, drop_rate=0.5, 
        pooling='attention', pred_head='default'):
        super(DeepMIL, self).__init__()
        assert pooling in ['mean', 'max', 'attention', 'gated_attention']

        if use_feat_proj:
            self.feat_proj = Feat_Projecter(dim_in, dim_in)
        else:
            self.feat_proj = None
        
        if pooling == 'gated_attention':
            self.sigma = Gated_Attention_Pooling(dim_in, dim_hid, dropout=drop_rate)
        elif pooling == 'attention':
            self.sigma = Attention_Pooling(dim_in, dim_hid)
        else:
            self.sigma = pooling
        
        if pred_head == 'default':
            self.g = nn.Linear(dim_in, num_cls)
        else:
            self.g = nn.Sequential(
                nn.Linear(dim_in, dim_emb),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_rate),
                nn.Linear(dim_emb, num_cls)
            )

    def forward(self, X, ret_with_attn=False):
        """
        X: initial bag features, with shape of [b, K, d]
           where b = 1 for batch size, K is the instance size of this bag, and d is feature dimension.
        """
        assert X.shape[0] == 1
        if self.feat_proj is not None:
            # ensure that feat_proj has been adapted 
            # for the input with shape of [B, N, C]
            X = self.feat_proj(X)
        
        # global pooling function sigma
        # [B, N, C] -> [B, C]
        if self.sigma == 'mean':
            X_vec = torch.mean(X, dim=1)
        elif self.sigma == 'max':
            X_vec, _ = torch.max(X, dim=1)
        else:
            X_vec, raw_attn = self.sigma(X)

        # prediction head w/o norm
        # [B, C] -> [B, num_cls]
        logit = self.g(X_vec)

        if ret_with_attn:
            attn = raw_attn.detach()
            return logit, attn # [B, num_cls], [B, N]
        else:
            return logit

################################################
# DSMIL, li et al., CVPR, 2021.
################################################

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats, **kwargs):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
    
    def forward(self, x, **kwargs):
        device = x.device
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self, input_size, hid_size, output_class, dropout_v=0.0): # K, L, N
        super(BClassifier, self).__init__()
        self.q = nn.Linear(input_size, hid_size)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, hid_size)
        )
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=hid_size)
        
    def forward(self, feats, c, **kwargs): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B 


class DSMIL(nn.Module):
    def __init__(self, dim_in=1024, dim_emb=1024, num_cls=2, use_feat_proj=True, drop_rate=0.5):
        super(DSMIL, self).__init__()
        if use_feat_proj:
            self.feat_proj = Feat_Projecter(dim_in, dim_in)
        else:
            self.feat_proj = None
        self.i_classifier = FCLayer(in_size=dim_in, out_size=num_cls)
        self.b_classifier = BClassifier(dim_in, dim_emb, num_cls, dropout_v=drop_rate)
        
    def forward(self, X, **kwargs):
        assert X.shape[0] == 1
        if self.feat_proj is not None:
            # ensure that feat_proj has been adapted 
            # for the input with shape of [B, N, C]
            X = self.feat_proj(X)
        X = X.squeeze(0) # to [N, C] for input to i and b classifier
        feats, classes = self.i_classifier(X)
        prediction_bag, A, B = self.b_classifier(feats, classes) # bag = [1, C], A = [N, C]
        
        max_prediction, _ = torch.max(classes, 0)
        logits = 0.5 * (prediction_bag + max_prediction) # logits = [1, C]

        if 'ret_with_attn' in kwargs and kwargs['ret_with_attn']:
            # average over class heads
            attn = A.detach()
            attn = attn.mean(dim=1).unsqueeze(0)
            return logits, attn
        
        return logits


################################################
# TransMIL, Shao et al., NeurlPS, 2021.
################################################
import numpy as np
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            out, attn = self.attn(self.norm(x), return_attn=True)
            x = x + out
            return x, attn.detach()
        else:
            x = x + self.attn(self.norm(x))
            return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, dim_in, dim_hid, n_classes, **kwargs):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=dim_hid)
        self._fc1 = nn.Sequential(nn.Linear(dim_in, dim_hid), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_hid))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=dim_hid)
        self.layer2 = TransLayer(dim=dim_hid)
        self.norm = nn.LayerNorm(dim_hid)
        self._fc2 = nn.Linear(dim_hid, self.n_classes)

    def forward(self, X, **kwargs):

        assert X.shape[0] == 1 # [1, n, 1024], single bag
        
        h = self._fc1(X) # [B, n, dim_hid]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, dim_hid]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1) # token: 1 + H + add_length
        n1 = h.shape[1] # n1 = 1 + H + add_length

        #---->Translayer x1
        h = self.layer1(h) #[B, N, dim_hid]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, dim_hid]
        
        #---->Translayer x2
        if 'ret_with_attn' in kwargs and kwargs['ret_with_attn']:
            h, attn = self.layer2(h, return_attn=True) # [B, N, dim_hid]
            # attn shape = [1, n_heads, n2, n2], where n2 = padding + n1
            if add_length == 0:
                attn = attn[:, :, -n1, (-n1+1):]
            else:
                attn = attn[:, :, -n1, (-n1+1):(-n1+1+H)]
            attn = attn.mean(1).detach()
            assert attn.shape[1] == H
        else:
            h = self.layer2(h) # [B, N, dim_hid]
            attn = None

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc2(h) #[B, n_classes]

        if attn is not None:
            return logits, attn

        return logits
