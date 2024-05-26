# This file contains the code of generating prototypical clusters with DIEM methods.
#  
# [Note by PANTHER] Codebase adapted from DIEM, ICLR 2022.
# [Note by PseMix] Further adapted from PANTHER, CVPR 2024.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.io import load_pkl

def mog_eval(mog, data):
    """
    This evaluates the log-likelihood of mixture of Gaussians
    """    
    B, N, d = data.shape    
    pi, mu, Sigma = mog    
    if len(pi.shape) == 1:
        pi = pi.unsqueeze(0).repeat(B, 1)
        mu = mu.unsqueeze(0).repeat(B, 1, 1)
        Sigma = Sigma.unsqueeze(0).repeat(B, 1, 1)
    p = pi.shape[-1]
    
    jll = -0.5 * (d * np.log(2 * np.pi) + 
        Sigma.log().sum(-1).unsqueeze(1) +
        torch.bmm(data**2, 1. / Sigma.permute(0, 2, 1)) + 
        ((mu**2) / Sigma).sum(-1).unsqueeze(1) + 
        -2. * torch.bmm(data, (mu / Sigma).permute(0, 2, 1))
    ) + pi.log().unsqueeze(1) 
    
    mll = jll.logsumexp(-1) 
    cll = jll - mll.unsqueeze(-1)
    
    return jll, cll, mll


class DirNIWNet(nn.Module):
    """
    Conjugate prior for the Gaussian mixture model

    Args:
    - p (int): Number of prototypes
    - dim_feat (int): Embedding dimension
    - eps (float): initial covariance (similar function to sinkorn entropic regularizer)
    """
    
    def __init__(self, p, dim_feat=None, eps=0.1, num_iters=1, tau=0.001, path_proto=None, fix_proto=True, **kws):
        """
        self.m: prior mean (p x dim_feat)
        self.V_: prior covariance (diagonal) (p x dim_feat)
        """
        super(DirNIWNet, self).__init__()

        self.eps = eps
        self.tau = tau
        self.num_iters = num_iters

        if path_proto is not None:
            if path_proto.endswith('pkl'):
                weights = load_pkl(path_proto)['prototypes'].squeeze()
            elif path_proto.endswith('npy'):
                weights = np.load(path_proto)

            self.m = nn.Parameter(torch.from_numpy(weights), requires_grad=not fix_proto)
            assert weights.shape[0] == p, "The specified and loaded phenotype numbers do no match."
            dim_feat = weights.shape[1]
            print(f"[DIEM] Initialized prototypes ({p} x {dim_feat}) by loading from {path_proto}.")
        else:
            assert dim_feat is not None, "Please specify `dim_feat`."
            self.m = nn.Parameter(0.1 * torch.randn(p, dim_feat), requires_grad=not fix_proto)
            print(f"[DIEM] Initialized random prototypes ({p} x {dim_feat}).")

        self.V_ = nn.Parameter(np.log(np.exp(1) - 1) * torch.ones((p, dim_feat)), requires_grad=not fix_proto)

        self.p, self.dim_feat = p, dim_feat
    
    def forward(self):
        """
        Return prior mean and covariance
        """
        V = self.eps * F.softplus(self.V_)
        return self.m, V
    
    def mode(self, prior=None):
        if prior is None:
            m, V = self.forward()
        else:
            m, V = prior
        pi = torch.ones(self.p).to(m) / self.p
        mu = m
        Sigma = V
        return pi.float(), mu.float(), Sigma.float()
    
    def loglik(self, theta): 
        raise NotImplementedError
        
    def map_m_step(self, data, weight, prior=None):
        if prior is None:
            m, V = self.forward()
        else:
            m, V = prior
        
        wsum = weight.sum(1)
        wsum_reg = wsum + self.tau
        wxsum = torch.bmm(weight.permute(0, 2, 1), data)
        wxxsum = torch.bmm(weight.permute(0, 2, 1), data**2)
        pi = wsum_reg / wsum_reg.sum(1, keepdim=True)
        mu = (wxsum + m.unsqueeze(0) * self.tau) / wsum_reg.unsqueeze(-1)
        Sigma = (wxxsum + (V + m**2).unsqueeze(0) * self.tau) / wsum_reg.unsqueeze(-1) - mu**2

        return pi.float(), mu.float(), Sigma.float()
    
    def map_em(self, data, mask=None, prior=None):
        B, N, d = data.shape
        assert d == self.dim_feat, "Unmatched feature dimension."
        
        if mask is None:
            mask = torch.ones(B, N).to(data)

        # Need to set to the mode for initial starting point
        pi, mu, Sigma = self.mode(prior)
        pi = pi.unsqueeze(0).repeat(B, 1)
        mu = mu.unsqueeze(0).repeat(B, 1, 1)
        Sigma = Sigma.unsqueeze(0).repeat(B, 1, 1)
        
        for emiter in range(self.num_iters):
            _, qq, _ = mog_eval((pi, mu, Sigma), data)
            qq = qq.exp() * mask.unsqueeze(-1)

            pi, mu, Sigma = self.map_m_step(data, weight=qq, prior=prior)
            
        return pi, mu, Sigma, qq