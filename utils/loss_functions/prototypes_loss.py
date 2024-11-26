from cmath import exp
import torch
from torch import nn
import numpy as np
import math

class Intra_Prototypes_Loss(nn.Module):
    def __init__(self, n_classes=2, alpha= 6):
        super(Intra_Prototypes_Loss, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha # default = 0.5
    def forward(self, x):
        B, C, K, D = x.shape
        x = x.reshape(B*C, K, D)
        x_c = torch.mean(x, dim=1) #[BC D]
        dist_map = x - x_c[:, None, :] #[BC K D]
        dist_map = dist_map**2
        dist_map = torch.sqrt(torch.sum(dist_map, dim=-1)) # [BC K]
        dist_map = dist_map.reshape(B, C, K)
        loss = torch.mean(dist_map, dim=-1) # [B C]
        loss = torch.mean(loss, dim=-1) # [B]
        loss = torch.mean(loss)-self.alpha
        return max(loss, loss*0)


class Cross_Prototypes_Loss(nn.Module):
    def __init__(self, n_classes=2, alpha= 100):
        super(Cross_Prototypes_Loss, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha # default = 50
    def forward(self, x):
        B, C, K, D = x.shape
        x = x.reshape(B*C, K, D)
        x_c = torch.mean(x, dim=1) #[BC D]
        x_c = x_c.reshape(B, C, D)
        x_c_c = torch.mean(x_c, dim=1) # [B D]

        dist_map = x_c - x_c_c[:, None, :] #[B C D]
        dist_map = dist_map**2
        dist_map = torch.sqrt(torch.sum(dist_map, dim=-1)) # [B C]
        loss = torch.mean(dist_map, dim=-1) # [B]
        loss = -loss + self.alpha
        loss = torch.mean(loss, dim=-1)
        #return max(loss, torch.tensor(0).type_as(loss))
        return max(loss, loss*0)



class CrossIntra_Prototypes_Loss(nn.Module):
    def __init__(self, n_classes=2, alpha= 60, beta=5):
        super(CrossIntra_Prototypes_Loss, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha # default = 60
        self.beta = beta # default = 5
    def forward(self, x):
        B, C, K, D = x.shape
        x = x.reshape(B*C, K, D)
        x_c = torch.mean(x, dim=1) #[BC D]

        #--- intra class
        intra_dist_map = x - x_c[:, None, :] #[BC K D]
        intra_dist_map = intra_dist_map**2
        intra_dist_map = torch.sqrt(torch.sum(intra_dist_map, dim=-1)) # [BC K]
        intra_dist_map = intra_dist_map.reshape(B, C, K)
        intra_dist_map = torch.mean(intra_dist_map, dim=-1) # [B C]
        #intra_dist_map[intra_dist_map< self.beta] = self.beta

        #--- cross class
        x_c = x_c.reshape(B, C, D).permute(0, 2, 1) #[B D C]
        cross_dist_map = x_c[:, :, :, None] - x_c[:, :, None, :] + 1e-4 #[B D C C]
        cross_dist_map = cross_dist_map**2
        cross_dist_map = torch.sqrt(torch.sum(cross_dist_map, dim=1)) # [B C C]
        cross_dist_map = torch.mean(cross_dist_map, dim=-1) # [B C]
        
        #--- combine
        loss = self.alpha + intra_dist_map - cross_dist_map
        loss = torch.mean(loss, dim=-1) #[B]
        loss = torch.mean(loss, dim=-1)
        return max(loss, loss*0)


class CrossIntra_Prototypes_Loss2(nn.Module):
    def __init__(self, n_classes=2, alpha=2, beta=0.5, k_class0=1.0):
        super(CrossIntra_Prototypes_Loss2, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha # default = 1
        self.beta = beta # default = 0.3

    def forward(self, x):
        B, C, K, D = x.shape
        x = x.reshape(B*C, K, D)
        x_c = torch.mean(x, dim=1) #[BC D]

        #--- intra class
        intra_dist_map = x - x_c[:, None, :] #[BC K D]
        intra_dist_map = intra_dist_map**2
        intra_dist_map = torch.sqrt(torch.sum(intra_dist_map, dim=-1)) # [BC K]
        intra_dist_map = intra_dist_map.reshape(B, C, K)
        intra_dist_map = torch.mean(intra_dist_map, dim=-1) # [B C]
        intra_dist_map[intra_dist_map< self.beta] = self.beta

        #--- cross class
        x_c = x_c.reshape(B, C, D)
        x_c_c = torch.mean(x_c, dim=1) # [B D]
        cross_dist_map = x_c - x_c_c[:, None, :] #[B C D]
        cross_dist_map = cross_dist_map**2
        cross_dist_map = torch.sqrt(torch.sum(cross_dist_map, dim=-1)) # [B C]
        
        #--- combine
        loss = self.alpha + intra_dist_map - cross_dist_map
        loss = torch.mean(loss, dim=-1) #[B]
        loss = torch.mean(loss, dim=-1)
        return max(loss, loss*0)
        

