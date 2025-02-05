
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

##Functions for continuous approximation of binary mask

def gumbel_softmax(p, temperature=1.0, epoch=0, hard=False):
    p = torch.clamp(p, 1e-6, 1 - 1e-6)
    logits = torch.log(p / (1 - p))
    logits_2d = torch.stack([logits, -logits], dim=-1)
    gumbels = -torch.empty_like(logits_2d).exponential_().log()
    y_soft = F.softmax((logits_2d + gumbels) / temperature, dim=-1)
    if hard:
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret[..., 0] 

def stochastic_mask(p):
    p = torch.clamp(p, 1e-6, 1 - 1e-6)
    return torch.clamp(2 * (p - 0.5) + 0.5 * torch.randn_like(p), 0, 1)

##MINE class

class MINE(nn.Module):
    def __init__(self, input_dim):
        super(MINE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(h))
    
##MINE loss
def mine_loss(model, joint, marginal):
    t_joint = model.mine(joint)
    t_marginal = model.mine(marginal)
    eps = 1e-6  
    mi_lb = torch.mean(t_joint) - torch.log(torch.mean(torch.exp(t_marginal)) + eps)
    
    return mi_lb 

##Regularizer to enforce structural constraints
def structured_mask_regularization(mask, row_norm_use=False):
    reg_loss = 0
    for i in range(mask.shape[0]):
        row_norm = torch.norm(mask[i, :], 1)
        other_rows_product = torch.prod(1 - mask[torch.arange(mask.shape[0]) != i, :], dim=0)
        interaction_norm = torch.norm(mask[i, :] * other_rows_product, 1)
        if(row_norm_use):
            reg_loss += row_norm*torch.relu(-1* (interaction_norm-1.9))  
        else:
            reg_loss += torch.relu(-1* (interaction_norm-1.9))
        
    return reg_loss 