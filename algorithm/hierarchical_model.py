
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithm.utils import *

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        logvar = torch.maximum(logvar,torch.tensor(-1.0))
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, torch.maximum(logvar,torch.tensor(-1.0))

class NodeFunction(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NodeFunction, self).__init__()
        self.fc1 = nn.Linear(input_dim + 1, 32) 
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x, eps):
        h = torch.cat([x, eps], dim=-1)
        h = F.relu(self.fc1(h))
        return self.fc2(h)

class HierarchicalLatentCausalModel(nn.Module):
    def __init__(self, x_dim, z_dim, num_x, num_z, temperature=0.5):
        super(HierarchicalLatentCausalModel, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.num_x = num_x
        self.num_z = num_z
        self.temperature = temperature

        total_vars = num_x + num_z
        self.vae = VAE(x_dim * num_x, num_z + num_x)
        
        self.mask_logits = nn.Parameter(torch.full((total_vars, total_vars), 0.5, dtype=torch.float32))
        with torch.no_grad():
            self.mask_logits.data = torch.triu(self.mask_logits.data, diagonal=1)
            self.mask_logits.data[self.num_x-1:, :] = 0
        
        self.node_functions = nn.ModuleList([
            NodeFunction(i * z_dim, z_dim) for i in range(num_z)
        ] + [
            NodeFunction(z_dim * num_z+x_dim*j, x_dim) for j in range(num_x)
        ])
        self.mine = MINE(num_z + num_x)

    def get_main_parameters(self):
        return [p for name, p in self.named_parameters() if not name.startswith('mine')]

    def get_mine_parameters(self):
        return self.mine.parameters()

    def get_mask(self, temperature=1.0, epoch=0):
        mask = gumbel_softmax(self.mask_logits, temperature=temperature, hard=False)
        mask = torch.triu(mask, diagonal=1)  
        mask[self.num_z:, :] = 0 
        mask = mask.unsqueeze(1).repeat(1, 1, 1).reshape(-1, mask.shape[1])
        return torch.clamp(mask,0,1)

    def forward(self, x, temperature, epoch):
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)

        eps, mu, logvar = self.vae(x_flat)
        eps = eps.reshape(batch_size, self.num_z + self.num_x, 1)
        eps[:,self.num_z:] = 0.0
        mask = self.get_mask(temperature=temperature, epoch=epoch)

        all_vars = []

        for i in range(self.num_z + self.num_x):
            if i == 0:
                parents = torch.zeros(batch_size, 0, device=x.device)
            else:
                parents = torch.cat([all_vars[j] for j in range(i)], dim=1)
            parents = torch.mul(mask[:i, i].unsqueeze(0), parents)
            var = self.node_functions[i](parents, eps[:, i])
            all_vars.append(var)

        z_computed = torch.stack(all_vars[:self.num_z], dim=1)
        x_computed = torch.stack(all_vars[self.num_z:], dim=1)
        
        return x_computed, z_computed, eps, mu, logvar

