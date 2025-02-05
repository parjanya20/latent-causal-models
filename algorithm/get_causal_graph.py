import torch
import torch.nn.functional as F
from algorithm.utils import *
from algorithm.hierarchical_model import *
from tqdm import tqdm

def get_graph(X, num_epochs=400, mine_epochs=100, lr=1e-3, mine_lr=1e-3, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Convert X to tensor and prepare data loader
    X = torch.tensor(X).float().to(device)
    data_loader = torch.utils.data.DataLoader(X, batch_size=32, shuffle=True)
    
    # Model dimensions
    x_dim = 1
    z_dim = 1
    num_x = X.shape[1] // x_dim
    num_z = X.shape[1] // 4 + X.shape[1] // 2
    
    # Initialize model
    model = HierarchicalLatentCausalModel(x_dim, z_dim, num_x, num_z).to(device)
    for param in model.parameters():
        param.data = param.data.float()
    
    # Optimizers
    main_optimizer = torch.optim.Adam(model.get_main_parameters(), lr=lr)
    mine_optimizer = torch.optim.Adam(model.get_mine_parameters(), lr=mine_lr)
    
    # MINE warmup
    print("Warming up MINE...")
    for _ in range(mine_epochs):
        for batch in data_loader:
            batch = batch.float().to(device)
            mine_optimizer.zero_grad()
            _, _, eps, _, _ = model(batch, temperature=1.0, epoch=0)
            eps_flat = eps.reshape(eps.shape[0], -1)
            joint = eps_flat
            marginal = torch.stack([col[torch.randperm(col.shape[0])] for col in eps_flat.T]).T            
            mi_est = mine_loss(model, joint, marginal)
            (-mi_est).backward()  # Maximize MI
            mine_optimizer.step()

    # Main training loop
    rolling_losses = []
    num_batches = len(data_loader)
    
    for epoch in tqdm(range(num_epochs), desc='Training'):
        total_recon_loss = 0
        total_kl_loss = 0
        total_mi_loss = 0
        total_mask_reg_loss = 0
        total_l1_loss = 0
        total_loss = 0
        
        temperature = 1
        for batch in data_loader:
            batch = batch.float()
            
            # Train MINE
            for _ in range(3):
                mine_optimizer.zero_grad()
                _, _, eps, _, _ = model(batch, temperature=temperature, epoch=epoch)
                eps_flat = eps.reshape(eps.shape[0], -1)
                joint = eps_flat
                marginal = torch.stack([col[torch.randperm(col.shape[0])] for col in eps_flat.T]).T                
                mi_est = mine_loss(model, joint, marginal)
                (-mi_est).backward()
                mine_optimizer.step()

            # Train main model
            main_optimizer.zero_grad()
            x_computed, z, eps, mu, logvar = model(batch, temperature, epoch=epoch)
            x_computed = x_computed.reshape(batch.shape)
            recon_loss = F.mse_loss(x_computed, batch)
            
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            eps_flat = eps.reshape(eps.shape[0], -1)
            joint = eps_flat
            marginal = torch.stack([col[torch.randperm(col.shape[0])] for col in eps_flat.T]).T            
            mi_loss = torch.relu(mine_loss(model, joint, marginal))
            
            l1_loss = torch.norm(model.mask_logits, 1)
            mask_reg_loss = structured_mask_regularization(model.mask_logits[:num_z,], row_norm_use=epoch>int(num_epochs*0.9))

            coeff = 10**(-3+4*epoch/num_epochs)
            loss = recon_loss + 1e-3*kl_loss + 10 * (mi_loss)**2 + (1e-6 / num_batches) * l1_loss + coeff*(mask_reg_loss)**2
            
            loss.backward()
            main_optimizer.step()
            
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_mi_loss += mi_loss.item()
            total_mask_reg_loss += mask_reg_loss.item()
            total_l1_loss += l1_loss.item()
            total_loss += loss.item()

        # Calculate average losses
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        avg_mi_loss = total_mi_loss / num_batches
        avg_mask_reg_loss = total_mask_reg_loss / num_batches
        avg_l1_loss = total_l1_loss / num_batches
        avg_total_loss = total_loss / num_batches

        # Update rolling losses
        rolling_losses.append(avg_recon_loss+1e-6*avg_l1_loss+1e6*avg_mask_reg_loss)
        if len(rolling_losses) > 5:
            rolling_losses.pop(0)

        # Mask logits updates and constraints
        with torch.no_grad():
            model.mask_logits.clamp_(0.0, 1)
            if epoch % 5 == 0 and epoch < int(num_epochs * 0.9):
                model.mask_logits.data = 0.9 * model.mask_logits.data + 0.05

            n = X.shape[1]
            k1 = n // 4
            k2 = n // 2
            
            temp_mask = model.mask_logits.clone()
            temp_mask[:k1, :k1] = 0
            temp_mask[:k1, k1+k2:] = 0
            temp_mask[k1:k2+k1, :k1+k2] = 0
            temp_mask[k2+k1:, :] = 0
            model.mask_logits.data = temp_mask
            model.mask_logits[num_z:, :].clamp_(0, 0)

            if epoch > int(num_epochs * 0.75):
                for col in range(model.mask_logits.shape[1]):
                    if torch.all(model.mask_logits[:, col] > 0.5):
                        model.mask_logits[:, col] = 1.0
            if epoch >= int(num_epochs * 0.9):
                with torch.no_grad():
                    model.mask_logits[model.mask_logits > threshold] = 1
                    model.mask_logits[model.mask_logits <= threshold] = 0
                model.mask_logits.requires_grad = False
                main_optimizer = torch.optim.Adam([p for n, p in model.named_parameters() 
                                                if not n.startswith('mine') and n != 'mask_logits'], lr=lr)
            
        # if epoch % 20 == 0:
        #     print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {avg_total_loss:.4f}, "
        #           f"Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}, "
        #           f"MI Loss: {avg_mi_loss:.4f}, Mask Reg Loss: {avg_mask_reg_loss:.4f}, "
        #           f"L1 Loss: {avg_l1_loss:.4f}")
        #     print(torch.round(model.mask_logits * 100) / 100)

    final_loss = sum(rolling_losses) / len(rolling_losses)

    return final_loss, model.mask_logits
