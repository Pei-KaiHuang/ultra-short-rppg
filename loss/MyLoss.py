import torch
import numpy as np
import torch.nn as nn

class NegPearsonLoss(nn.Module):
    def __init__(self):
        super(NegPearsonLoss, self).__init__()
        return
    def forward(self, x, y, inf=None):
        # for i in range(x.shape[0]):
        vx = x - torch.mean(x, dim = 1, keepdim = True)
        vy = y - torch.mean(y, dim = 1, keepdim = True)
        r = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        cost = 1 - r
            
        return cost

class NCCLoss(nn.Module):
    def __init__(self):
        super(NCCLoss, self).__init__()
    def normalized_cross_correlation(self, si, sj):
        if si.size(0)!=sj.size(0):
            raise ValueError(f"The sizes of si_centered ({si.size(0)}) and sj_centered ({sj.size(0)}) must be equal.")
        # Calculate mean
        si_mean = torch.mean(si)
        sj_mean = torch.mean(sj)
        
        # Subtract mean
        si_centered = si - si_mean
        sj_centered = sj - sj_mean

        # Calculate cross-correlation
        ncc = torch.nn.functional.conv1d(
            si_centered.unsqueeze(0).unsqueeze(0), #[1, 1, signal_length_si]
            sj_centered.unsqueeze(0).unsqueeze(0), #[1, 1, signal_length_sj]
            padding=(sj_centered.size(0))
        ).squeeze()

        # Normalize the cross-correlation
        norm = torch.sqrt(torch.sum(si_centered ** 2) * torch.sum(sj_centered ** 2))
        ncc /= norm
        
        return ncc

    def forward(self, si, sj, inf=None):
        # Ensure si and sj have shape [batch_size, signal_length]
        assert si.dim() == 2 and sj.dim() == 2, "Input tensors must have shape [batch_size, signal_length]"
        
        batch_size = si.size(0)
        losses = []
        ncc_data = [] 
        all_best_ncc = torch.zeros(batch_size, device=si.device) 

        for i in range(batch_size):
            ncc = self.normalized_cross_correlation(si[i], sj[i])
            ncc_data.append(ncc)
            best_ncc=torch.max(ncc)
            all_best_ncc[i] = best_ncc 
            count_max = torch.sum(ncc == best_ncc).item()  
            #print("Count of maximum value:", count_max)
            loss = 1 - best_ncc  # Map NCC from [-1, 1] to [0, 2] and subtract 1 to make loss closer to 0 better
            losses.append(loss)
        return torch.stack(losses).mean(), all_best_ncc

class OrthogonalLoss(nn.Module):
    def __init__(self):
        super(OrthogonalLoss, self).__init__()
        self.ST_GlobalAvgpool = nn.AdaptiveAvgPool3d((1,1,1))
        return

    def forward(self, rPPG_feat, id_feat, domain_feat):
        # print(f"rPPG_feat.size(): {rPPG_feat.size()}")
        # print(f"id_feat.size(): {id_feat.size()}")
        # print(f"domain_feat.size(): {domain_feat.size()}")
        rPPG_vector = self.ST_GlobalAvgpool(rPPG_feat).squeeze()
        id_vector = self.ST_GlobalAvgpool(id_feat).squeeze()
        domain_vector = self.ST_GlobalAvgpool(domain_feat).squeeze()
        # print(f"rPPG_vector.size(): {rPPG_vector.size()}")
        # print(f"id_vector.size(): {id_vector.size()}")
        # print(f"domain_vector.size(): {domain_vector.size()}")
        loss = 0.0
        for i in range(rPPG_vector.shape[0]):
            inner_R_I = torch.inner(rPPG_vector[i], id_vector[i])
            inner_R_D = torch.inner(rPPG_vector[i], domain_vector[i])
            inner_I_D = torch.inner(id_vector[i], domain_vector[i])
            # print(f"rPPG_vector: {rPPG_vector[i]}")
            # print(f"id_vector: {id_vector[i]}")
            # print(f"domain_vector: {domain_vector[i]}")
            # print(f"inner_R_I:\n{inner_R_I}")
            # print(f"inner_R_D:\n{inner_R_D}")
            # print(f"inner_I_D:\n{inner_I_D}")
            loss += (inner_R_I+inner_R_D+inner_I_D)
        return loss

class Cos_Sim_loss(nn.Module):
    
    def __init__(self):
        super(Cos_Sim_loss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, output, label):
        return 1 - torch.mean(self.cos(output, label))
    
    
if __name__ == "__main__":
    print("MyLoss.py")
    neg_pearson_loss = NegPearsonLoss()
    x = torch.randn(6, 5, 300)
    y = torch.randn(6, 5, 300)
    loss = neg_pearson_loss(x, y)
    print(f"loss: {loss}")
    
    x = x.view(30, 300)
    y = y.view(30, 300)
    loss = neg_pearson_loss(x, y)
    print(f"loss: {loss}")