import torch
import torch.nn as nn
tr = torch
import torch.nn.functional as F
import numpy as np
import torch.fft
# from util import *




class BandwidthLoss(nn.Module):
    
    # Penealize the energy outside the HR region
    
    # we reuse the code in Gideon2021 to get bandwidth loss
    # Gideon, John, and Simon Stent. "The way to my heart is through contrastive learning: Remote photoplethysmography from unlabelled video." Proceedings of the IEEE/CVF international conference on computer vision. 2021.
    def __init__(self, Fs, high_pass, low_pass):
        super(BandwidthLoss, self).__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, preds):        

        X_real = torch.view_as_real(torch.fft.rfft(preds, dim=-1, norm='forward'))

        # Determine ratio of energy between relevant and non-relevant regions
        Fn = self.Fs / 2
        freqs = torch.linspace(0, Fn, X_real.shape[-2])
        non_HR_freqs = torch.logical_or(freqs <= self.high_pass / 60, self.low_pass / 60 <= freqs)
        
        all_energy = tr.sum(tr.linalg.norm(X_real, dim=-1), dim=-1)
        non_HR_energy = tr.sum(tr.linalg.norm(X_real[:, non_HR_freqs], dim=-1), dim=-1)

        energy_ratio = tr.ones_like(all_energy)
        for ii in range(len(all_energy)):
            if all_energy[ii] > 0:
                energy_ratio[ii] = non_HR_energy[ii] / all_energy[ii]
                
        return energy_ratio




class NegBandwidthLoss(nn.Module):
    
    # Penealize the energy in the HR region
    
    # we reuse the code in Gideon2021 to get negative bandwidth loss
    # Gideon, John, and Simon Stent. "The way to my heart is through contrastive learning: Remote photoplethysmography from unlabelled video." Proceedings of the IEEE/CVF international conference on computer vision. 2021.
    def __init__(self, Fs, high_pass, low_pass):
        super(NegBandwidthLoss, self).__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, preds):        

        X_real = torch.view_as_real(torch.fft.rfft(preds, dim=-1, norm='forward'))

        # Determine ratio of energy between relevant and non-relevant regions
        Fn = self.Fs / 2
        freqs = torch.linspace(0, Fn, X_real.shape[-2])
        HR_freqs = torch.logical_and(self.high_pass / 60 <= freqs, freqs <= self.low_pass / 60)
        
        all_energy = tr.sum(tr.linalg.norm(X_real, dim=-1), dim=-1)
        HR_energy = tr.sum(tr.linalg.norm(X_real[:, HR_freqs], dim=-1), dim=-1)
        energy_ratio = tr.ones_like(all_energy)
        for ii in range(len(all_energy)):
            if all_energy[ii] > 0:
                energy_ratio[ii] = HR_energy[ii] / all_energy[ii]
                
        # print("In SparsityRatio, energy_ratio: ", energy_ratio)
        return energy_ratio


if __name__ == "__main__":
    
    loss = BandwidthLoss(30, 40, 255)
    
    x = torch.randn(2, 300)
    y = loss(x)