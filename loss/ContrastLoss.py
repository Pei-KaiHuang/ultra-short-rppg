import torch
import torch.nn as nn
import torch.nn.functional as F
tr = torch


class FrequencySampler(nn.Module):
    """ Frequency contrast wrapper around a backbone model e.g. PhysNet
    """
    def __init__(self, model, seq_len=300):
        super().__init__()

        self.backbone = model
        self.upsampler = nn.Upsample(size=(seq_len), mode='linear', align_corners=False)

    def forward(self, x_anc):
        
        T = x_anc.shape[2]

        # Resample input
        # freq_factor = 1.25 + (torch.rand(1, device=x_a.device) / 4)
        freq_factor = torch.FloatTensor(1).uniform_(0.6, 1.4).to(x_anc.device)
        while(torch.abs(freq_factor - 1) < 0.1):
            freq_factor = torch.FloatTensor(1).uniform_(0.6, 1.4).to(x_anc.device)
        
        target_size = int(T / freq_factor)
        resampler = nn.Upsample(size=(target_size, x_anc.shape[3], x_anc.shape[4]),
                                mode='trilinear',
                                align_corners=False)
        
        x_neg = resampler(x_anc)
        if x_neg.shape[2] < T:
            x_neg = F.pad(x_neg, (0, 0, 0, 0, 0, T - target_size))
        else:
            x_neg = x_neg[:, :, :T] 

        # Get negative PPG
        y_neg = self.backbone(x_neg)
        y_neg = y_neg[:, -1].unsqueeze(1)
        
        # Remove padding
        y_neg = y_neg[:,:,:target_size]

        # Resample negative PPG to create positive sample
        y_pos = self.upsampler(y_neg)

        return y_pos, y_neg
    
    

class CalculateNormPSD(nn.Module):
    # we reuse the code in Gideon2021 to get the normalized power spectral density
    # Gideon, John, and Simon Stent. "The way to my heart is through contrastive learning: Remote photoplethysmography from unlabelled video." Proceedings of the IEEE/CVF international conference on computer vision. 2021.
    
    def __init__(self, Fs, high_pass, low_pass):
        super().__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, x, zero_pad=0, if_norm=True):
        x = x - torch.mean(x, dim=-1, keepdim=True)
        if zero_pad > 0:
            L = x.shape[-1]
            x = F.pad(x, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant', 0)

        # Get PSD
        x = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm='forward'))
        x = tr.add(x[:, 0] ** 2, x[:, 1] ** 2)

        # Filter PSD for relevant parts
        Fn = self.Fs / 2
            
        freqs = torch.linspace(0, Fn, x.shape[0])
        use_freqs = torch.logical_and(freqs >= self.high_pass / 60, freqs <= self.low_pass / 60)
        x = x[use_freqs]

        # Normalize PSD
        if if_norm:
            x = x / torch.sum(x, dim=-1, keepdim=True)
                
        return x

# class CalculateNormPSD(nn.Module):
#     # we reuse the code in Gideon2021 to get the normalized power spectral density
#     # Gideon, John, and Simon Stent. "The way to my heart is through contrastive learning: Remote photoplethysmography from unlabelled video." Proceedings of the IEEE/CVF international conference on computer vision. 2021.
    
#     def __init__(self, Fs, high_pass, low_pass):
#         super().__init__()
#         self.Fs = Fs
#         self.high_pass = high_pass
#         self.low_pass = low_pass

#     def forward(self, x, zero_pad=0):
#         x = x - torch.mean(x, dim=-1, keepdim=True)
#         if zero_pad > 0:
#             L = x.shape[-1]
#             x = F.pad(x, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant', 0)

#         # Get PSD
#         x = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm='forward'))
#         x = tr.add(x[..., 0] ** 2, x[..., 1] ** 2)

#         # Filter PSD for relevant parts
#         Fn = self.Fs / 2
            
#         freqs = torch.linspace(0, Fn, x.shape[-1])
#         use_freqs = torch.logical_and(freqs >= self.high_pass / 60, freqs <= self.low_pass / 60)
#         x = x[..., use_freqs]

#         # Normalize PSD
#         x = x / torch.sum(x, dim=-1, keepdim=True)
                
#         return x
    



class ST_sampling(nn.Module):
    # spatiotemporal sampling on ST-rPPG block.
    
    def __init__(self, delta_t, numSample, Fs, high_pass, low_pass):
        super().__init__()
        self.delta_t = delta_t # time length of each rPPG sample
        self.numSample = numSample # the number of rPPG samples at each spatial position
        self.norm_psd = CalculateNormPSD(Fs, high_pass, low_pass)

    def forward(self, input): # input: (B, N, T)
        samples = []
        
        # if input is 2D, add a spatial view dimension
        if input.dim() == 2:
            input = input.unsqueeze(1)
            
        for b in range(input.shape[0]): # loop over videos
            samples_per_video = []
            for c in range(input.shape[1]): # loop for sampling over spatial dimension
                for i in range(self.numSample): # loop for sampling with time length delta_t along temporal dimension
                    offset = torch.randint(0, input.shape[-1] - self.delta_t + 1, (1,), device=input.device) # randomly sample along temporal dimension
                    x = self.norm_psd(input[b, c, offset:offset + self.delta_t])
                    
                    samples_per_video.append(x)
            samples.append(samples_per_video)
        return samples
        # for b in range(input.shape[0]): # loop over videos
        #     xs = []
        #     for c in range(input.shape[1]): # loop for sampling over spatial dimension
        #         for i in range(self.numSample): # loop for sampling with time length delta_t along temporal dimension
        #             offset = torch.randint(0, input.shape[-1] - self.delta_t + 1, (1,), device=input.device) # randomly sample along temporal dimension
        #             x = input[b, c, offset:offset + self.delta_t]
        #             xs.append(x)
            
        #     xs = torch.stack(xs, dim=0)
        #     samples.append(self.norm_psd(xs))
            
        # samples = torch.stack(samples, dim=0)
        # return samples
    


class ContrastLoss(nn.Module):
    def __init__(self, delta_t, K, Fs, high_pass, low_pass):
        
        super(ContrastLoss, self).__init__()
        self.ST_sampling = ST_sampling(delta_t, K, Fs, high_pass, low_pass) # spatiotemporal sampler
        self.distance_func = nn.MSELoss(reduction = 'mean') # mean squared error for comparing two PSDs


    def compare_samples(self, list_a, list_b, exclude_same=False):
        if exclude_same:
            total_distance = 0.
            M = 0
            for i in range(len(list_a)):
                for j in range(len(list_b)):
                    if i != j:
                        total_distance += self.distance_func(list_a[i], list_b[j])
                        M += 1
        else:
            total_distance = 0.
            M = 0
            for i in range(len(list_a)):
                for j in range(len(list_b)):
                    total_distance += self.distance_func(list_a[i], list_b[j])
                    M += 1
        return total_distance / M
    
    
    def self_pos_neg_loss(self, output):
        
        # output shape [B, 4+1, 300]
        
        B = output.size(0)
        output = self.ST_sampling(output)
        
        pos_loss = 0
        count = 0
        for i in range(B):
            pos_loss += self.compare_samples(output[i], output[i], exclude_same=True)
            count += 1
        pos_loss = pos_loss / count

        neg_loss = 0
        count = 0
        for i in range(B):
            for j in range(i + 1, B):
                neg_loss += -self.compare_samples(output[i], output[j])
                count += 1
        neg_loss = neg_loss / count

        return pos_loss + neg_loss
    
    
    def custom_neg_loss(self, output, output2):
        
        # output shape [B, 4+1, 300]
        B = output.size(0)
        output = self.ST_sampling(output)
        output2 = self.ST_sampling(output2)
        
        neg_loss = 0
        count = 0
        for i in range(B):
            neg_loss += -self.compare_samples(output[i], output2[i])
            count += 1
        neg_loss = neg_loss / count

        return neg_loss
    
    

    def forward(self, rPPG_output=None, bg_output=None, 
                rPPG_anc=None, rPPG_pos=None, rPPG_neg=None):
        
        loss_rPPG, loss_bg, loss_triplet = 0, 0, 0
        
        
        # rPPG contrastive loss
        if (rPPG_output is not None):
            
            loss_rPPG = self.self_pos_neg_loss(rPPG_output)
        
        
        # noise contrastive loss
        if(bg_output is not None):

            loss_bg = self.self_pos_neg_loss(bg_output)# + self.custom_neg_loss(rPPG_output, bg_output)


        if (rPPG_anc is not None) and (rPPG_pos is not None) and (rPPG_neg is not None):
            
            B = rPPG_anc.size(0)
            
            rPPG_anc = self.ST_sampling(rPPG_anc)
            rPPG_pos = self.ST_sampling(rPPG_pos)
            rPPG_neg = self.ST_sampling(rPPG_neg)
            
            loss_triplet = 0
            
            for i in range(B):
                loss_triplet += self.compare_samples(rPPG_anc[i], rPPG_pos[i])
                loss_triplet -= self.compare_samples(rPPG_anc[i], rPPG_neg[i])
            loss_triplet = loss_triplet / B
        
        
        return loss_rPPG, loss_bg, loss_triplet

    
    def forward_aug(self, rPPG_output, rPPG_output_aug):

        B = rPPG_output.size(0)
        rPPG_output = self.ST_sampling(rPPG_output)
        rPPG_output_aug = self.ST_sampling(rPPG_output_aug)        


        loss_pos, loss_neg = 0, 0
        for i in range(B):
            loss_pos += self.compare_samples(rPPG_output[i], rPPG_output[i], exclude_same=True)
            loss_pos += self.compare_samples(rPPG_output_aug[i], rPPG_output_aug[i], exclude_same=True)
        loss_pos = loss_pos / (2 * B)
        
        count=0
        for i in range(B):
            for j in range(i + 1, B):
                loss_neg += -self.compare_samples(rPPG_output[i], rPPG_output[j])
                loss_neg += -self.compare_samples(rPPG_output_aug[i], rPPG_output_aug[j])
                loss_neg += -self.compare_samples(rPPG_output[i], rPPG_output_aug[j])
                count += 3 
        loss_neg = loss_neg / count

        
        return loss_pos, loss_neg
        

    
def get_PSD_length(seq_len):
    
    a = torch.randn(1, 1, seq_len)

    # from ContrastLoss import ST_sampling
    samplePSDs = ST_sampling(delta_t=seq_len, numSample=1, 
                                Fs=30, high_pass=40, low_pass=250,)

    psd = samplePSDs(a)

    return psd[0][0].shape[0]



    
class SimilarityLoss(nn.Module):
    def __init__(self, delta_t, K, Fs, high_pass, low_pass):
        
        super(SimilarityLoss, self).__init__()
        self.ST_sampling = ST_sampling(delta_t, K, Fs, high_pass, low_pass) # spatiotemporal sampler
        self.distance_func = nn.MSELoss(reduction = 'mean') # mean squared error for comparing two PSDs
        
        self.B = 0
    
    
    def compare_samples(self, list_a, list_b, exclude_same=False):
        if exclude_same:
            total_distance = 0.
            M = 0
            for i in range(len(list_a)):
                for j in range(len(list_b)):
                    if i != j:
                        total_distance += self.distance_func(list_a[i], list_b[j])
                        M += 1
        else:
            total_distance = 0.
            M = 0
            for i in range(len(list_a)):
                for j in range(len(list_b)):
                    total_distance += self.distance_func(list_a[i], list_b[j])
                    M += 1
        return total_distance / M
    
    
    def forward_self(self, anc):
        
        _loss = 0

        for i in range(self.B):
            _loss += self.compare_samples(anc[i], anc[i], exclude_same=True)
        _loss = _loss / self.B
        
        return _loss
        
    
    def forward_pair(self, anc, target, if_positive=True):
        
        _loss = 0
        
        for i in range(self.B):
            _loss += self.compare_samples(anc[i], target[i])
        _loss = _loss / self.B
        
        if not if_positive:
            _loss = _loss * -1
            
        return _loss
        
    
    def forward(self, anc, pos=None, neg=None, factor=1.0):
        
        total_loss = 0
        self.B = anc.size(0)


        anc_sampling = self.ST_sampling(anc)
        
        loss_anc = self.forward_self(anc_sampling)
        total_loss = total_loss + loss_anc
        
        
        # only when positive pair is provided
        if pos is not None:
                
            pos_sampling = self.ST_sampling(pos)
            
            loss_anc_pos = self.forward_pair(anc_sampling, pos_sampling)
            loss_pos = self.forward_self(pos_sampling)
            
            total_loss = total_loss + loss_anc_pos + loss_pos
        

        # only when negative pair is provided
        if neg is not None:
    
            neg_sampling = self.ST_sampling(neg)
            
            loss_anc_neg = self.forward_pair(anc_sampling, neg_sampling, if_positive=False)
            # loss_neg = self.forward_self(neg_sampling)
            loss_anc_neg = loss_anc_neg * torch.abs(1 - factor)
                        
            total_loss = total_loss + loss_anc_neg# + loss_neg
        
        
        return total_loss

        
        
        
    

if __name__ == "__main__":
    
    x = torch.rand(4, 300)
    
    loss = SimilarityLoss(delta_t=300//2, K=4, Fs=30, high_pass=0.7, low_pass=4)
    loss(x)
    loss(x, x)
    
    freq_factor = torch.FloatTensor(1).uniform_(0.6, 1.4)
    
    loss(x, x, x, freq_factor)