import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import CalculateNormPSD
import numpy as np
from util import *

from einops import rearrange

class ConditionalGenerator(nn.Module):
    def __init__(self, seq_len=300, psd_dim=35, device='cuda'):
        super(ConditionalGenerator, self).__init__()
        
        self.T = seq_len
        self.device = device
        self.norm_psd = CalculateNormPSD(Fs=30, high_pass=40, low_pass=250)
        
        self.c1 = nn.Conv3d(2, 16, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.c2 = nn.Conv3d(16, 32, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        self.c3 = nn.Conv3d(32, 32, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.c4 = nn.Conv3d(32, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        
        self.d1 = nn.ConvTranspose3d(64, 64, kernel_size=(1, 2, 2), stride=(1, 1, 1), padding=(0, 0, 0))
        self.d2 = nn.ConvTranspose3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0))
        
        self.attention = nn.MultiheadAttention(embed_dim=1, num_heads=1, batch_first=True) 
        # Fully connected layers for processing 2s ppg
        self.fc_ppg2s = nn.Sequential(
            nn.Linear(60, 120),
            nn.ReLU(),
            nn.Linear(120, 120),
        )
        self.fc_ppg4s = nn.Sequential(
            nn.Linear(120, 180),
            nn.ReLU(),
            nn.Linear(180, 180),
            #nn.Linear(180, 180),
        )
        self.fc_ppg6s = nn.Sequential(
            nn.Linear(180, 240),
            nn.ReLU(),
            nn.Linear(240, 240),
            #nn.Linear(240, 240),
        )
        self.fc_ppg8s = nn.Sequential(
            nn.Linear(240, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            #nn.Linear(300, 300),
        )

        self.fc_psd = nn.Sequential(
            nn.Linear(psd_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.T)
        )
    
        
    def generate_ppg(self, bpm, fps=30, alpha=0.25, seq_len=300):
        
        # v = bpm
        # t = np.linspace(0, seq_len, seq_len)
        # m = (t * v / (60 * fps)).astype(int)

        # x = t - (60 * m * fps) / v
        # wave_t = np.zeros_like(t)
        # for i in range(len(t)):
        #     if 0 <= x[i] < 60 * alpha * fps / v:
        #         wave_t[i] = 0.5 * np.cos((x[i] * v * np.pi) / (60 * alpha * fps)) + 0.5
        #     else:
        #         wave_t[i] = 0.5 * np.cos((x[i] * v * np.pi - 60 * alpha * np.pi * fps) / ((1 - alpha) * 60 * fps) + np.pi) + 0.5
        # return wave_t
                
        # Parameters
        # fs = 30  # sampling frequency in Hz
        # duration = 10  # seconds (1 minute)

        # # Time axis
        t = np.linspace(0, seq_len/fps, seq_len, endpoint=False)

        # # Cardiac signal parameters
        M = np.random.uniform(0.5, 2.0)
        #M=1.5
        omega = 2 * np.pi * (bpm / 60)  # Convert bpm to Hz and then to rad/s
        phi = np.random.uniform(0, 2 * np.pi)

        S = (M * np.sin(omega * t + phi) + 0.1 * M * np.sin(2 * omega * t + phi))

        return S

    def generate_input(self, bpms=None, batch_size=2, outlen=300):
        if bpms is None:
            bpms = np.random.randint(40, 240, batch_size)
        else:
            bpms = np.array(bpms)
            batch_size = bpms.shape[0]
        # print(bpms)
            
        noise = torch.randn(batch_size, 300).to(self.device)
        
        ppg = np.array([self.generate_ppg(bpm=bpms[i], fps=30, alpha=0.25, seq_len=outlen) for i in range(batch_size)])
        ppg = torch.from_numpy(ppg).float().to(self.device)
        
        psd = torch.stack([self.norm_psd(ppg[i]) for i in range(ppg.shape[0])])
        
        return ppg#, noise, psd


    #def forward(self, ppg_sig=None, bpms=None, batch_size=2, inputlen=None, tar=None):
    def forward(self, ppg_sig=None, bpms=None, batch_size=2, inputlen=60, tar=300):
        
        #ppg_origin, noise, psd_origin = self.generate_input(bpms=bpms, batch_size=batch_size)
        
        target_len=tar
        if target_len!=300:
            target_len=tar
        if tar==300:
            target_len=300

        ppg_origin = ppg_sig
        psd_origin, ppg_extend, psd_extend=None, None, None

        """if ppg_sig is not None:#input is from real 2s video or synthetic 2s signal
            batch_size=ppg_sig.shape[0]
            if inputlen==60 and tar==120:
                ppg_sig = ppg_sig.unsqueeze(2) #[B, T, 1]
                ppg_sig, _ = self.attention(ppg_sig, ppg_sig, ppg_sig) # attn_output, attn_output_weights
                ppg_sig = ppg_sig.squeeze(2) 
                ppg_extend = self.fc_ppg2s(ppg_sig)
            elif inputlen==60 and tar==180:
                ppg_sig = ppg_sig.unsqueeze(2) #[B, T, 1]
                ppg_sig, _ = self.attention(ppg_sig, ppg_sig, ppg_sig) # attn_output, attn_output_weights
                ppg_sig = ppg_sig.squeeze(2) 
                ppg_extend = self.fc_ppg2s(ppg_sig)
                ppg_extend = self.fc_ppg4s(ppg_extend)
            elif inputlen==60 and tar==240:
                ppg_sig = ppg_sig.unsqueeze(2) #[B, T, 1]
                ppg_sig, _ = self.attention(ppg_sig, ppg_sig, ppg_sig) # attn_output, attn_output_weights
                ppg_sig = ppg_sig.squeeze(2) 
                ppg_extend = self.fc_ppg2s(ppg_sig)
                ppg_extend = self.fc_ppg4s(ppg_extend)
                ppg_extend = self.fc_ppg6s(ppg_extend)
            elif inputlen==60 and tar==300:
                ppg_sig = ppg_sig.unsqueeze(2) #[B, T, 1]
                ppg_sig, _ = self.attention(ppg_sig, ppg_sig, ppg_sig) # attn_output, attn_output_weights
                ppg_sig = ppg_sig.squeeze(2) 
                ppg_extend = self.fc_ppg2s(ppg_sig)
                ppg_extend = self.fc_ppg4s(ppg_extend)
                ppg_extend = self.fc_ppg6s(ppg_extend)
                ppg_extend = self.fc_ppg8s(ppg_extend)
            elif inputlen==300 and tar==300:
                ppg_extend=ppg_origin"""
        if ppg_sig is not None:  # input is from real 2s video or synthetic 2s signal
            batch_size = ppg_sig.shape[0]
            # 擴展邏輯
            if inputlen == 60:
                # 準備進行擴展
                ppg_sig = ppg_sig.unsqueeze(2)  # [B, T, 1]
                ppg_sig, _ = self.attention(ppg_sig, ppg_sig, ppg_sig)  # attn_output, attn_output_weights
                ppg_sig = ppg_sig.squeeze(2) 
                ppg_extend = self.fc_ppg2s(ppg_sig)
                if tar >= 180:
                    ppg_extend = self.fc_ppg4s(ppg_extend)
                if tar >= 240:
                    ppg_extend = self.fc_ppg6s(ppg_extend)
                if tar == 300:
                    ppg_extend = self.fc_ppg8s(ppg_extend)

            elif inputlen == 120:
                ppg_extend = ppg_sig  # 直接開始於 120
                if tar >= 240:
                    ppg_extend = self.fc_ppg6s(ppg_extend)  # 120 → 240
                if tar == 300:
                    ppg_extend = self.fc_ppg8s(ppg_extend)  # 240 → 300

            elif inputlen == 180:
                ppg_extend = ppg_sig  # 直接開始於 180
                if tar >= 240:
                    ppg_extend = self.fc_ppg6s(ppg_extend)  # 180 → 240
                if tar == 300:
                    ppg_extend = self.fc_ppg8s(ppg_extend)  # 240 → 300

            elif inputlen == 240:
                ppg_extend = self.fc_ppg8s(ppg_sig) if tar == 300 else ppg_sig

            elif inputlen == 300:
                ppg_extend = ppg_origin  # 直接使用原始訊號
                
        # 10s Synthetic to synthetic
        if ppg_sig is None: 
            if bpms is not None and len(bpms) > 0:
                batch_size =bpms.shape[0]
            ppg_origin = self.generate_input(bpms=bpms, batch_size=batch_size, outlen=300)

            psd_origin = torch.stack([self.norm_psd(ppg_origin[i]) for i in range(ppg_origin.shape[0])])
            psd = self.fc_psd(psd_origin)


        noise = torch.randn(batch_size, target_len).to(self.device)

        if ppg_sig is not None:
            #psd = rearrange(psd_extend, 'b t -> b 1 t 1 1')
            ppg = rearrange(ppg_extend, 'b t -> b 1 t 1 1')
        else:
            #psd = rearrange(psd, 'b t -> b 1 t 1 1')
            ppg = rearrange(ppg_origin, 'b t -> b 1 t 1 1')
        
        noise = rearrange(noise, 'b t -> b 1 t 1 1')
        
        #x = torch.cat([noise, psd, ppg], dim=1)
        #print("noise",noise.shape)
        #print("ppg",ppg.shape)
        x = torch.cat([noise, ppg], dim=1)

        x = self.c1(x)
        x = F.relu(x)
        
        x = self.c2(x)
        x = F.relu(x)
        
        x = self.c3(x)
        x = F.relu(x)
        
        x = self.c4(x)
        x = F.relu(x)
        
        x = self.d1(x)
        x = F.relu(x)
        x = self.d2(x)
        
        #print("x",x.shape)
        return x, psd_origin, ppg_origin, ppg_extend



if __name__ == '__main__':


    generator = ConditionalGenerator(seq_len=300).to('cuda')

    # import neurokit2 as nk
    # ppg = nk.ppg_simulate(duration=100, sampling_rate=30, heart_rate=80)


    generator()

    # import matplotlib.pyplot as plt
    # ppg = ppg.numpy()
    # hr, psd, x_hr = hr_fft(ppg, 30, harmonics_removal=False)
    # hr2 = predict_heart_rate(ppg, 30)

    # print(hr, hr2)
    # plt.plot(x_hr, psd)
    # plt.plot(psd)
    # # plt.xlim(40, 250)
    # plt.show()
    # plt.savefig('psd.png')

    # # Generate output
    # generated_data = generator(noise, psd)
    # print(generated_data.shape)  # Should be torch.Size([B, 64, T/2, 4, 4])