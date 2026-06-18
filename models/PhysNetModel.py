import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from loss.ContrastLoss import ST_sampling, get_PSD_length
from loss import CalculateNormPSD

# -------------------------------------------------------------------------------------------------------------------
# PhysNet model
# 
# the output is an ST-rPPG block rather than a rPPG signal.
# -------------------------------------------------------------------------------------------------------------------
class _PhysNet(nn.Module):
    def __init__(self, S=2, in_ch=3, conv3x3x3=nn.Conv3d):
        super().__init__()
        
        self.S = S  # S is the spatial dimension of ST-rPPG block
        #self.mapping_layer = MappingLayer()
        #self.dimension_restorer = DimensionRestorer(target_temporal_length=150)  # Set your target temporal length here
        self.start = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=32, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ELU()
        )

        # 1x
        self.loop1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # encoder
        self.encoder1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.encoder2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        #
        self.loop4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # decoder to reach back initial temporal length
        self.decoder1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )
                    
        self.end = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, S, S)),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        )
        
    def forward(self, x, y = None):
        means_x = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        stds_x = torch.std(x, dim=(2, 3, 4), keepdim=True)
        x = (x - means_x) / stds_x  # (B, C, T, 128, 128)

        parity_x = []
        x = self.start(x)  # (B, C, T, 128, 128) 
        x = self.loop1(x)
        parity_x.append(x.size(2) % 2)

        x = self.encoder1(x)

        parity_x.append(x.size(2) % 2)
        x = self.encoder2(x)
        x = self.loop4(x)

        x = F.interpolate(x, scale_factor=(2, 1, 1)) # (B, 64, T/2, 8, 8)

        #print("x.shape",x.shape)#[15, 64, 150, 4, 4]   #[15, 64, 30, 4, 4]-->[B, 3, 300, 1, 1]-->[15, 64, 150, 4, 4] 
        
        #exit()                                         #[B, 1, 60, 1, 1, conditional]--> E -->[B, 1, 60]-->[B, 1, 300]
                                                        #noise[B, 1, 60], con[B, 1, 60]=[0,...0] or [1,...,1]
        x_middle = x                                    # g_input[B, 2, 60, 1, 1] //[B, 64, 30, 4, 4]-->g_output[B, 64, 30, 4, 4]
        # Apply mapping                                 # [0,1,2,3,4]
        #print("x.shape",x.shape)                       # [60][60][60][60][60]
        """if x.size(2) < 150: #[15, 64, 74, 4, 4]
            x = self.temporal_outpainting(x) [15, 64, 150, 4, 4]"""


        x = self.decoder1(x) # (B, 64, T/2, 8, 8)
        x = F.pad(x, (0,0,0,0,0,parity_x[-1]), mode='replicate')
        x = F.interpolate(x, scale_factor=(2, 1, 1)) # (B, 64, T, 8, 8)
        x = self.decoder2(x) # (B, 64, T, 8, 8)
        x = F.pad(x, (0,0,0,0,0,parity_x[-2]), mode='replicate')
        x = self.end(x) # (B, 1, T, S, S), ST-rPPG block

        x_list = []
        for a in range(self.S):
            for b in range(self.S):
                x_list.append(x[:, :, :, a, b])  # (B, 1, T)

        x = sum(x_list) / (self.S * self.S)  # (B, 1, T)
        X = torch.cat(x_list + [x], 1)
        
        return X, x_middle
        
        
    def forward_cGAN(self, x):
    
        x = self.decoder1(x)  # (B, 64, T/2, 8, 8) 
        
        x = F.interpolate(x, scale_factor=(2, 1, 1))  # (B, 64, T, 8, 8)
        x = self.decoder2(x)  # (B, 64, T, 8, 8)
        x = self.end(x)  # (B, 1, T, S, S), ST-rPPG block
    
        x_list = []
        for a in range(self.S):
            for b in range(self.S):
                x_list.append(x[:, :, :, a, b])  # (B, 1, T)

        x = sum(x_list) / (self.S * self.S)  # (B, 1, T)
        X = torch.cat(x_list + [x], 1)
        return X
    


class PhysNet(nn.Module):
    def __init__(self, S=2, in_ch=3, conv_type=None, seq_len=300,
                 delta_t=300, numSample=1, class_num=2):
        super().__init__()
        
        conv3x3x3 = nn.Conv3d
        
        self.model = _PhysNet(S, in_ch, conv3x3x3)
        self.norm_psd = CalculateNormPSD(Fs=30, high_pass=40, low_pass=250)
        
    def forward(self, x, y=None, return_feature=False):
        
        if y is not None:
            rPPG_output, bg_output, x_middle, y_middle = self.model(x, y)

            if not return_feature:
                return rPPG_output, bg_output
            else:
                return rPPG_output, bg_output, x_middle, y_middle
        
        else:
            rPPG_output, x_middle = self.model(x)


            return rPPG_output, x_middle
        
    
    def forward_cGAN(self, x):
        
        rPPG_output = self.model.forward_cGAN(x)
        
        return rPPG_output
    """
    def forward_finetune(self, rPPG_anc, rPPG_feature):
       
        rPPG_output = self.model.forward_finetune(rPPG_anc, rPPG_feature)
        
        return rPPG_output"""


if __name__ == "__main__":
    
    B, S, T = 4, 2, 300
    x = torch.randn([B, 3, T, 64, 64])
    # y = torch.randn([B, 3, T, 64, 64])
    model = PhysNet(S=S, conv_type='LDC_M', seq_len=T, delta_t=T, numSample=1, class_num=2)
    # rPPG, bg = model(x, y)
    rPPG = model(x)
    print(rPPG.shape)

    # loss2 = BCE_loss(y_bg, bg_class_label[:, 0].long())
    
    # print(loss1, loss2)
    # print(loss1, loss2)

    # for conv in ['LDC_T', 'LDC_M', 'CDC_T', 'CDC_ST', 'CDC_TR', 'vanilla']:
    #     print(conv)
    #     model = PhysNet(conv_type=conv)
    #     model(x)
    #     # print(model(x))
    #     print(conv, 'done')
