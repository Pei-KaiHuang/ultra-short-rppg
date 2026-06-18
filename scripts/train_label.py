import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch import optim

from einops import rearrange
from models.PhysNetModel import PhysNet
from util import *
from datasets.dataloader import get_loader
from loss import *
import glob 
from scipy.stats import pearsonr
from scipy.signal import find_peaks
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

args = get_args()

name=args.name

finetune=False
target_dataset = args.train_dataset
if args.finetune_dataset:
    finetune=True
    target_dataset = args.finetune_dataset

trainName, _, finetuneName = get_name(args, finetune=finetune, model_name=PhysNet.__name__)

if finetune:
    log = get_logger(get_log_dir(args, "finetune", target_dataset), finetuneName)
    result_dir = get_result_dir(args, target_dataset, finetuneName)
else:
    log = get_logger(get_log_dir(args, "train", target_dataset), trainName)
    result_dir = get_result_dir(args, target_dataset, trainName)

print(f"{trainName=}, {finetuneName=}")

os.makedirs(f"{result_dir}/weight", exist_ok=True)

seq_len = args.train_T*args.fps
not_preload = args.do_not_preload
if_bg = args.bg
train_loader = get_loader(_datasets=target_dataset,
                          _seq_length=seq_len,
                          batch_size=args.bs,
                          train=True,
                          if_bg=if_bg,
                          shuffle=True, 
                          real_or_fake="real",
                          if_preload=not_preload,
                          if_aug=False,
                          testFold=args.testFold,
                          **get_loader_path_kwargs(args))

model = PhysNet(S=args.model_S,
                in_ch=args.in_ch,
                conv_type=args.conv,
                seq_len=seq_len, 
                delta_t=args.delta_T*args.fps, 
                numSample=args.numSample,
                class_num=2).to(device).train()

# 計算模型參數數量
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info(f"Total Trainable Parameters: {total_params}")

# 計算 FLOPs
"""dummy_input = torch.randn(4, 3, 60, 64, 64).to(device) #是用兩秒算嗎?
log.info(f"Input shape: {dummy_input.shape}")
total_flops = profile_macs(model, dummy_input)
total_gflops = total_flops / 1e9  # 轉換為 GFLOPs
log.info(f"Total GFLOPs: {total_gflops:.3f}")"""

opt_fg = optim.AdamW(model.parameters(), lr=args.lr)

if finetune:
    pretrained_dir = os.path.join(get_result_dir(args, args.train_dataset, trainName), "weight")
    model_pth_list = sorted(glob.glob(os.path.join(pretrained_dir, "fg_epoch*.pt")))
    opt_pth_list = sorted(glob.glob(os.path.join(pretrained_dir, "fg_opt_epoch*.pt")))
    if not model_pth_list or not opt_pth_list:
        raise FileNotFoundError(f"Missing pretrained checkpoints in {pretrained_dir}")
    print(f"getting pretrained path in {pretrained_dir}")
    print(f"{model_pth_list[-1]=}")
    print(f"{opt_pth_list[-1]=}")
    model.load_state_dict(torch.load(model_pth_list[-1], map_location=device)) 
    opt_fg.load_state_dict(torch.load(opt_pth_list[-1], map_location=device)) 

IPR = IrrelevantPowerRatio(Fs=args.fps, 
                           high_pass=args.high_pass, low_pass=args.low_pass)


SR = SparsityRatio(Fs=args.fps, 
                   high_pass=args.high_pass, low_pass=args.low_pass)

loss_func_name = 'ncc'
if loss_func_name == 'ncc':
    sig_loss = NCCLoss()
elif loss_func_name == 'np':
    sig_loss = NegPearsonLoss()

CE_loss= nn.CrossEntropyLoss()
norm_psd = CalculateNormPSD(Fs=30, high_pass=40, low_pass=250)

for epoch in range(args.epoch):

    print(f"epoch_train: {epoch}/{args.epoch}:")
    for step, (face_frames, bg_frames, ls_label, ppg_label, subjects) in enumerate(train_loader):

        # face_frames = rearrange(face_frames, 'b d t c h w -> (b d) t c h w').to(device)
        # if batch size < 2, skip this round since we need at least 2 samples for the contrastive loss
        if(face_frames.shape[0] < 2):
            continue
        face_frames = face_frames.to(device) # [B, 3, 300, 64, 64]
        ppg_label = ppg_label.to(device) # [B, 300]

        # print(f"{rPPG_anc.shape=}")
        
        ppg_label_tensor = ppg_label
        ppg_label = ppg_label.detach().cpu().numpy()
        #ppg_label = butter_bandpass_batch(ppg_label, lowcut=0.6, highcut=4, fs=args.fps)#add
        # gt->hr
        hr_rppg_gt = predict_heart_rate_batch(ppg_label.copy(), fs=args.fps) # this is for generate syn signal
        # ppg->hr->syn signal
        syn_ppg = best_syn_ppg(hr_rppg_gt, ppg_label_tensor, fps=30, alpha=0.25, seq_len=seq_len, num_phases=300, num_amplitudes=10, device=device)
        psd_syn = [norm_psd(syn_ppg[i]) for i in range(syn_ppg.shape[0])]
        psd_syn = torch.stack(psd_syn)

        cut_list=[300, 240, 180, 120, 60] #300, 240, 180, 
        #========== Weighted spectral-based cross entropy loss(calculate dynamic weight) ===========
        entropy_list=[]
        for index, cut in enumerate(cut_list):
            syn_ppg_clip = syn_ppg[:, 150 - (cut // 2):150 + (cut // 2)] #
            entropy_clip=syn_ppg_clip.unsqueeze(1)
            entropy=PSD_entropy(entropy_clip)
            entropy_list.append(entropy)
        entropy_list = torch.stack(entropy_list) #[5, B]
        
        # Min-Max normalization
        min_val = entropy_list.min()
        max_val = entropy_list.max()
        normalized_entropy_list = (entropy_list - min_val) / (max_val - min_val)
        normalized_entropy_list=normalized_entropy_list.to(device)
        normalized_entropy_list=1-normalized_entropy_list
        
        #========== Periodicity-guided rPPG estimation ==========
        for index, cut in enumerate(cut_list):

            rPPG_clip = None
            #cut to get the clip
            segment_frames = face_frames[:, :, 150 - (cut // 2):150 + (cut // 2), :, :] 
            syn_ppg_clip = syn_ppg[:, 150 - (cut // 2):150 + (cut // 2)] 
            rPPG_clip, _ = model(segment_frames)#[6, 5, 300]
            rPPG_clip = rPPG_clip[:, -1]

            psd_clip = [norm_psd(rPPG_clip[i]) for i in range(rPPG_clip.shape[0])]
            psd_clip = torch.stack(psd_clip)
            psd_syn_clip = [norm_psd(syn_ppg_clip[i]) for i in range(syn_ppg_clip.shape[0])]
            psd_syn_clip = torch.stack(psd_syn_clip)

            #loss_sig_clip,_ = sig_loss(syn_ppg_clip, rPPG_clip) 
            #dynamic ce loss
            loss_ce=0
            for i in range(psd_clip.shape[0]):
                tmp = CE_loss(psd_clip[i], psd_syn_clip[i]) * normalized_entropy_list[index][i]
                loss_ce += tmp         
            loss_ce_clip=loss_ce/psd_clip.shape[0]
            #loss_ce_clip = CE_loss(psd_clip, psd_syn_clip) * normalized_entropy_list[index]
            #========== Maximized periodic similarity loss ==========
            count=300-cut+1
            ncc_label = []
            ncc_pre = []
            for i in range(0,count):
                syn_ppg_shift=syn_ppg[:,i:i+cut]#0,1,2,...,1/61/121/181/241
                if syn_ppg_shift.shape[1]!=cut:
                    raise ValueError(f"syn_ppg_shift shape is not correct {syn_ppg_shift.shape}.")
                ncc_val_pre, best_ncc_pre = sig_loss(syn_ppg_shift, rPPG_clip)
                ncc_pre.append(best_ncc_pre)

            ncc_pre = torch.stack(ncc_pre).transpose(0, 1)  # Shape: [batch_size, shift]

            peaks = cus_find_peaks_batch(ncc_pre) #return list
            loss_ncc_clip = process_peak_values(ncc_pre, peaks)
            loss_ncc_clip = loss_ncc_clip.mean()
    
            opt_fg.zero_grad()
            total_loss =  loss_ce_clip + loss_ncc_clip 
            total_loss.backward()
            opt_fg.step()
            loss_string =  f"[epoch {epoch} step {step} cut {cut}]"
            loss_string += f" loss_ce_clip: {loss_ce_clip.item():.5f}"
            loss_string += f" loss_ncc_clip: {loss_ncc_clip.item():.5f}"
            log.info(loss_string)

    torch.save(model.state_dict(), result_dir + '/weight/fg_epoch{:03d}.pt'.format(epoch))
    torch.save(opt_fg.state_dict(), result_dir + '/weight/fg_opt_epoch{:03d}.pt'.format(epoch))
