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
from models.conditional_GAN import ConditionalGenerator

from util import *
from datasets.dataloader import get_loader
from loss import *
import glob 
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
    

args = get_args()
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


os.makedirs(f"{result_dir}/weight/trainG", exist_ok=True)

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

#model(rPPG_model(fixed))
rPPG_model = PhysNet(S=args.model_S,
                in_ch=args.in_ch,
                conv_type=args.conv,
                seq_len=seq_len, 
                delta_t=args.delta_T*args.fps, 
                numSample=args.numSample,
                class_num=2).to(device).train()

for param in rPPG_model.parameters():
     param.requires_grad = False

cGAN_model = ConditionalGenerator(device=device).to(device).train()

#optimizer
opt_fg = optim.AdamW(rPPG_model.parameters(), lr=args.lr)
opt_cGAN = optim.AdamW(cGAN_model.parameters(), lr=args.lr)

#load pretrain model
pretrained_dir = os.path.join(get_result_dir(args, args.train_dataset, trainName), "weight")
model_pth_best = os.path.join(pretrained_dir, "fg_epoch_best.pt")
opt_pth_best = os.path.join(pretrained_dir, "fg_opt_epoch_best.pt")
print(f"getting pretrained path in {pretrained_dir}")
print(f"{model_pth_best=}")
print(f"{opt_pth_best=}")
rPPG_model.load_state_dict(torch.load(model_pth_best, map_location=device))  # load weights to the model
opt_fg.load_state_dict(torch.load(opt_pth_best, map_location=device))  # load weights to the model

cGAN_model_pth = os.path.join(get_result_dir(args, target_dataset, trainName), "weight", "cGAN.pt")
opt_cGAN_pth_best = os.path.join(pretrained_dir, "opt_cGAN.pt")
cGAN_model.load_state_dict(torch.load(cGAN_model_pth, map_location=device))
opt_cGAN.load_state_dict(torch.load(opt_cGAN_pth_best, map_location=device))  # load weights to the model
print(f"{cGAN_model_pth=}")
print(f"{opt_cGAN_pth_best=}")

CE_loss= nn.CrossEntropyLoss()
NCC_loss=NCCLoss()
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

        #get syn
        ppg_label_tensor = ppg_label
        ppg_label = ppg_label.detach().cpu().numpy()
        # gt->hr
        hr_rppg_gt = predict_heart_rate_batch(ppg_label.copy(), fs=args.fps) # this is for generate syn signal
        # ppg->hr->syn signal
        syn_ppg = best_syn_ppg(hr_rppg_gt, ppg_label_tensor, fps=30, alpha=0.25, seq_len=seq_len, num_phases=300, num_amplitudes=10, device=device)
        psd_10 = [norm_psd(syn_ppg[i]) for i in range(syn_ppg.shape[0])]
        psd_10 = torch.stack(psd_10)

        #signal
        ppg_target_2s = syn_ppg[:,120:180]
        ppg_target_4s = syn_ppg[:,90:210]
        ppg_target_6s = syn_ppg[:,60:240]
        ppg_target_8s = syn_ppg[:,30:270]

        #video
        seg_frames_2s = face_frames[:, :, 120:180, :, :]  #face_frames[:, :, 120:180, :, :] 
        cut_list=[120,180,240,300] #2s->4s->6s->8s->10s  #120,180,240,
        for index, cut in enumerate(cut_list):

            feature_real=None
            if cut==120:
                ppg_target=ppg_target_4s
                ppg_pre=ppg_target_2s
            elif cut==180:
                ppg_target=ppg_target_6s
                ppg_pre=ppg_target_4s
            elif cut==240:
                ppg_target=ppg_target_8s
                ppg_pre=ppg_target_6s
            elif cut==300:
                ppg_target=syn_ppg
                ppg_pre=ppg_target_8s

            # estimate 2s signal
            rPPG_clip2s, _ = rPPG_model(seg_frames_2s)
            rPPG_clip2s = rPPG_clip2s[:, -1]
            feature, _, _, _ = cGAN_model(ppg_sig=rPPG_clip2s, inputlen=60, tar=cut)


            # generate feature
            rPPG2tn = rPPG_model.forward_cGAN(feature)
            rPPG2tn = rPPG2tn[:, -1]
            loss_NCCnt10=0
            #learn the same periodicity characteristics from 10s signal
            if cut==300:
                loss_NCCnt10 ,_= NCC_loss(syn_ppg, rPPG2tn)
            else:
                loss_NCCnt10 = shift_window(syn_ppg, rPPG2tn, NCC_loss)
            #learn the same periodicity characteristics from previous one
            loss_NCCntpre = shift_window(ppg_pre, rPPG2tn, NCC_loss)

            opt_cGAN.zero_grad()
            total_loss = loss_NCCnt10 + loss_NCCntpre 

            total_loss.backward()
            opt_cGAN.step()
        
            loss_string =  f"[epoch {epoch} step {step}_Cut{cut}]"
            loss_string += f" loss_NCCnt10: {loss_NCCnt10.item():.4f}"
            loss_string += f" loss_NCCntpre: {loss_NCCntpre.item():.4f}"

            log.info(loss_string)

    torch.save(cGAN_model.state_dict(), result_dir + '/weight/trainG/adapt_cGAN{:03d}.pt'.format(epoch))
    #torch.save(opt_cGAN.state_dict(), result_dir + '/weight/trainG/opt_adapt_cGAN{:03d}.pt'.format(epoch))
