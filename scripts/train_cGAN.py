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


os.makedirs(f"{result_dir}/weight", exist_ok=True)

seq_len = args.train_T*args.fps

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

CE_loss= nn.CrossEntropyLoss()
NCC_loss=NCCLoss()


for epoch in range(args.epoch):
    print(f"epoch_train: {epoch}/{args.epoch}:")


    feature10, psd_target10, ppg_target10s, _ = cGAN_model(batch_size=args.bs, inputlen=300, tar=300)
    ppg_target_2s = ppg_target10s[:,120:180]
    ppg_target_4s = ppg_target10s[:,90:210]
    ppg_target_6s = ppg_target10s[:,60:240]
    ppg_target_8s = ppg_target10s[:,30:270]

    cut_list=[120,180,240,300] #2s->4s->6s->8s->10s

    for index, cut in enumerate(cut_list):
        if cut==120:
            ppg_target=ppg_target_4s
        elif cut==180:
            ppg_target=ppg_target_6s
        elif cut==240:
            ppg_target=ppg_target_8s
        elif cut==300:
            ppg_target=ppg_target10s
        #2s->4s/
        feature, _, _ , ppg_extend = cGAN_model(ppg_sig=ppg_target_2s, inputlen=60, tar=cut)

        rPPG2tn = rPPG_model.forward_cGAN(feature)
        rPPG2tn = rPPG2tn[:, -1]
        #psd
        psd_tar_2tn = [cGAN_model.norm_psd(ppg_target[i]) for i in range(ppg_target.shape[0])]
        psd_tar_2tn = torch.stack(psd_tar_2tn)

        psd2tn = [cGAN_model.norm_psd(rPPG2tn[i]) for i in range(rPPG2tn.shape[0])]
        psd2tn = torch.stack(psd2tn)
        loss_ce = CE_loss(psd2tn, psd_tar_2tn)
        loss_NCC ,_= NCC_loss(ppg_target, rPPG2tn)

        opt_cGAN.zero_grad()
        total_loss = loss_NCC + loss_ce #+ loss_NCC_L
        total_loss.backward()
        opt_cGAN.step()

    
        loss_string =  f"[epoch {epoch}_Cut{cut}]"
        loss_string += f" loss_NCC: {loss_NCC.item():.4f}"
        loss_string += f" loss_ce: {loss_ce.item():.4f}"

        log.info(loss_string)

    # torch.cuda.empty_cache()
torch.save(cGAN_model.state_dict(), result_dir + '/weight/cGAN.pt')
torch.save(opt_cGAN.state_dict(), result_dir + '/weight/opt_cGAN.pt')
