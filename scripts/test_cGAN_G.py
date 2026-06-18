import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.PhysNetModel import PhysNet
from util import *
import argparse
from models.conditional_GAN import ConditionalGenerator
from datasets.dataloader import get_loader
from loss import *
import random
try:
    from torchprofile import profile_macs
except ImportError:
    profile_macs = None
import time
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

#seed
seed_everything(seed=42) 

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

else:
    device = torch.device('cpu')


args = get_args()
stride=300
finetune=False
target_dataset = args.train_dataset
if args.finetune_dataset:
    finetune=True
    
trainName, testName, finetuneName = get_name(args, finetune=finetune, model_name=PhysNet.__name__)
log = get_logger(get_log_dir(args, "test_cGAN_G", args.test_dataset), testName)

print(f"{trainName=}")
print(f"{finetuneName=}")
print(f"{testName=}")

targetName = trainName
if finetune:
    target_dataset = args.finetune_dataset
    targetName = finetuneName
    
seq_len = args.train_T * args.fps
num_batch_per_sample=1
not_preload = args.do_not_preload

test_loader = get_loader(_datasets=args.test_dataset,
                         _seq_length=seq_len,
                         batch_size=args.bs,
                         train=False,
                         if_bg=False,
                         shuffle=False, 
                         real_or_fake="real",
                         num_batch_per_sample=num_batch_per_sample,
                         if_preload=not_preload,
                         if_aug=False,
                         testFold=args.testFold,
                         **get_loader_path_kwargs(args))

targetName = trainName if not finetune else finetuneName


rPPG_model = PhysNet(S=args.model_S,
                in_ch=args.in_ch,
                conv_type=args.conv,
                seq_len=seq_len, 
                delta_t=args.delta_T*args.fps, 
                numSample=args.numSample,
                class_num=2).to(device).eval()

rPPG_model_pth = os.path.join(get_result_dir(args, args.train_dataset, trainName), "weight", "fg_epoch_best.pt")
rPPG_model.load_state_dict(torch.load(rPPG_model_pth, map_location=device))  # load weights to the model
print(f"{rPPG_model_pth=}")
cGAN_model = ConditionalGenerator(device=device).to(device).eval()
#cGAN_model_pth = f"./results/{target_dataset}/{targetName}/weight/trainG/"
#cGAN_model_pth = f"./results/{target_dataset}/{targetName}/weight/cGAN.pt"
#cGAN_model.load_state_dict(torch.load(cGAN_model_pth, map_location=device))
#print(f"{cGAN_model_pth=}")


best_mae, best_rmse, best_R ,best_epoch= float('inf'), float('inf'), float('-inf'), float('inf')



norm_psd = CalculateNormPSD(Fs=30, high_pass=40, low_pass=250)


# 計算模型參數數量
rPPG_model_total_params = sum(p.numel() for p in rPPG_model.parameters() if p.requires_grad)
log.info(f"rPPG_model Total Trainable Parameters: {rPPG_model_total_params}")

cGAN_model_total_params = sum(p.numel() for p in cGAN_model.parameters() if p.requires_grad)
log.info(f"cGAN_model Total Trainable Parameters: {cGAN_model_total_params}")
# 計算總參數量
total_params = rPPG_model_total_params + cGAN_model_total_params
log.info(f"Total Trainable Parameters: {total_params}")

# 計算 FLOPs
# 計算 rPPG_model GFLOPs
if profile_macs is not None:
    dummy_input = torch.randn(4, 3, args.test_seq, 64, 64).to(device)
    flops_rPPG = profile_macs(rPPG_model, dummy_input)
    log.info(f"rPPG_model GFLOPs: {flops_rPPG / 1e9:.3f}")

# 計算 cGAN_model GFLOPs
#dummy_rPPG, _ = rPPG_model(dummy_input)  # 取得 rPPG 輸出
    dummy_input_cGAN = torch.randn(4, args.test_seq).to(device)
    flops_cGAN = profile_macs(cGAN_model,dummy_input_cGAN)
    log.info(f"cGAN_model GFLOPs: {flops_cGAN / 1e9:.3f}")

# 總 GFLOPs
    total_gflops = (flops_rPPG + flops_cGAN) / 1e9
    log.info(f"Total Method GFLOPs: {total_gflops:.3f}")
else:
    log.info("torchprofile is not installed; skipping FLOPs logging.")

"""dummy_input = torch.randn(4, 3, test_seq, 64, 64).to(device) #是用兩秒算嗎?
log.info(f"Input shape: {dummy_input.shape}")
total_flops = profile_macs(model, dummy_input)
total_gflops = total_flops / 1e9  # 轉換為 GFLOPs
log.info(f"Total GFLOPs: {total_gflops:.3f}")"""
test_seq = args.test_seq
print(f"test_seq : {test_seq}")

with torch.no_grad():

    for epoch in range (0,args.epoch-1):   
        
        start_time = time.time()
        
        epoch_time = 0
        epoch_frame = 0

        model_dir = os.path.join(get_result_dir(args, target_dataset, targetName), "weight", "trainG")
        cGAN_model_pth = os.path.join(model_dir, f"adapt_cGAN{epoch:03d}.pt")
        if os.path.exists(rPPG_model_pth) and os.path.exists(cGAN_model_pth):
            cGAN_model.load_state_dict(torch.load(cGAN_model_pth, map_location=device))
        else:
            continue
        all_mae = []
        all_rmse = []
        all_R = []
        
        
        for step, (face_frames, _, _, ppg_label, subjects) in enumerate(test_loader):
            
            imgs = face_frames
            label_PPG = ppg_label
            
            hr_predicts = []
            hr_labels = []
            for i in range(0, imgs.shape[2] - seq_len + stride, stride):

                print(f"{i=}, {imgs.shape=}")
                
                _imgs = imgs[:, :, i:i + test_seq, :, :]

                _imgs = _imgs.to(device)

                print(f"{_imgs.shape=}")

                _label = label_PPG[:, i:i + seq_len].detach().cpu().numpy()

                output_fg, rPPG_middle_2s = rPPG_model.forward(_imgs)
         
                rppg = output_fg[:, -1]  # get rppg 

                
                #print(f"{rppg.shape=}")
                #exit()

                #generator output
                rppg_feature_2t10, _, _, _ = cGAN_model(ppg_sig = rppg,inputlen=test_seq, tar=300)
                #print("rppg_feature_2t10", rppg_feature_2t10.shape)
                rPPG_2t10 = rPPG_model.forward_cGAN(rppg_feature_2t10)

                

                rPPG_anc_2t10 = rPPG_2t10[:, -1]

                #rPPG_anc_2t10 =rppg #add for 10s

                _label = butter_bandpass(_label, lowcut=0.6, highcut=4, fs=args.fps)
                rPPG_anc_2t10 = butter_bandpass(rPPG_anc_2t10.detach().cpu().numpy(), lowcut=0.6, highcut=4, fs=args.fps)
                
                # predict heart rate
                hr_label = predict_heart_rate(_label.copy(), Fs=args.fps)
                hr_2t10 = predict_heart_rate(rPPG_anc_2t10.copy(), Fs=args.fps)

                hr_predicts.append(hr_2t10)
                hr_labels.append(hr_label)
        
            epoch_frame += test_seq*num_batch_per_sample
            hr_predicts = np.array(hr_predicts)
            hr_labels = np.array(hr_labels)

            print(f"{hr_predicts=}")
            print(f"{hr_labels=}")

            mae = np.mean(np.abs(hr_predicts - hr_labels))
            rmse = np.sqrt(np.mean((hr_predicts - hr_labels) ** 2))

            pearson_corr = Pearson_np(hr_predicts, hr_labels)
    
            logging.info('[epoch %d step %d mae %.5f rmse %.5f pearson_corr %.5f subjeccts %s]' 
                        % (epoch, step, mae, rmse, pearson_corr, subjects))
            
            all_mae.append(mae)
            all_rmse.append(rmse)
            all_R.append(pearson_corr)

        logging.info('[epoch %d avg all_mae %.5f all_rmse %.5f all_R %.5f]'
                    % (epoch, np.mean(all_mae), np.mean(all_rmse), np.mean(all_R)))
        
        # Update best performance metrics
        if np.mean(all_mae) <= best_mae:
            best_mae = np.mean(all_mae)
            best_rmse = np.mean(all_rmse)
            best_R = np.mean(all_R)
            best_epoch = epoch
        
        end_time = time.time()
        epoch_time = end_time - start_time
        log.info(f"Testing time for epoch {epoch}: {epoch_time:.3f} seconds")
        print(f"epoch_frame: {epoch_frame}, epoch_time: {epoch_time}")    
        log.info(f"Testing time for epoch {epoch}: {epoch_time:.3f} seconds, fps: {epoch_frame / epoch_time:.3f}")

    # Log the best performance metrics
    logging.info('[epoch %d best_mae %.5f all_rmse %.5f all_R %.5f]'
                    % (best_epoch, best_mae, best_rmse, best_R)) 

