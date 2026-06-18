import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.PhysNetModel import PhysNet
from util import *
import argparse

from datasets.dataloader import get_loader
from loss import *
import shutil


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
    
name=args.name
trainName, testName, finetuneName = get_name(args, finetune=finetune, model_name=PhysNet.__name__)
log = get_logger(get_log_dir(args, "test_label", args.test_dataset), testName)

print(f"{trainName=}")
print(f"{finetuneName=}")
print(f"{testName=}")

targetName = trainName
if finetune:
    target_dataset = args.finetune_dataset
    targetName = finetuneName
    

#args.train_T=10
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

model_path = os.path.join(get_result_dir(args, target_dataset, targetName), "weight")

model_fg = PhysNet(S=args.model_S,
                in_ch=args.in_ch,
                conv_type=args.conv,
                seq_len=seq_len, 
                delta_t=args.delta_T*args.fps, 
                numSample=args.numSample,
                class_num=2).to(device).eval()


best_mae, best_rmse, best_R ,best_epoch= float('inf'), float('inf'), float('-inf'), float('inf')

test_seq = args.test_seq
logging.info("Now test clip length is: %s", test_seq)

#norm_psd = CalculateNormPSD(Fs=30, high_pass=40, low_pass=250)
with torch.no_grad():

    for epoch in range (0,args.epoch):
        
        
        #model_fg.load_state_dict(torch.load(model_path + '/fg_epoch%d.pt' % (epoch), map_location=device))  # load weights to the model
        model_state = torch.load(model_path + f'/fg_epoch{epoch:03d}.pt', map_location=device)
        model_fg.load_state_dict(model_state)

        all_mae = []
        all_rmse = []
        all_R = []
        
        for step, (face_frames, _, _, ppg_label, subjects) in enumerate(test_loader):
            
            imgs = face_frames
            label_PPG = ppg_label
            
            hr_predicts = []
            hr_labels = []
            for i in range(0, imgs.shape[2] - seq_len + stride, stride):

                #_imgs = imgs[:, :, i:i + seq_len, :, :].to(device)
                clip_start = i + max((seq_len - test_seq) // 2, 0)
                _imgs = imgs[:, :, clip_start:clip_start + test_seq, :, :].to(device)
                _label = label_PPG[:, i:i + seq_len].detach().cpu().numpy()

                output_fg,_ = model_fg.forward(_imgs)
                    
                    
                rppg = output_fg[:, -1]  # get rppg 
                rppg = rppg[0].detach().cpu().numpy()
                rppg = butter_bandpass(rppg, lowcut=0.6, highcut=4, fs=args.fps)

                _label = butter_bandpass(_label, lowcut=0.6, highcut=4, fs=args.fps)

                hr_rppg = predict_heart_rate(rppg.copy(), Fs=args.fps)
                hr_label = predict_heart_rate(_label, Fs=args.fps)

                hr_predicts.append(hr_rppg)
                hr_labels.append(hr_label)

            hr_predicts = np.array(hr_predicts)
            hr_labels = np.array(hr_labels)

            #print(f"{hr_predicts=}")
            #print(f"{hr_labels=}")

            mae = np.mean(np.abs(hr_predicts - hr_labels))
            rmse = np.sqrt(np.mean((hr_predicts - hr_labels) ** 2))

            pearson_corr = Pearson_np(hr_predicts, hr_labels)
    
            logging.info('[epoch %d step %d mae %.5f rmse %.5f pearson_corr %.5f]' 
                        % (epoch, step, mae, rmse, pearson_corr))
            
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

    # Log the best performance metrics
    if best_epoch is not None :
        # Copy the best model files
        logging.info(f"save the model at epoch {best_epoch}")
        best_model_path = os.path.join(model_path, f'fg_epoch{best_epoch:03d}.pt')
        best_opt_path = os.path.join(model_path, f'fg_opt_epoch{best_epoch:03d}.pt')

        shutil.copy(best_model_path, os.path.join(model_path, 'fg_epoch_best.pt'))
        shutil.copy(best_opt_path, os.path.join(model_path, 'fg_opt_epoch_best.pt'))

    logging.info('[epoch %d best_mae %.5f all_rmse %.5f all_R %.5f]'
                    % (best_epoch, best_mae, best_rmse, best_R)) 

            # break
        # break
    #  
