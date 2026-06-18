from __future__ import print_function, division
import os
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 
from torchvision import transforms
import glob
import random

from io import BytesIO
from tqdm import tqdm

from datasets.dataloader_VIPL import rPPG_Dataset as rPPG_Dataset_VIPL


START_OFFSET=30

        
class RandomApplyTransform:
    def __init__(self, crop, flip, rotate, resize, to_tensor):
        self.crop = crop
        self.flip = flip
        self.rotate = rotate
        self.resize = resize
        self.to_tensor = to_tensor

    def __call__(self, frames):
        # Apply the same random crop to all frames
        # i, j, h, w = self.crop.get_params(frames[0], self.crop.size)
        # frames = [transforms.functional.crop(frame, i, j, h, w) for frame in frames]

        # Apply the same random horizontal flip to all frames
        if random.random() > 0.5:
            frames = [transforms.functional.hflip(frame) for frame in frames]

        # Apply the same random rotation to all frames
        angle = transforms.RandomRotation.get_params(self.rotate.degrees)
        frames = [transforms.functional.rotate(frame, angle) for frame in frames]

        # Resize and convert to tensor
        frames = [self.resize(frame) for frame in frames]
        frames = [self.to_tensor(frame) for frame in frames]

        return torch.stack(frames)
    


import threading
def process_images(paths, transform_face, transform_face_aug, gray, results, results_aug, index):
    frames, frames_aug, tmp = [], [], []
    for path in paths:
        try:
            frame = Image.open(path).convert('RGB')
            if gray: frame = frame.convert('L')
            # 觸發 PIL 實際讀取數據，若損毀會在此噴錯
            frame.load() 
            tmp.append(frame)
        except Exception as e:
            # 補一張全黑圖，確保執行緒不崩潰
            print(f"Skipping broken image: {path}")
            dummy = Image.new('RGB', (64, 64), (0, 0, 0))
            if gray: dummy = dummy.convert('L')
            tmp.append(dummy)

    # 確保無論如何都會存回結果，避免 NoneType 錯誤
    results[index] = [transform_face(f) for f in tmp]
    if transform_face_aug:
        results_aug[index] = transform_face_aug(tmp)
# def process_images(paths, transform_face, transform_face_aug, gray, results, results_aug, index):
#     """Function that each thread will execute."""
#     frames = []
#     frames_aug = []
#     tmp = []

#     for path in paths:
#         frame = Image.open(path)
#         if gray:
#             frame = frame.convert('L')
        
#         tmp.append(frame)

#     frames = [transform_face(frame) for frame in tmp]
    
#     # apply the same transform to all frames
#     if transform_face_aug:
#         frames_aug = transform_face_aug(tmp)


#     results[index] = frames
#     if transform_face_aug:
#         results_aug[index] = frames_aug



def preload_frames_multithreaded(paths, transform_face, transform_face_aug=None, gray=False, num_threads=8):
    """Loads and processes frames using multiple threads."""
    thread_list = []
    results = [None] * num_threads  # List to store results from each thread
    results_aug = [None] * num_threads  # List to store results from each thread
    n = len(paths)
    images_per_thread = n // num_threads

    for i in range(num_threads):
        start_index = i * images_per_thread
        end_index = start_index + images_per_thread
        if i == num_threads - 1:  # Handle the last batch
            end_index = n
        thread = threading.Thread(target=process_images, 
                                  args=(paths[start_index:end_index], transform_face, transform_face_aug, gray, results, results_aug, i))
        
        thread_list.append(thread)
        thread.start()

    for thread in thread_list:
        thread.join()

    # Combine the results from all threads
    all_frames = []
    all_frames_aug = []
    for frames in results:
        all_frames.extend(frames)
        
    if transform_face_aug:
        for frames_aug in results_aug:
            all_frames_aug.extend(frames_aug)
    
    return all_frames, all_frames_aug



def get_rPPG(path):

    f = open(path, 'r')
    lines = f.readlines()
    PPG = [float(ppg) for ppg in lines[0].split()]
    # hr = [float(ppg) for ppg in lines[1].split()[:100]]
    # no = [float(ppg) for ppg in lines[2].split()[:100]]
    f.close()

    return PPG



def getSubjects(configPath):
    
    f = open(configPath, "r")
    
    all_live, all_spoof = [], []
    ls_dict = {}
    
    while(True):
        line = f.readline()
        if not line:
            break
        line = line.strip()
        # print(line)
        
        subj, ls = line.split(",")
        if(ls == "+1"):
            all_live.append(subj)
            ls_dict[subj] = 1
            # print("live", subj)
        else:
            all_spoof.append(subj)
            ls_dict[subj] = 0
            # print("spoof", subj)
    
    print(f"{configPath=}")
    print(f"{len(all_live)=}, {len(all_spoof)=}")
    
    return all_live, all_spoof, ls_dict


class rPPG_Dataset(Dataset):
    def __init__(self, datasets, seq_length, train, if_bg=True, real_or_fake="both",
                 num_batch_per_sample=1, if_preload=False, if_aug=True, testFold=0,
                 data_root="./data", cache_dir="./cache/preprocessed", vipl_root=None,
                 vipl_gt_root=None, vipl_bg_root=None):
        
        
        assert real_or_fake in ["real", "fake", "both"]
        
        self.train = train
        self.if_bg = if_bg
        self.if_aug = if_aug
        self.if_preload = if_preload
        self.rPPG_dataset = ["C", "P", "U", "M", "V"]
        self.num_batch_per_sample = num_batch_per_sample
        
        datasets = datasets.split(',')
        self.datasets = datasets
        print(f"{datasets=}")

        
        face_folder = "crop_MTCNN"
        prefix = data_root
        self.root_dir = {
            "C" : f"{prefix}/COHFACE/{face_folder}_30fps",
            "P" : f"{prefix}/pure/{face_folder}",
            "U" : f"{prefix}/UBFC/{face_folder}",
            "M" : f"{prefix}/MR-NIRP/NIR_{face_folder}",
            "H1" : f"{prefix}/HKBU_MARs_V1+/{face_folder}",
            "H2" : f"{prefix}/HKBU_MARs_V2/{face_folder}",
            "3" : f"{prefix}/3DMAD/{face_folder}",
            "V" : "",
        }
        
        bg_folder = "bg_MTCNN"
        self.root_bg_dir = {
            "C" : f"{prefix}/COHFACE/{bg_folder}_30fps",
            "P" : f"{prefix}/pure/{bg_folder}",
            "U" : f"{prefix}/UBFC/{bg_folder}",
            "M" : f"{prefix}/MR-NIRP/NIR_MiDaS",
            "H1" : f"{prefix}/HKBU_MARs_V1+/{bg_folder}",
            "H2" : f"{prefix}/HKBU_MARs_V2/{bg_folder}",
            "3" : f"{prefix}/3DMAD/{bg_folder}",
            "V" : "",
        }
        
        type = "train"
        if not train:
            type = "test"

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
        self.subject_file = {
            "C" : os.path.join(repo_root, "config", f"COHFACE_{type}.txt"),
            "P" : os.path.join(repo_root, "config", f"PURE_{type}.txt"),
            "U" : os.path.join(repo_root, "config", f"UBFC_{type}.txt"),
            "M" : os.path.join(repo_root, "config", f"MR-NIRP_{type}.txt"),
            "H1" : os.path.join(repo_root, "config", f"HKBU_MARs_V1+_{type}.txt"),
            "H2" : os.path.join(repo_root, "config", f"HKBU_MARs_V2_{type}.txt"),
            "3" : os.path.join(repo_root, "config", f"3DMAD_{type}.txt"),
            "V" : "",
        }
        
        self.subjects= {}
        
        self.preloaded_subject_images = {}
        self.preloaded_bg_images = {}
        self.subject_GT_PPG = {}
        
        
        if self.train:
            _h, _w = 64, 64
        else:
            _h, _w = 64, 64

        self.transform_face = transforms.Compose([
            transforms.Resize((_h, _w)),
            transforms.ToTensor()
        ])
        
        self.transform_face_aug = RandomApplyTransform(
            transforms.RandomCrop(100),        # Randomly crop a 50x50 portion of the image
            transforms.RandomHorizontalFlip(), # Randomly flip the images horizontally
            transforms.RandomRotation(180),    # Randomly rotate between -180 and +180 degrees
            transforms.Resize((_h, _w)),
            transforms.ToTensor()
        )


        # Adjust the initialization to preload images into memory
        for key in datasets:
            assert key in self.root_dir
            
            
            
            if key == "V":
                VIPL_dataset = rPPG_Dataset_VIPL(datasets="V",
                                                 seq_length=seq_length,
                                                 train=train,
                                                 if_bg=if_bg,
                                                 testFold=testFold,
                                                 num_batch_per_sample=num_batch_per_sample,
                                                 vipl_root=vipl_root,
                                                 vipl_gt_root=vipl_gt_root,
                                                 vipl_bg_root=vipl_bg_root)
            
                self.preloaded_subject_images.update(VIPL_dataset.subject_images)
                self.subject_GT_PPG.update(VIPL_dataset.subject_GT_PPG)
                
                # --- 補上這段：確保背景字典裡有 VIPL 的 key，即便它是空的 ---
                for v_key in VIPL_dataset.subject_images.keys():
                    self.preloaded_bg_images[v_key] = [] 
                
                continue
            
            
            if key not in self.rPPG_dataset:
                all_real, all_fake, label_dict = getSubjects(self.subject_file[key])
                self.label_dict = label_dict
                if real_or_fake == "real":
                    self.subjects[key] = all_real
                elif real_or_fake == "fake":
                    self.subjects[key] = all_fake
                elif real_or_fake == "both":
                    self.subjects[key] = all_real + all_fake
            else:
                
                with open(self.subject_file[key]) as file:
                    self.subjects[key] = [line.rstrip() for line in file]
                    print(self.subjects[key])
            

            print(f"Loading {key} dataset, if_bg={if_bg}, real_or_fake={real_or_fake}, if_preload={if_preload}")
            for _subject in tqdm(self.subjects[key]):
                def get_files(path, subj):
                    files = []
                    for ext in ('*.png', '*.jpg'):
                        files.extend(sorted(glob.glob(os.path.join(path, subj, ext))))
                    return files
                
                image_paths = get_files(self.root_dir[key], _subject)
                bg_paths = get_files(self.root_bg_dir[key], _subject)
                # print(_subject, len(image_paths), len(bg_paths))
                
                _key = f"{key}_{_subject}"
                
                
                if self.if_preload:
                    # Preload images and store them into the dictionary
                    gray = True if key == "M" else False
                    prefix = cache_dir
                    if not os.path.exists(f'{prefix}/{_key}_fg.pt'):
                        self.preloaded_subject_images[_key], _ = preload_frames_multithreaded(image_paths, self.transform_face, gray=gray)
                        os.makedirs(prefix, exist_ok=True)
                        torch.save(self.preloaded_subject_images[_key], f'{prefix}/{_key}_fg.pt')
                    else:
                        self.preloaded_subject_images[_key] = torch.load(f'{prefix}/{_key}_fg.pt')
                    
                    
                    if self.if_bg:
                        if not os.path.exists(f'{prefix}/{_key}_bg.pt'):
                            self.preloaded_bg_images[_key], _ = preload_frames_multithreaded(bg_paths, self.transform_face, gray=gray)
                            os.makedirs(prefix, exist_ok=True)
                            torch.save(self.preloaded_bg_images[_key], f'{prefix}/{_key}_bg.pt')
                        else:
                            self.preloaded_bg_images[_key] = torch.load(f'{prefix}/{_key}_bg.pt')
                
                else:
                    self.preloaded_subject_images[_key] = image_paths
                    self.preloaded_bg_images[_key] = bg_paths
                
                        
                
                if key in self.rPPG_dataset:
                    ground_truth = os.path.join(self.root_dir[key], _subject, "ground_truth.txt")
                    self.subject_GT_PPG[_key] = get_rPPG(ground_truth)
                else:
                    self.subject_GT_PPG[_key] = []
        
        
        self.all_keys = list(self.preloaded_subject_images.keys())
        self.seq_length = seq_length

        
    def __getitem__(self, idx):
        """修正後的 getitem：自動判斷路徑或 Tensor"""
        _key = self.all_keys[idx]
        domain, subject = _key.split("_", 1)
        _face_frame, _bg_frame, _ls_label, _ppg = [], [], [], []
        _face_frame_aug, _bg_frame_aug = [], []
        
        # 取得原始資料（可能是 Tensor 列表或路徑列表）
        raw_face_data = self.preloaded_subject_images[_key]
        total_frame = len(raw_face_data)
        
        start = START_OFFSET if not self.train else random.randint(START_OFFSET, 
                                                                   max(START_OFFSET, 
                                                                       total_frame - self.seq_length)
                                                                   )

        num_batch = min(max(total_frame // self.seq_length, 1), self.num_batch_per_sample) if "V" in self.datasets else self.num_batch_per_sample
        
        end_idx = start + self.seq_length * num_batch
        face_sample = raw_face_data[start:end_idx]
        gray = True if domain == "M" else False

        # --- 臉部影像讀取核心邏輯 ---
        # 檢查抓到的是不是路徑字串
        if isinstance(face_sample[0], str):
            # 如果是路徑 (VIPL 會走這裡)，呼叫讀取函式
            trans_aug = self.transform_face_aug if self.if_aug else None
            _face_list, _face_aug_list = preload_frames_multithreaded(face_sample, self.transform_face, trans_aug, gray=gray)
            _face_frame = torch.stack(_face_list).transpose(0, 1)
            if self.if_aug and _face_aug_list:
                _face_frame_aug = torch.stack(_face_aug_list).transpose(0, 1)
        else:
            # 如果是已經 preload 好的 Tensor
            _face_frame = torch.stack(face_sample).transpose(0, 1)

        # --- 背景影像讀取 (加上 VIPL 安全檢查) ---
        if self.if_bg:
            raw_bg_data = self.preloaded_bg_images.get(_key, [])
            if len(raw_bg_data) > 0:
                bg_sample = raw_bg_data[start:end_idx]
                if isinstance(bg_sample[0], str):
                    _bg_list, _ = preload_frames_multithreaded(bg_sample, self.transform_face, None, gray=gray)
                    _bg_frame = torch.stack(_bg_list).transpose(0, 1)
                else:
                    _bg_frame = torch.stack(bg_sample).transpose(0, 1)
            else:
                # VIPL 沒背景，給全黑 Tensor
                _bg_frame = torch.zeros_like(_face_frame)

        # --- 標籤讀取 ---
        if domain in self.rPPG_dataset:
            _ppg = torch.FloatTensor(self.subject_GT_PPG[_key][start:end_idx])
            _ls_label = torch.tensor(1).unsqueeze(0)
        else:
            _ppg = torch.FloatTensor([0] * (end_idx - start))
            _ls_label = torch.tensor(self.label_dict.get(subject, 0)).unsqueeze(0)

        # --- Padding 邏輯 ---
        target_len = self.seq_length * num_batch
        if _face_frame.shape[1] < target_len:
            pad_val = target_len - _face_frame.shape[1]
            _face_frame = F.pad(_face_frame, (0, 0, 0, 0, 0, pad_val))
            _ppg = F.pad(_ppg, (0, pad_val))
            if self.if_aug and len(_face_frame_aug) > 0:
                _face_frame_aug = F.pad(_face_frame_aug, (0, 0, 0, 0, 0, pad_val))
            if self.if_bg:
                _bg_frame = F.pad(_bg_frame, (0, 0, 0, 0, 0, pad_val))

        if self.if_aug and len(_face_frame_aug) > 0:
            _face_frame = torch.cat([_face_frame, _face_frame_aug], dim=1)
        
        return _face_frame, _bg_frame, _ls_label, _ppg, _key


    def __len__(self):
        return len(self.all_keys)
    
    
    
def get_loader(_datasets, _seq_length, batch_size=1, shuffle=True, train=True, if_bg=True,
               real_or_fake="both", num_batch_per_sample=1, if_preload=False, if_aug=False,
               testFold=0, data_root="./data", cache_dir="./cache/preprocessed",
               vipl_root=None, vipl_gt_root=None, vipl_bg_root=None):
    
    _dataset = rPPG_Dataset(datasets=_datasets, 
                            seq_length=_seq_length,
                            train=train,
                            if_bg=if_bg,
                            real_or_fake=real_or_fake,
                            num_batch_per_sample=num_batch_per_sample,
                            if_preload=if_preload,
                            if_aug=if_aug,
                            testFold=testFold,
                            data_root=data_root,
                            cache_dir=cache_dir,
                            vipl_root=vipl_root,
                            vipl_gt_root=vipl_gt_root,
                            vipl_bg_root=vipl_bg_root)

    # num_id, num_domain = _dataset.get_id_domain_num()
    
    return DataLoader(_dataset, batch_size=batch_size, shuffle=shuffle)



if __name__ == "__main__":
    
    from einops import rearrange
    import numpy as np
    
    def saveFrames(frames):
        
        os.makedirs("test_dataloader", exist_ok=True)
        # save tensor to images
        for i in range(frames.shape[2]):
            img = frames[0, :, i, :, :]
            # print(img.shape)
            # print(torch.max(img), torch.min(img))
            
            img = img.permute(1, 2, 0)
            img = img.cpu().numpy()
            img = (img*255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(f"test_dataloader/{i}.jpg", img)

    
    train_loader = get_loader(_datasets="U,P,V",
                              _seq_length=10*30,
                              batch_size=3,
                              train=True,
                              if_bg=False,
                              real_or_fake="real",
                              if_preload=False,
                              if_aug=False)
    
    test_loader = get_loader(_datasets="U,P,V",
                              _seq_length=10*30,
                              batch_size=1,
                              train=False,
                              if_bg=False,
                              real_or_fake="both",
                              if_preload=False,
                              if_aug=False,
                              num_batch_per_sample=3)

    print(f"{len(train_loader)=}")
    for step, (face_frames, bg_frames, ls_label, ppg_label, subjects) in enumerate(train_loader):
        print(f"{face_frames.shape=}")
        # print(f"{bg_frames.shape=}")
        print(f"{ls_label=}")
        # print(f"{ppg_label.shape=}")
        print(subjects)
        
        seq_len=300
        # face_frames_ori = face_frames[:, :, :seq_len]
        # face_frames_aug = face_frames[:, :, seq_len:]
        # print(f"{face_frames.shape=}, {face_frames_ori.shape=}, {face_frames_aug.shape=}")
        
            
        break
        
    
    print(f"{len(test_loader)=}")
    for step, (face_frames, bg_frames, ls_label, ppg_label, subjects) in enumerate(test_loader):
        print(f"{face_frames.shape=}")
        print(f"{ls_label=}")
        # print(f"{ppg_label.shape=}")
        print(subjects)
        # break
