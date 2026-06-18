from __future__ import print_function, division

import glob
import os
import random
import threading

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


START_OFFSET = 30


def process_images(paths, transform_face, gray, results, index):
    frames = []
    for path in paths:
        frame = Image.open(path)
        if gray:
            frame = frame.convert("L")
        frames.append(transform_face(frame))
    results[index] = frames


def get_frame(paths, transform_face, gray=False, num_threads=8):
    if not paths:
        return []

    num_threads = min(num_threads, len(paths))
    results = [None] * num_threads
    thread_list = []
    images_per_thread = max(1, len(paths) // num_threads)

    for i in range(num_threads):
        start_index = i * images_per_thread
        end_index = len(paths) if i == num_threads - 1 else start_index + images_per_thread
        thread = threading.Thread(
            target=process_images,
            args=(paths[start_index:end_index], transform_face, gray, results, i),
        )
        thread_list.append(thread)
        thread.start()

    for thread in thread_list:
        thread.join()

    all_frames = []
    for frames in results:
        if frames is not None:
            all_frames.extend(frames)
    return all_frames


def get_rPPG(path):
    with open(path, "r") as f:
        lines = f.readlines()
    return [float(ppg) for ppg in lines[0].split()]


class rPPG_Dataset(Dataset):
    def __init__(self, datasets, seq_length, train, if_bg=True, num_batch_per_sample=1,
                 testFold=1, vipl_root=None, vipl_gt_root=None, vipl_bg_root=None):
        self.train = train
        self.if_bg = if_bg
        self.num_batch_per_sample = num_batch_per_sample

        self.root_dir = vipl_root or os.path.join("data", "VIPL", "RGB_crop")
        self.root_bg_dir = vipl_bg_root or os.path.join("data", "VIPL", "VIPL-HR_MiDaS")
        self.gt_root = vipl_gt_root or os.path.join("data", "VIPL", "GT")

        self.seq_length = seq_length
        self.datasets = datasets
        self.subjects = {"V": []}
        self.subject_images = {}
        self.bg_images = {}
        self.subject_GT_path = {}
        self.subject_GT_PPG = {}

        allFolds = [1, 2, 3, 4, 5]
        prefix = "preprocess_data_fold"
        if testFold != 0:
            assert testFold in allFolds
            if train:
                allFolds.remove(testFold)
            else:
                allFolds = [testFold]

        print("Training fold:", allFolds)

        for fold in allFolds:
            fold_dir = os.path.join(self.root_dir, f"{prefix}{fold}")
            if not os.path.isdir(fold_dir):
                continue

            subjects = [
                f"{prefix}{fold}/{subject}"
                for subject in os.listdir(fold_dir)
                if os.path.isdir(os.path.join(fold_dir, subject))
            ]

            missing_gt_subject = "preprocess_data_fold1/p49_v2_source1"
            if fold == 1 and missing_gt_subject in subjects:
                subjects.remove(missing_gt_subject)

            self.subjects["V"].extend(subjects)

        for subject in self.subjects["V"]:
            image_paths = sorted(glob.glob(os.path.join(self.root_dir, subject, "*.png")))
            bg_paths = sorted(glob.glob(os.path.join(self.root_bg_dir, subject, "*.png")))

            key = f"V_{subject}"
            self.subject_images[key] = image_paths
            self.bg_images[key] = bg_paths

            ground_truth = os.path.join(self.gt_root, subject.split("/")[-1], "ground_truth.txt")
            self.subject_GT_path[key] = ground_truth
            self.subject_GT_PPG[key] = get_rPPG(ground_truth)

            if len(self.subject_GT_PPG[key]) < (self.seq_length + 31):
                print(f"{key} has been removed due to insufficient length ({len(self.subject_GT_PPG[key])})")
                del self.subject_GT_path[key]
                del self.subject_GT_PPG[key]
                del self.subject_images[key]
                del self.bg_images[key]

        self.all_keys = list(self.subject_images.keys())
        print(f"Total number of samples: {len(self.all_keys)}")

        self.transform_face = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        key = self.all_keys[idx]

        start = START_OFFSET if not self.train else random.randint(
            START_OFFSET,
            max(START_OFFSET, len(self.subject_images[key]) - self.seq_length),
        )

        total_frame = len(self.subject_images[key])
        num_batch = min(max(total_frame // self.seq_length, 1), self.num_batch_per_sample)
        target_len = self.seq_length * num_batch

        face_frame_paths = self.subject_images[key][start:start + target_len]
        face_frames = get_frame(face_frame_paths, self.transform_face)
        face_frame = torch.stack(face_frames).transpose(0, 1)

        if self.if_bg:
            bg_frame_paths = self.bg_images[key][start:start + target_len]
            bg_frames = get_frame(bg_frame_paths, self.transform_face)
            if bg_frames:
                bg_frame = torch.stack(bg_frames).transpose(0, 1)
            else:
                bg_frame = torch.zeros_like(face_frame)
        else:
            bg_frame = []

        ppg = torch.FloatTensor(self.subject_GT_PPG[key][start:start + target_len])

        if face_frame.shape[1] < target_len:
            pad_val = target_len - face_frame.shape[1]
            face_frame = F.pad(face_frame, (0, 0, 0, 0, 0, pad_val))
            ppg = F.pad(ppg, (0, pad_val))

        if self.if_bg and bg_frame.shape[1] < target_len:
            bg_frame = F.pad(bg_frame, (0, 0, 0, 0, 0, target_len - bg_frame.shape[1]))

        return face_frame, bg_frame, ppg, key, num_batch

    def __len__(self):
        return len(self.all_keys)


def get_loader(_datasets, _seq_length, batch_size=1, shuffle=True, train=True, if_bg=True,
               testFold=5, num_batch_per_sample=1, vipl_root=None, vipl_gt_root=None,
               vipl_bg_root=None):
    dataset = rPPG_Dataset(
        datasets=_datasets,
        seq_length=_seq_length,
        train=train,
        if_bg=if_bg,
        testFold=testFold,
        num_batch_per_sample=num_batch_per_sample,
        vipl_root=vipl_root,
        vipl_gt_root=vipl_gt_root,
        vipl_bg_root=vipl_bg_root,
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    train_loader = get_loader(
        _datasets=list("V"),
        _seq_length=300,
        batch_size=1,
        testFold=5,
        train=False,
        if_bg=False,
        num_batch_per_sample=3,
    )

    for step, (face_frames, bg_frames, ppg_labels, subjects, num_batch) in enumerate(train_loader):
        print(f"{face_frames.shape=}, {ppg_labels.shape=}, {subjects=}, {num_batch=}")
