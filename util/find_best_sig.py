
import torch
import torch.nn as nn
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
from loss import *
def generate_ppg_torch(bpm, fps=30, alpha=0.25, seq_len=300, phase=None, amplitude=None, device='cpu'):
    t = torch.linspace(0, seq_len/fps, seq_len, device=device)
    omega = 2 * np.pi * (bpm / 60)
    S = amplitude * torch.sin(omega * t + phase) + 0.1 * amplitude * torch.sin(2 * omega * t + phase)
    return S

def best_syn_ppg(hr_rppg_gt, ppg_label, fps=30, alpha=0.25, seq_len=300, num_phases=300, num_amplitudes=10, device='cuda'):
    phases = torch.linspace(0, 2 * np.pi, num_phases, device=device)
    #amplitudes = torch.linspace(1.0, 2.0, num_amplitudes, device=device)
    amplitudes = torch.tensor([1.5], device=device)
    best_signals = torch.zeros_like(ppg_label, device=device)

    for i, bpm in enumerate(hr_rppg_gt):
        best_corr = float('-inf')
        best_signal = None
        for phase in phases:
            for amplitude in amplitudes:
                syn_ppg = generate_ppg_torch(bpm=bpm, fps=fps, alpha=alpha, seq_len=seq_len, phase=phase, amplitude=amplitude, device=device)
                corr = torch.corrcoef(torch.stack([syn_ppg, ppg_label[i].to(device)]))[0, 1]
                if corr > best_corr:
                    best_corr = corr
                    best_signal = syn_ppg
        best_signals[i] = best_signal
    return best_signals

def shift_window(sig1,sig2, NCC_loss):

    if sig1.shape[1] >= sig2.shape[1]:
        long_sig = sig1
        short_sig = sig2
    else:
        long_sig = sig2
        short_sig = sig1

    count = long_sig.shape[1] - short_sig.shape[1]
    cut=short_sig.shape[1]

    ncc_pre = []
    for i in range(0,count+1):
        long_sig_shift=long_sig[:,i:i+cut]#0,1,2,...,1/61/121/181/241   0-240~60-300 
        if long_sig_shift.shape[1]!=cut:
            raise ValueError(f"syn_ppg_shift shape is not correct {long_sig_shift.shape}.")
        
        ncc_val_pre, best_ncc_pre = NCC_loss(long_sig_shift, short_sig)
        ncc_pre.append(best_ncc_pre)
    #ncc_label = torch.stack(ncc_label).transpose(0, 1)  # Shape: [batch_size, shift]
    ncc_pre = torch.stack(ncc_pre).transpose(0, 1)  # Shape: [batch_size, shift]

    peaks = cus_find_peaks_batch(ncc_pre) #return list
    loss_ncc_clip = process_peak_values(ncc_pre, peaks)
    loss_ncc_clip = loss_ncc_clip.mean()
    return loss_ncc_clip


"""def cus_find_peaks_batch(x, threshold=0, min_distance=8):
    batch_size, n_samples = x.shape
    all_peaks = []  # List to store the peaks for each sample in the batch

    if n_samples == 1:
        for b in range(batch_size):
            all_peaks.append(torch.tensor([0], device=x.device))
        return all_peaks
    sample_b=x.detach().cpu().numpy()
    for b in range(batch_size):
        sample = sample_b[b]  # Convert to numpy array for scipy function
        peaks, _ = find_peaks(sample, height=np.mean(sample), distance=min_distance)
        all_peaks.append(torch.tensor(peaks, device=x.device))  # Ensure the tensor is on the same device
    return all_peaks"""

def segment_find_peak(x):
    segment_length = 60
    num_segments = (len(x) - 1) // segment_length
    peak_f = []

    #print(f"num_segments: {num_segments}")

    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        if i==num_segments-1:
            end_idx=max((start_idx + segment_length),x.shape[0])
        peak_f.append(np.argmax(x[start_idx:end_idx]) + start_idx)

    return np.array(peak_f)

def cus_find_peaks_batch(x, threshold=0, min_distance=8):
    batch_size, n_samples = x.shape
    all_peaks = []  # List to store the peaks for each sample in the batch
    
    if n_samples == 1:
        for b in range(batch_size):
            all_peaks.append(torch.tensor([0], device=x.device))
        return all_peaks
    sample_b=x.detach().cpu().numpy()
    for b in range(batch_size):
        sample = sample_b[b]  # Convert to numpy array for scipy function
        #peaks, _ = find_peaks(sample, height=np.mean(sample), distance=min_distance)
        peaks = segment_find_peak(sample)
        all_peaks.append(torch.tensor(peaks, device=x.device)) 
    return all_peaks


def process_peak_values(ncc_pre, peaks): 
    # Filter out the indices with empty peaks
    non_empty_indices = [i for i, peak_indices in enumerate(peaks) if len(peak_indices) > 0]
    filtered_ncc_pre = ncc_pre[non_empty_indices]
    filtered_peaks = [peaks[i] for i in non_empty_indices]
    
    results = torch.empty(len(filtered_peaks))  # Initialize an empty tensor to store results
    
    for i, peak_indices in enumerate(filtered_peaks):
        peak_values = filtered_ncc_pre[i, peak_indices]  # Retrieve the values at the peak indices
        modified_values = 1 - peak_values  # Compute 1 - value for each peak
        average_value = modified_values.sum() / len(peak_indices)  # Sum and average the modified values
        results[i] = average_value  # Store in the results tensor

    return results

#draw the normalize signal
def normalize_signal(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)
    # Normalize the signal to the range [-1, 1]
    return 2 * (signal - min_val) / (max_val - min_val) - 1 if max_val != min_val else signal

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def compute_power_spectrum_torch(signal, Fs, zero_pad=None, high_pass=40, low_pass=250):
    # Assuming signal is a PyTorch tensor
    if zero_pad is not None:
        L = signal.size(0)
        padding = int(zero_pad / 2 * L)
        signal = torch.nn.functional.pad(signal, (padding, padding), 'constant', 0)

    # Compute the FFT frequencies
    freqs = torch.fft.fftfreq(signal.size(0), 1 / Fs) * 60  # in bpm

    # Compute the FFT and power spectrum
    ps = torch.abs(torch.fft.fft(signal))**2

    # Only keep the positive frequencies (one-sided spectrum)
    cutoff = len(freqs) // 2
    freqs = freqs[:cutoff]
    ps = ps[:cutoff]
    
    valid_freqs = (freqs >= high_pass) & (freqs <= low_pass)
    freqs = freqs[valid_freqs]
    ps = ps[valid_freqs]
    
    return freqs, ps

def PSD_entropy(x: torch.Tensor) -> torch.Tensor:
    # (B, spatial_window^2, T) -> (B * spatial_window^2, T)    [6, 5, T]->[30, T]
    x = x.view((x.size(0) * x.size(1), x.size(2)))
    psd = []
    peaks = []
    for i in range(x.size(0)):
        freqs, ps = compute_power_spectrum_torch(x[i], 30, zero_pad=100)
        psd.append(ps)

        # Find the peak value
        peak_value = torch.max(ps)
        peaks.append(peak_value.item())

    psd = torch.stack(psd)
    entropy = softmax_entropy(psd)

    return entropy

