import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torchaudio

def main():
    # Find any wav file in dataset
    wav_files = glob.glob('dataset/**/*.wav', recursive=True)
    if not wav_files:
        print("No wav files found.")
        # Try finding in other directories
        wav_files = glob.glob('**/*.wav', recursive=True)
        if not wav_files:
            print("Really no wav files found.")
            return
    
    # Skip any files in .venv
    wav_files = [f for f in wav_files if '.venv' not in f and 'node_modules' not in f]
    if not wav_files:
        print("No non-venv wav files found.")
        return
        
    wav_path = wav_files[0]
    print(f"Loading {wav_path}")
    
    waveform, sample_rate = torchaudio.load(wav_path)
    
    # Ensure it's 1.0 second long
    if waveform.shape[1] != 16000:
        if waveform.shape[1] < 16000:
            waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
        else:
            waveform = waveform[:, :16000]
            
    # Compute Mel Spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=200,  # Align with Hop size
        n_mels=64
    )
    mel_spec = mel_transform(waveform)
    
    # Compute MFCC
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=40,
        melkwargs={"n_fft": 400, "hop_length": 200, "n_mels": 64}
    )
    mfcc = mfcc_transform(waveform)
    
    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Mel Spectrogram
    db_mel = 10 * np.log10(mel_spec[0].numpy() + 1e-6)
    im0 = axes[0].imshow(
        db_mel,
        origin='lower',
        aspect='auto',
        cmap='magma'
    )
    axes[0].set_title("Log-Mel Spectrogram (64 Mels)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Time Frames", fontsize=10)
    axes[0].set_ylabel("Mel Frequency Bands", fontsize=10)
    fig.colorbar(im0, ax=axes[0], label='Decibels (dB)')
    
    # MFCC
    im1 = axes[1].imshow(
        mfcc[0].numpy(),
        origin='lower',
        aspect='auto',
        cmap='magma'
    )
    axes[1].set_title("MFCC Features (40 Coefficients)", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Time Frames", fontsize=10)
    axes[1].set_ylabel("MFCC Index", fontsize=10)
    fig.colorbar(im1, ax=axes[1], label='Amplitude')
    
    plt.tight_layout()
    os.makedirs('Documentation', exist_ok=True)
    plot_path = 'Documentation/spectrogram_mfcc.png'
    plt.savefig(plot_path, dpi=200)
    print(f"Saved plot to {plot_path}")

if __name__ == '__main__':
    main()
