import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


# -------------------------------------------------
# Utility: Spectrogram → 8-bit Image
# -------------------------------------------------
def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)

    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min + eps)

    return spec_scaled.astype(np.uint8)


# -------------------------------------------------
# Load + Pad/Cut Audio to 5 Seconds
# -------------------------------------------------
def load_audio_fixed_length(file_path, sr=44100, duration=5):
    wav, _ = librosa.load(file_path, sr=sr)

    target_length = sr * duration

    if len(wav) < target_length:
        pad_amount = target_length - len(wav)
        wav = np.pad(wav, (pad_amount // 2, pad_amount - pad_amount // 2), mode="reflect")
    else:
        wav = wav[:target_length]

    return wav, sr


# -------------------------------------------------
# Create Mel Spectrogram (dB)
# -------------------------------------------------
def get_melspectrogram_db(wav, sr, n_fft=2048, hop_length=512,
                          n_mels=128, fmin=20, fmax=8300, top_db=80):

    spec = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )

    return librosa.power_to_db(spec, top_db=top_db)


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":

    # Absolute path relative to script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(base_dir, "down3.wav")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    # Load audio once
    wav, sr = load_audio_fixed_length(filename)

    # Create spectrogram
    spec_db = get_melspectrogram_db(wav, sr)
    spec_img = spec_to_image(spec_db)

    # Plot
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(wav, sr=sr)
    plt.title("Waveform")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spec_img, sr=sr, hop_length=512,
                             x_axis='time', y_axis='mel', cmap='viridis')
    plt.title("Mel Spectrogram (Normalized Image)")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.show()