import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

import json
def load_keywords():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        p = os.path.join(d, "config.json")
        if os.path.exists(p):
            with open(p, "r") as f: return json.load(f)["keywords"]
        d = os.path.dirname(d)
    return ["yes", "no", "up", "down"]


def analyze_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "..", "dataset"))
    classes = load_keywords()
    
    durations = []
    sample_rates = set()
    frame_counts = []

    print(f"Analyzing .wav files in {dataset_dir}...\n")

    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        
        for file in os.listdir(cls_dir):
            if file.endswith(".wav"):
                path = os.path.join(cls_dir, file)
                try:
                    info = sf.info(path)
                    durations.append(info.duration)
                    sample_rates.add(info.samplerate)
                    frame_counts.append(info.frames)
                except Exception as e:
                    print(f"Error reading {file}: {e}")

    if not durations:
        print("No .wav files found.")
        return

    durations = np.array(durations)
    frame_counts = np.array(frame_counts)

    # Prepare string output
    lines = []
    lines.append(f"Total Files Analyzed: {len(durations)}")
    lines.append(f"Unique Sample Rates: {sample_rates} Hz\n")
    
    lines.append("--- Duration Stats ---")
    lines.append(f"Min Duration:  {np.min(durations):.4f} seconds")
    lines.append(f"Max Duration:  {np.max(durations):.4f} seconds")
    lines.append(f"Mean Duration: {np.mean(durations):.4f} seconds\n")

    lines.append("--- Frame Counts (Samples) ---")
    unique_frames, counts = np.unique(frame_counts, return_counts=True)
    
    # Sort by how common they are (descending)
    sorted_indices = np.argsort(-counts)
    
    lines.append("Top 10 most common clip lengths:")
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        frames = unique_frames[idx]
        count = counts[idx]
        duration_sec = frames / 16000.0  # assuming 16kHz
        percentage = (count / len(durations)) * 100
        lines.append(f"  {frames} samples ({duration_sec:.4f}s): {count} files ({percentage:.1f}%)")

    output_text = "\n".join(lines)
    print(output_text)
    
    # Write to text file
    txt_path = os.path.join(base_dir, "dataset_statistics.txt")
    with open(txt_path, "w") as f:
        f.write(output_text)
    print(f"\nSaved statistics text to: {txt_path}")

    # Plot distribution image
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Audio Clip Durations')
    plt.xlabel('Duration (Seconds)')
    plt.ylabel('Number of Files')
    plt.grid(axis='y', alpha=0.75)
    
    # Add a vertical line for the mean
    plt.axvline(np.mean(durations), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(durations):.2f}s')
    plt.legend()
    
    img_path = os.path.join(base_dir, "dataset_distribution.png")
    plt.savefig(img_path)
    plt.close()
    print(f"Saved distribution plot to: {img_path}")

if __name__ == '__main__':
    analyze_dataset()
