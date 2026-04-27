import os
import wave
import struct
import math
import argparse
import shutil
from pathlib import Path

##################### FIND EMPTY WAV FILES IN DATASET DIRECTORY ##########################

# Run with:
#    >>>  python find_empty_wavs.py --rms [00.0] --peak [00.0]
# if you are fine with the results, add --move to automatically relocate the files:
#    >>>  python find_empty_wavs.py --rms 50.0 --peak 500 --move

##########################################################################################

def check_empty_wavs(dataset_dir, rms_threshold=50.0, peak_threshold=500):
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"Dataset directory not found: {dataset_path.resolve()}")
        return [], []

    silent_in_keywords = []  # Category 1: Silent but NOT in 'other' folder
    sound_in_other = []      # Category 2: Has sound but IS in 'other' folder
    corrupt_files = []

    print(f"Scanning directory: {dataset_path.resolve()} for .wav files...")
    
    wav_count = 0
    for wav_file in dataset_path.rglob("*.wav"):
        wav_count += 1
        try:
            # 1. Check if file is literally 0 bytes
            if wav_file.stat().st_size == 0:
                if wav_file.parent.name != "other":
                    silent_in_keywords.append((str(wav_file), "0 bytes"))
                continue
            
            # 2. Check using wave module for valid header and 0 frames
            with wave.open(str(wav_file), 'rb') as w:
                is_other = wav_file.parent.name == "other"
                if w.getnframes() == 0:
                    if not is_other:
                        silent_in_keywords.append((str(wav_file), "0 frames"))
                    continue
            
            # 3. Check for absolute silence or near-silence (low RMS)
            with wave.open(str(wav_file), 'rb') as w:
                frames = w.readframes(w.getnframes())
                sampwidth = w.getsampwidth()
                
                num_samples = len(frames) // sampwidth
                is_other = wav_file.parent.name == "other"
                
                if num_samples == 0:
                    if not is_other:
                        silent_in_keywords.append((str(wav_file), "0 frames"))
                    continue
                
                fmt = f"<{num_samples}h" if sampwidth == 2 else f"<{num_samples}b"
                try:
                    samples = struct.unpack(fmt, frames[:num_samples*sampwidth])
                    rms = math.sqrt(sum(s**2 for s in samples) / num_samples)
                    peak = max(abs(s) for s in samples)
                    
                    is_silent = (rms < rms_threshold and peak < peak_threshold)
                    is_other = wav_file.parent.name == "other"
                    
                    if is_silent and not is_other:
                        silent_in_keywords.append((str(wav_file), f"RMS={rms:.1f}, Peak={peak}"))
                    elif not is_silent and is_other:
                        sound_in_other.append((str(wav_file), f"RMS={rms:.1f}, Peak={peak}"))
                except Exception:
                    pass
                
        except Exception as e:
            # Catch files that fail to read (corrupted header, not a real wav, etc.)
            corrupt_files.append((str(wav_file), str(e)))

    print(f"Scanned {wav_count} .wav files.")

    if not silent_in_keywords and not sound_in_other and not corrupt_files:
        print("No misplaced files found based on current thresholds.")
    
    if silent_in_keywords:
        print("\n--- CATEGORY 1: Silent Files in Keyword Folders (Delete Recommended) ---")
        for f, reason in silent_in_keywords:
            print(f"  {f} ({reason})")
            
    if sound_in_other:
        print("\n--- CATEGORY 2: Non-Silent Files in 'Other' Folder (Review Suggested) ---")
        for f, stats in sound_in_other:
            print(f"  {f} ({stats})")
            
    if corrupt_files:
        print("\n--- Corrupt/Unreadable Files ---")
        for f, err in corrupt_files:
            print(f"{f} - Error: {err}")
            
    return silent_in_keywords, sound_in_other

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find empty or silent .wav files in a dataset.")
    parser.add_argument("--dir", type=str, help="Dataset directory (defaults to ../dataset or ./dataset)")
    parser.add_argument("--rms", type=float, default=50.0, help="RMS threshold for silence (default: 10.0)")
    parser.add_argument("--peak", type=int, default=500, help="Peak threshold for silence (default: 200)")
    parser.add_argument("--move", action="store_true", help="Perform Automove operations")
    
    args = parser.parse_args()

    # Resolving path assuming script is run from inside Utils/ or project root
    current_dir = Path.cwd()
    
    if args.dir:
        dataset_dir = Path(args.dir)
    elif current_dir.name == "Utils":
        dataset_dir = current_dir.parent / "dataset"
    else:
        dataset_dir = current_dir / "dataset"
        
    print(f"Thresholds: RMS < {args.rms}, Peak < {args.peak}")
    silent_in_keywords, sound_in_other = check_empty_wavs(dataset_dir, rms_threshold=args.rms, peak_threshold=args.peak)
    
    if args.move:
        if not silent_in_keywords and not sound_in_other:
            print("\nNothing to move.")
        else:
            print("\n--- Performing Automove ---")
            # 1. Category 1: Silent in Keyword Folder -> other/prefix_name.wav
            for f_path, reason in silent_in_keywords:
                src = Path(f_path)
                keyword = src.parent.name
                new_name = f"{keyword}_{src.name}"
                dst_dir = src.parent.parent / "other"
                dst_dir.mkdir(exist_ok=True)
                dst = dst_dir / new_name
                try:
                    shutil.move(str(src), str(dst))
                    print(f"  [Cat 1] {src.name} -> other/{new_name}")
                except Exception as e:
                    print(f"  [Error] Failed to move {src.name}: {e}")

            # 2. Category 2: Sound in Other Folder -> prefix/name.wav
            for f_path, stats in sound_in_other:
                src = Path(f_path)
                filename = src.name
                parts = filename.split("_", 1)
                if len(parts) > 1 and (src.parent.parent / parts[0]).is_dir():
                    keyword = parts[0]
                    new_name = parts[1]
                    dst_dir = src.parent.parent / keyword
                    dst = dst_dir / new_name
                    try:
                        shutil.move(str(src), str(dst))
                        print(f"  [Cat 2] {filename} -> {keyword}/{new_name}")
                    except Exception as e:
                        print(f"  [Error] Failed to move {filename}: {e}")
                else:
                    print(f"  [Skip] {filename} (No valid keyword prefix found)")
    elif silent_in_keywords or sound_in_other:
        print("\nTip: Use --move to automatically relocate these files.")
