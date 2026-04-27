import os
import sys
import subprocess
import getpass
import zipfile
import wave
import struct
import math
import shutil
from pathlib import Path

# Configuration
import json
def load_keywords():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        p = os.path.join(d, "config.json")
        if os.path.exists(p):
            with open(p, "r") as f: return json.load(f)["keywords"]
        d = os.path.dirname(d)
    return ["yes", "no", "up", "down", "other"]

KEYWORDS = load_keywords()
DATASET = "neehakurelli/google-speech-commands"
ZIP_FILE = "google-speech-commands.zip"
TARGET_DIR = "../dataset"

def main():
    print("This script automates downloading and extracting the dataset.")
    print("To proceed, you need a Kaggle account and an API key.")
    print("Generate your API key at: https://www.kaggle.com/settings (under 'API'). You will need to enter it later.")
    print("Alternatively, you can skip this script and download the dataset manually.")
    proceed = input("Continue with automated download? (y/n): ").strip().lower()
    
    if proceed not in ['y', 'yes']:
        print("Download aborted by user.")
        sys.exit(0)

    try:
        import importlib.util
        if importlib.util.find_spec("kaggle") is None:
            raise ImportError
    except ImportError:
        print("Installing required 'kaggle' package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])

    username = input("Enter Kaggle Username: ").strip()
    key = getpass.getpass("Enter Kaggle API Key (input will be hidden): ").strip()

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    from kaggle.api.kaggle_api_extended import KaggleApi

    try:
        print("Attempting authentication...")
        api = KaggleApi()
        api.authenticate()
        print("Authentication successful.")
    except Exception as e:
        print(f"Authentication failed. Please check your credentials. Error: {e}")
        sys.exit(1)

    print(f"\nDownloading dataset (~1.4GB)...")
    print("A progress bar will appear below shortly.")
    try:
        # Reverted to dataset_download_cli to restore the progress bar
        api.dataset_download_cli(DATASET, path=".", unzip=False)
        
        slug_zip = DATASET.split('/')[-1] + ".zip"
        if os.path.exists(slug_zip) and slug_zip != ZIP_FILE:
            os.rename(slug_zip, ZIP_FILE)
            
    except Exception as e:
        print(f"Download failed. Error: {e}")
        sys.exit(1)

    print("\nExtracting specified keywords...")
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    try:
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            files_to_extract = [
                f for f in zip_ref.namelist() 
                if any(f.startswith(f"{word}/") for word in KEYWORDS)
            ]
           
            
            if not files_to_extract:
                print("Warning: No files matched the specified keywords.")
            else:
                zip_ref.extractall(path=TARGET_DIR, members=files_to_extract)
    except FileNotFoundError:
        print(f"Error: {ZIP_FILE} not found. The download may have failed.")
        sys.exit(1)

    print("Cleaning up temporary files...")
    if os.path.exists(ZIP_FILE):
        os.remove(ZIP_FILE)
        
    print("Scanning for silent/empty background noise files to move to 'other'...")
    other_dir = os.path.join(TARGET_DIR, "other")
    os.makedirs(other_dir, exist_ok=True)
    
    dataset_path = Path(TARGET_DIR)
    moved_count = 0
    for wav_file in dataset_path.rglob("*.wav"):
        if wav_file.parent.name == "other":
            continue
            
        is_empty = False
        try:
            if wav_file.stat().st_size == 0:
                is_empty = True
            else:
                with wave.open(str(wav_file), 'rb') as w:
                    frames = w.readframes(w.getnframes())
                    sampwidth = w.getsampwidth()
                    num_samples = len(frames) // sampwidth
                    if num_samples == 0:
                        is_empty = True
                    else:
                        fmt = f"<{num_samples}h" if sampwidth == 2 else f"<{num_samples}b"
                        samples = struct.unpack(fmt, frames[:num_samples*sampwidth])
                        rms = math.sqrt(sum(s**2 for s in samples) / num_samples)
                        if rms < 50.0:
                            is_empty = True
        except Exception:
            is_empty = True
            
        if is_empty:
            # Prefix with original folder name to prevent overwriting files with the same name
            new_name = f"{wav_file.parent.name}_{wav_file.name}"
            shutil.move(str(wav_file), os.path.join(other_dir, new_name))
            moved_count += 1
            
    print(f"Moved {moved_count} silent/empty files to the 'other' dataset.")
    print(f"Success. The dataset is ready in the '{TARGET_DIR}' directory.")

if __name__ == "__main__":
    main()