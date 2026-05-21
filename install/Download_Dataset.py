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
    print("Checking for existing dataset archive...")
    
    if True:
        print(f"{ZIP_FILE} not found. Starting download process.")
        print("To proceed, you need a Kaggle account and an API key.")
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
            print(f"Authentication failed. Error: {e}")
            sys.exit(1)

        print(f"\nDownloading dataset (~1.4GB)...")
        try:
            api.dataset_download_cli(DATASET, path=".", unzip=False)
            slug_zip = DATASET.split('/')[-1] + ".zip"
            if os.path.exists(slug_zip) and slug_zip != ZIP_FILE:
                os.rename(slug_zip, ZIP_FILE)
        except Exception as e:
            print(f"Download failed. Error: {e}")
            sys.exit(1)
    else:
        print(f"Found {ZIP_FILE}. Skipping download and authentication.")

    print("\nExtracting specified keywords and background noise...")
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    try:
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            targets = KEYWORDS + ["_background_noise_"]
            files_to_extract = [
                f for f in zip_ref.namelist() 
                if any(f.startswith(f"{word}/") for word in targets)
            ]
            
            if not files_to_extract:
                print("Warning: No files matched the specified keywords.")
            else:
                zip_ref.extractall(path=TARGET_DIR, members=files_to_extract)
    except Exception as e:
        print(f"Extraction failed. Error: {e}")
        #sys.exit(1)

    # Move README.md from background noise folder to dataset root
    readme_src = os.path.join(TARGET_DIR, "_background_noise_", "README.md")
    readme_dst = os.path.join(TARGET_DIR, "README.md")
    if os.path.exists(readme_src):
        print("Moving README.md to dataset root...")
        shutil.move(readme_src, readme_dst)

    print("Cleaning up temporary files...")
    if os.path.exists(ZIP_FILE):
        try:
            os.remove(ZIP_FILE)
        except PermissionError:
            print(f"Warning: Could not delete {ZIP_FILE}. It is still in use.")
        

    print(f"Success. The dataset is ready in the '{TARGET_DIR}' directory.")

if __name__ == "__main__":
    main()