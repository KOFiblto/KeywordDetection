import os
import sys
import subprocess
import getpass
import zipfile

# Configuration
KEYWORDS = ["yes", "no", "up", "down"]
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

    # Ensure kaggle package is installed
    try:
        import kaggle
    except ImportError:
        print("Installing required 'kaggle' package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])

    # Handle credentials securely
    username = os.environ.get("KAGGLE_USERNAME")
    if not username:
        username = input("Enter Kaggle Username: ").strip()

    key = os.environ.get("KAGGLE_KEY")
    if not key:
        key = getpass.getpass("Enter Kaggle API Key (input will be hidden): ").strip()

    # Set environment variables for the Kaggle API authentication
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    # Import KaggleApi after setting environment variables
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    
    try:
        api.authenticate()
    except Exception as e:
        print(f"Authentication failed. Please check your credentials. Error: {e}")
        sys.exit(1)

    print(f"\nDownloading dataset (~2GB)...")
    print("A progress bar will appear below shortly.")
    try:
        # dataset_download_cli includes a built-in progress bar
        api.dataset_download_cli(DATASET)
    except Exception as e:
        print(f"Download failed. Error: {e}")
        sys.exit(1)

    print("\nExtracting specified keywords...")
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    try:
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            # Filter files to only extract the required keywords
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

    # Clean up the archive
    print("Cleaning up temporary files...")
    if os.path.exists(ZIP_FILE):
        os.remove(ZIP_FILE)
        
    print(f"Success. The dataset is ready in the '{TARGET_DIR}' directory.")

if __name__ == "__main__":
    main()