import os
import sys
import subprocess
import getpass
import zipfile
import shutil
import time
import json
from pathlib import Path

# ANSI colors for premium terminal UI styling
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    GRAY = '\033[90m'

# Enable virtual terminal processing on Windows for ANSI colors
if sys.platform == 'win32':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

def print_progress(current, total, prefix='', suffix='', bar_length=40):
    """Draws a premium command-line progress bar."""
    percent = float(current) * 100 / total
    filled_len = int(percent / 100 * bar_length)
    arrow = '█' * filled_len
    spaces = '░' * (bar_length - filled_len)
    sys.stdout.write(f'\r{Colors.BOLD}{Colors.BLUE}{prefix}{Colors.END} |{Colors.CYAN}{arrow}{Colors.GRAY}{spaces}{Colors.END}| {percent:.1f}% {suffix}')
    sys.stdout.flush()

def apply_dataset_cleanup(root_dir, dataset_dir, config_keywords):
    # 7. Dataset Cleanup using Wav File Cleanup logs
    print(f"\n{Colors.BOLD}{Colors.CYAN}--- Dataset Cleanup Check ---{Colors.END}")
    run_cleanup = input(f"{Colors.BOLD}Do you want to cleanup the dataset? (y/N): {Colors.END}").strip().lower()
    if run_cleanup in ['y', 'yes']:
        cleanup_dirs = [
            os.path.join(root_dir, "Utils", "WavCleanUp"),
            os.path.join(root_dir, "Utils", "Wav File cleanup")
        ]
        
        # Locate all JSON cleanup log files
        cleanups_found = []
        for d in cleanup_dirs:
            if os.path.exists(d):
                for f in os.listdir(d):
                    if f.endswith("_data.json"):
                        cleanups_found.append(os.path.join(d, f))
                        
        if cleanups_found:
            print(f"\n{Colors.GREEN}Found cleanup logs for the following categories:{Colors.END}")
            for p in cleanups_found:
                print(f" - {Colors.BOLD}{os.path.basename(p)}{Colors.END}: {os.path.relpath(p, root_dir)}")
                
            print(f"\n{Colors.YELLOW}Applying cleanups...{Colors.END}")
            total_deleted = 0
            total_moved = 0
            
            for p in cleanups_found:
                try:
                    with open(p, 'r') as f:
                        cleanup_data = json.load(f)
                    logs = cleanup_data.get("logs", [])
                    
                    for entry in logs:
                        filepath = entry.get("filepath")
                        action = entry.get("action")
                        if not filepath or not action:
                            continue
                            
                        # Standardize path
                        filepath = filepath.replace('\\', '/')
                        src_path = os.path.join(dataset_dir, filepath)
                        
                        if os.path.exists(src_path):
                            if action == "delete":
                                try:
                                    os.remove(src_path)
                                    total_deleted += 1
                                except Exception as e:
                                    print(f"Error deleting {filepath}: {e}")
                            elif action == "keep":
                                pass
                            else:
                                # Action represents a destination category
                                target_cat = action.strip().lower()
                                target_cat_dir = os.path.join(dataset_dir, target_cat)
                                os.makedirs(target_cat_dir, exist_ok=True)
                                
                                # Move and rename
                                parts = filepath.split('/')
                                old_cat = parts[0]
                                filename = parts[-1]
                                new_name = f"{old_cat}_{filename}"
                                dst_path = os.path.join(target_cat_dir, new_name)
                                
                                # Handle collisions
                                if os.path.exists(dst_path):
                                    base, ext = os.path.splitext(new_name)
                                    counter = 1
                                    while os.path.exists(os.path.join(target_cat_dir, f"{base}_{counter}{ext}")):
                                        counter += 1
                                    dst_path = os.path.join(target_cat_dir, f"{base}_{counter}{ext}")
                                    
                                try:
                                    shutil.move(src_path, dst_path)
                                    total_moved += 1
                                except Exception as e:
                                    print(f"Error moving {filepath} to {target_cat}: {e}")
                                    
                except Exception as e:
                    print(f"{Colors.RED}Error reading cleanup log {p}: {e}{Colors.END}")
                    
            print(f"\n{Colors.GREEN}Cleanup applied successfully!{Colors.END}")
            print(f"Files deleted: {Colors.BOLD}{total_deleted}{Colors.END}")
            print(f"Files moved/reclassified: {Colors.BOLD}{total_moved}{Colors.END}")
        else:
            print("No cleanup logs found in Utils/WavCleanUp/ or Utils/Wav File cleanup/.")
    else:
        print("Dataset cleanup skipped.")

def main():
    # Print welcome banner
    print(f"{Colors.BOLD}{Colors.HEADER}==================================================={Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}      SPEECH COMMANDS DATASET SETUP UTILITY        {Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}==================================================={Colors.END}")

    # Dynamically resolve root and dataset paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))
    dataset_dir = os.path.join(root_dir, "dataset")

    # Add root path to sys.path to enable importing Utils.config_loader
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    # 1. Check if dataset already exists
    dataset_exists = False
    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
        # We consider dataset to exist if it contains subdirectories
        subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        if len(subdirs) > 0:
            dataset_exists = True

    if dataset_exists:
        print(f"\n{Colors.YELLOW}Warning: A dataset already exists at '{dataset_dir}'{Colors.END}")
        print("Existing directories found: " + ", ".join(subdirs[:10]) + ("..." if len(subdirs) > 10 else ""))
        reinstall = input(f"\n{Colors.BOLD}Do you want to fully reinstall the dataset? (y/N): {Colors.END}").strip().lower()
        
        if reinstall not in ['y', 'yes']:
            print(f"\n{Colors.GREEN}Existing dataset retained.{Colors.END}")
            try:
                from Utils.config_loader import get_keywords
                config_keywords = get_keywords()
            except Exception:
                config_keywords = ["yes", "no", "up", "down", "other"]
            apply_dataset_cleanup(root_dir, dataset_dir, config_keywords)
            sys.exit(0)
        else:
            print(f"\n{Colors.RED}Removing existing dataset directory structure...{Colors.END}")
            try:
                shutil.rmtree(dataset_dir)
                time.sleep(0.5) # Allow filesystem time to release locks
                print(f"{Colors.GREEN}Clean slate ready.{Colors.END}")
            except Exception as e:
                print(f"{Colors.YELLOW}Warning: Could not remove directory completely: {e}. Attempting to overwrite...{Colors.END}")

    # 2. Ask user for source method
    print(f"\n{Colors.BOLD}{Colors.CYAN}Choose Installation Source:{Colors.END}")
    print(f" {Colors.BOLD}[1]{Colors.END} Choose an existing local ZIP file")
    print(f" {Colors.BOLD}[2]{Colors.END} Download via Kaggle API (requires Kaggle Account & API key)")
    
    choice = ""
    while choice not in ['1', '2']:
        choice = input(f"\n{Colors.BOLD}Select choice (1 or 2): {Colors.END}").strip()

    zip_path = ""
    keep_zip = True

    if choice == '1':
        # Look for zip files in installer and project root directories for quick selection
        zip_options = []
        for search_dir in [script_dir, root_dir]:
            if os.path.exists(search_dir):
                for file in os.listdir(search_dir):
                    if file.lower().endswith(".zip"):
                        full_path = os.path.join(search_dir, file)
                        if full_path not in zip_options:
                            zip_options.append(full_path)

        if zip_options:
            print(f"\n{Colors.GREEN}Detected the following ZIP archives in workspace:{Colors.END}")
            for idx, opt in enumerate(zip_options, 1):
                rel = os.path.relpath(opt, script_dir)
                print(f"  [{idx}] {rel}")
            print(f"  [{len(zip_options) + 1}] Enter a custom path manually...")

            sel = 0
            while sel < 1 or sel > len(zip_options) + 1:
                try:
                    sel = int(input(f"\nSelect option (1-{len(zip_options) + 1}): ").strip())
                except ValueError:
                    pass

            if sel <= len(zip_options):
                zip_path = zip_options[sel - 1]

        if not zip_path:
            while True:
                path_input = input(f"\nEnter the absolute or relative path to the dataset ZIP file: ").strip()
                # Strip quotes in case user dragged and dropped file into cmd
                path_input = path_input.strip('"').strip("'")
                if os.path.exists(path_input) and path_input.lower().endswith(".zip"):
                    zip_path = os.path.abspath(path_input)
                    break
                print(f"{Colors.RED}Error: File not found or not a ZIP archive. Try again.{Colors.END}")

        print(f"\n{Colors.GREEN}Selected ZIP: {zip_path}{Colors.END}")
        keep_zip = True # Local files should never be deleted by default

    else:
        # Download from Kaggle
        keep_input = input(f"\nDo you want to keep the downloaded ZIP file after extraction? (y/N): ").strip().lower()
        keep_zip = keep_input in ['y', 'yes']

        # Ensure Kaggle package is installed
        try:
            import importlib.util
            if importlib.util.find_spec("kaggle") is None:
                raise ImportError
        except ImportError:
            print(f"\n{Colors.YELLOW}Installing required 'kaggle' python package...{Colors.END}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])

        username = input(f"\nEnter Kaggle Username: ").strip()
        key = getpass.getpass("Enter Kaggle API Key (input will be hidden): ").strip()

        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key

        # Set up kaggle api
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            print("\nAuthenticating with Kaggle...")
            api = KaggleApi()
            api.authenticate()
            print(f"{Colors.GREEN}Authentication successful.{Colors.END}")
        except Exception as e:
            print(f"{Colors.RED}Kaggle Authentication failed: {e}{Colors.END}")
            sys.exit(1)

        DATASET = "neehakurelli/google-speech-commands"
        ZIP_NAME = "google-speech-commands.zip"
        zip_path = os.path.join(script_dir, ZIP_NAME)

        print(f"\nDownloading dataset '{DATASET}' (approx. 1.4GB) via Kaggle...")
        try:
            api.dataset_download_cli(DATASET, path=script_dir, unzip=False)
            downloaded_zip = os.path.join(script_dir, DATASET.split('/')[-1] + ".zip")
            if os.path.exists(downloaded_zip) and downloaded_zip != zip_path:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                os.rename(downloaded_zip, zip_path)
            print(f"{Colors.GREEN}Download finished.{Colors.END}")
        except Exception as e:
            print(f"{Colors.RED}Download failed: {e}{Colors.END}")
            sys.exit(1)

    # 3. Extraction
    print(f"\n{Colors.BOLD}{Colors.CYAN}--- Extracting Dataset ---{Colors.END}")
    os.makedirs(dataset_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            namelist = zip_ref.namelist()
            file_entries = [name for name in namelist if not name.endswith('/')]
            
            # Check for a single top-level wrapper directory in the zip
            common_prefix = ""
            if file_entries:
                first_parts = Path(file_entries[0]).parts
                if len(first_parts) > 1:
                    first_dir = first_parts[0]
                    if all(Path(name).parts[0] == first_dir for name in file_entries):
                        common_prefix = first_dir + "/"
                        print(f"Stripping top-level archive directory: '{first_dir}'")

            total_files = len(file_entries)
            print(f"Extracting {total_files} files into '{dataset_dir}'...")

            for idx, name in enumerate(file_entries, 1):
                # Calculate path under dataset/ without wrapper directory
                rel_path = name[len(common_prefix):] if common_prefix else name
                target_path = os.path.join(dataset_dir, rel_path)

                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with zip_ref.open(name) as src, open(target_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst)

                if idx % 100 == 0 or idx == total_files:
                    print_progress(idx, total_files, prefix="Extracting", suffix=f"{idx}/{total_files} files")
            print(f"\n{Colors.GREEN}Extraction complete.{Colors.END}")

    except Exception as e:
        print(f"{Colors.RED}Extraction failed: {e}{Colors.END}")
        sys.exit(1)

    # 4. Restructuring and Keyword processing
    print(f"\n{Colors.BOLD}{Colors.CYAN}--- Restructuring Dataset Categories ---{Colors.END}")

    # Load configured keywords dynamically from Utils/config_loader.py
    try:
        from Utils.config_loader import get_keywords
        config_keywords = get_keywords()
        print(f"Configured keywords loaded: {config_keywords}")
    except Exception as e:
        print(f"{Colors.YELLOW}Warning: Could not load keywords from Utils.config_loader. Error: {e}{Colors.END}")
        print("Falling back to standard keywords: ['yes', 'no', 'up', 'down', 'other']")
        config_keywords = ["yes", "no", "up", "down", "other"]

    # Filter out catch-all 'other' category to get the exact core folders to keep
    core_keywords = [k.lower() for k in config_keywords if k.lower() not in ["other", "others"]]
    print(f"Core folders to retain: {core_keywords}")

    other_dir = os.path.join(dataset_dir, "other")
    os.makedirs(other_dir, exist_ok=True)

    # Scan and list all extracted category directories
    all_items = os.listdir(dataset_dir)
    categories = [item for item in all_items if os.path.isdir(os.path.join(dataset_dir, item))]

    # Separate operations: files to move vs background noise files to slice
    moves_to_make = []
    noise_files_to_process = []

    for category in categories:
        # Retain config keywords in place
        if category.lower() in core_keywords or category.lower() in ["other", "others"]:
            continue

        cat_dir = os.path.join(dataset_dir, category)

        # Handle the special background noise directory
        if category == "_background_noise_":
            # Note: We delete README.md from background noise folder later (during cleanup or explicitly here)
            for file in os.listdir(cat_dir):
                if file.lower().endswith(".wav"):
                    noise_files_to_process.append(os.path.join(cat_dir, file))
        else:
            # Move normal category folder wav files to other/ prepending folder name
            for file in os.listdir(cat_dir):
                if file.lower().endswith(".wav"):
                    src_file = os.path.join(cat_dir, file)
                    new_name = f"{category}_{file}"
                    dst_file = os.path.join(other_dir, new_name)

                    # Collision handling
                    if os.path.exists(dst_file):
                        base, ext = os.path.splitext(new_name)
                        counter = 1
                        while os.path.exists(os.path.join(other_dir, f"{base}_{counter}{ext}")):
                            counter += 1
                        dst_file = os.path.join(other_dir, f"{base}_{counter}{ext}")

                    moves_to_make.append((src_file, dst_file))

    # Move non-keyword categories
    total_moves = len(moves_to_make)
    if total_moves > 0:
        print(f"Merging and renaming {total_moves} WAV files into 'other/' folder...")
        for idx, (src, dst) in enumerate(moves_to_make, 1):
            try:
                shutil.move(src, dst)
            except Exception as e:
                print(f"\nError moving {src}: {e}")

            if idx % 100 == 0 or idx == total_moves:
                print_progress(idx, total_moves, prefix="Moving files", suffix=f"{idx}/{total_moves}")
        print(f"\n{Colors.GREEN}Successfully moved non-keyword files.{Colors.END}")

    # Process and slice background noise files into overlapping 1s segments
    total_noise_files = len(noise_files_to_process)
    if total_noise_files > 0:
        print(f"\nSlicing background noise files into timeInSeconds * 3 clips per file...")
        try:
            import numpy as np
            import soundfile as sf
            
            for idx, src in enumerate(noise_files_to_process, 1):
                filename = os.path.basename(src)
                try:
                    data, sr = sf.read(src)
                    # Convert stereo to mono
                    if len(data.shape) > 1:
                        data = data[:, 0]

                    N = len(data)
                    time_in_seconds = N / sr
                    num_clips = int(time_in_seconds * 3)
                    # Special override for doing_the_dishes.wav
                    if "doing_the_dishes" in filename.lower():
                        num_clips = 270

                    # We need exactly num_clips overlapping 1-second (sr samples) clips
                    if N > sr:
                        if num_clips > 1:
                            start_indices = [int(j * (N - sr) / (num_clips - 1)) for j in range(num_clips)]
                        else:
                            start_indices = [0]
                        for clip_num, start_idx in enumerate(start_indices, 1):
                            clip_data = data[start_idx : start_idx + sr]
                            clip_name = f"background_noise_{clip_num}_{filename}"
                            dst_file = os.path.join(other_dir, clip_name)
                            sf.write(dst_file, clip_data, sr)
                    else:
                        # Pad with zeros if shorter than 1 second
                        clip_data = np.pad(data, (0, sr - N), 'constant')
                        clip_name = f"background_noise_1_{filename}"
                        dst_file = os.path.join(other_dir, clip_name)
                        sf.write(dst_file, clip_data, sr)

                except Exception as e:
                    print(f"\nError processing noise file '{filename}': {e}")

                print_progress(idx, total_noise_files, prefix="Slicing noise", suffix=f"{idx}/{total_noise_files} files")
            print(f"\n{Colors.GREEN}Background noise slicing completed.{Colors.END}")
        except ImportError:
            print(f"\n{Colors.RED}Error: 'soundfile' or 'numpy' are required but not imported. Cannot slice background noise.{Colors.END}")

    # 5. Clean up original empty category directories
    print(f"\nCleaning up empty source folders...")
    for category in categories:
        if category.lower() in core_keywords or category.lower() in ["other", "others"]:
            continue
        cat_dir = os.path.join(dataset_dir, category)
        if os.path.exists(cat_dir):
            try:
                shutil.rmtree(cat_dir)
            except Exception:
                pass
                
    # Find and delete all README.md/README.MD files recursively under dataset_dir
    print(f"\nSearching for and deleting README files...")
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.lower() == "readme.md":
                fp = os.path.join(root, f)
                try:
                    os.remove(fp)
                    print(f"Deleted: {os.path.relpath(fp, dataset_dir)}")
                except Exception as e:
                    print(f"Could not delete {fp}: {e}")
                
    print(f"{Colors.GREEN}README cleanup completed.{Colors.END}")

    # 6. Delete downloaded ZIP archive if not keeping it
    if not keep_zip and os.path.exists(zip_path):
        print(f"\nDeleting temporary download archive '{zip_path}'...")
        try:
            os.remove(zip_path)
            print(f"{Colors.GREEN}Zip archive deleted successfully.{Colors.END}")
        except Exception as e:
            print(f"{Colors.YELLOW}Warning: Could not delete zip archive: {e}{Colors.END}")

    # 7. Dataset Cleanup using Wav File Cleanup logs
    apply_dataset_cleanup(root_dir, dataset_dir, config_keywords)

    # Final success message
    print(f"\n{Colors.BOLD}{Colors.GREEN}==================================================={Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}  SUCCESS: DATASET SETUP COMPLETED SUCCESSFULLY!   {Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}==================================================={Colors.END}")
    print(f"Dataset path: {dataset_dir}")
    print(f"Active category folders:")
    for d in os.listdir(dataset_dir):
        p = os.path.join(dataset_dir, d)
        if os.path.isdir(p):
            count = len([f for f in os.listdir(p) if f.lower().endswith(".wav")])
            print(f" - {Colors.BOLD}{d}/{Colors.END}: {count} WAV files")
    print(f"{Colors.BOLD}{Colors.GREEN}==================================================={Colors.END}\n")

if __name__ == "__main__":
    main()