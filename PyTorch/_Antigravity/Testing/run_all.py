import os
import subprocess
import glob
import re
import sys
from datetime import datetime

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(base_dir, "..", ".."))
    project_root = os.path.abspath(os.path.join(src_dir, ".."))
    python_exe = os.path.join(project_root, ".venv", "Scripts", "python.exe")
    
    if not os.path.exists(python_exe):
        print(f"Error: Could not find virtual env Python at {python_exe}")
        sys.exit(1)

    # Get all scripts matching pattern
    scripts = sorted(glob.glob(os.path.join(base_dir, "*.py")))
    # Filter out run_all.py itself
    scripts = [s for s in scripts if os.path.basename(s) != "run_all.py"]

    if len(sys.argv) > 1:
        target_files = sys.argv[1:]
        scripts = [s for s in scripts if os.path.basename(s) in target_files]

    results_dir = os.path.join(base_dir, "Results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "Results.txt")

    # Open in append mode instead of write mode to avoid overwriting
    with open(results_file, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\nModel Evaluation Results - [{timestamp}]\n")
        f.write("===================================================\n\n")

    for script in scripts:
        script_name = os.path.basename(script)
        rel_script_path = os.path.join("_Antigravity", "Testing", script_name)
        
        print(f"Running '{script_name}' on GPU (will take some time)...")
        # Run process from the src directory
        result = subprocess.run(
            [python_exe, rel_script_path],
            cwd=src_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        output = result.stdout
        
        # Regex to find all summary lines:
        # === Epoch X/Y [Z.ZZs] Summary | Train Loss: 0.XX | Train Acc: XX.XX% | Test Acc: XX.XX% ===
        summary_matches = re.findall(
            r"===\s*Epoch\s+(\d+/\d+)\s+\[([\d.]+)s\]\s+Summary\s+\|\s+Train Loss:\s+([\d.]+)\s+\|\s+Train Acc:\s+([\d.]+)%\s+\|\s+Test Acc:\s+([\d.]+)%\s*===", 
            output
        )
        
        if summary_matches:
            epochs_run = summary_matches[-1][0]
            avg_time = sum(float(m[1]) for m in summary_matches) / len(summary_matches)
            total_time = sum(float(m[1]) for m in summary_matches)
            final_train_loss = summary_matches[-1][2]
            final_train_acc = summary_matches[-1][3]
            final_test_acc = summary_matches[-1][4]
            
            # Check for best accuracy specifically tracked
            best_acc_match = re.search(r"Best Test Accuracy:\s+([\d.]+)%", output)
            best_test_acc = best_acc_match.group(1) if best_acc_match else final_test_acc

            out_str = (
                f"[{script_name}]\n"
                f"  - Epochs:          {epochs_run}\n"
                f"  - Avg Time/Epoch:  {avg_time:.2f}s (Total: {total_time:.1f}s)\n"
                f"  - Final Train Loss: {final_train_loss}\n"
                f"  - Final Train Acc:  {final_train_acc}%\n"
                f"  - Final Test Acc:   {final_test_acc}%\n"
                f"  - Best Test Acc:    {best_test_acc}%\n"
                f"---------------------------------------------------\n"
            )
        else:
            out_str = f"[{script_name}]\n  -> Failed to parse epoch summaries. Output size: {len(output)} chars.\n\n"
            
        print(out_str.strip())
        with open(results_file, "a") as f:
            f.write(out_str)
            
        # If there are any errors we can log them
        if result.returncode != 0:
            error_str = f"[{script_name}] -> Finished with errors. Check run log manually.\n\n"
            print(error_str.strip())
            with open(results_file, "a") as f:
                f.write(error_str)

    print(f"\nAll jobs completed. Results appended to {results_file}")

if __name__ == "__main__":
    main()
