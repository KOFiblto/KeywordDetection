import os
import subprocess
import glob
import re
import sys

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(base_dir, ".."))
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

    results_file = os.path.join(base_dir, "Results.txt")

    # Open in append mode instead of write mode to avoid overwriting
    with open(results_file, "a") as f:
        f.write("Model Evaluation Results:\n")
        f.write("=========================\n\n")

    for script in scripts:
        script_name = os.path.basename(script)
        rel_script_path = os.path.join("_Antigravity", script_name)
        
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
        
        # Regex to find last "Test Acc: XX.XX%"
        matches = re.findall(r"Test Acc: (\d+\.\d+)%", output)
        if matches:
            final_acc = matches[-1]
            out_str = f"[{script_name}] -> Final Test Accuracy: {final_acc}%"
        else:
            out_str = f"[{script_name}] -> Failed to find Test Accuracy. Output size: {len(output)} chars."
            
        print(out_str)
        with open(results_file, "a") as f:
            f.write(out_str + "\n")
            
        # If there are any errors we can log them
        if result.returncode != 0:
            error_str = f"[{script_name}] -> Finished with errors. Check manually."
            print(error_str)
            with open(results_file, "a") as f:
                f.write(error_str + "\n")

    print(f"\nAll jobs completed. Results saved to {results_file}")

if __name__ == "__main__":
    main()
