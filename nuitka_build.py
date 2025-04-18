import os
import sys
import subprocess
import platform
import shutil

def main():
    print("Building Astraea PC Control Application with Nuitka...")
    
    # Make sure Nuitka is installed
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nuitka"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Clean previous build if it exists
    output_dir = "dist_nuitka"
    if os.path.exists(output_dir):
        print(f"Cleaning previous build in {output_dir}...")
        shutil.rmtree(output_dir, ignore_errors=True)
    
    # Determine icon path
    icon = ""
    if platform.system() == "Windows":
        icon_path = "src/ui/icons/app_icon.ico"
        if os.path.exists(icon_path):
            icon = f"--windows-icon-from-ico={icon_path}"
    
    # Build command for Nuitka
    nuitka_cmd = [
        sys.executable, 
        "-m", 
        "nuitka",
        "--standalone",
        "--follow-imports",
        "--windows-disable-console",
        "--include-package=cv2",
        "--include-package=onnxruntime",
        "--include-package=mediapipe",
        "--include-package=numpy",
        "--include-package=PyQt6",
        "--include-package=keyboard",
        "--include-package=sounddevice",
        "--include-data-dir=models=models",
        "--include-data-files=src/commands.json=src/commands.json",
        f"--output-dir={output_dir}",
        "--windows-company-name=Astraea",
        "--windows-product-name=Astraea PC Control",
        "--windows-file-version=1.0.0.0",
        "--windows-product-version=1.0.0.0",
    ]
    
    # Add icon if available
    if icon:
        nuitka_cmd.append(icon)
    
    # Add main script
    nuitka_cmd.append("main.py")
    
    # Run Nuitka
    print("Running command:", " ".join(nuitka_cmd))
    subprocess.check_call(nuitka_cmd)
    
    print("Build complete! Executable is in the 'dist_nuitka' folder.")
    print("You can distribute Astraea.exe and its folder to your users.")

if __name__ == "__main__":
    main() 