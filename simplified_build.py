import os
import sys
import subprocess
import platform
import shutil

def main():
    print("Building Astraea PC Control Application (Simplified)...")
    
    # Make sure PyInstaller is installed
    print("Installing PyInstaller...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Clean previous build if it exists
    if os.path.exists("dist"):
        print("Cleaning previous build...")
        shutil.rmtree("dist", ignore_errors=True)
    if os.path.exists("build"):
        shutil.rmtree("build", ignore_errors=True)
    
    # Determine icon path
    icon = ""
    if platform.system() == "Windows":
        icon = "src/ui/icons/app_icon.ico" if os.path.exists("src/ui/icons/app_icon.ico") else ""
    
    # Install core dependencies directly instead of from requirements.txt
    print("Installing core dependencies...")
    core_deps = [
        "opencv-python",
        "numpy",
        "pyautogui",
        "keyboard",
        "onnxruntime",
        "pyqt6",
        "sounddevice"
    ]
    
    for dep in core_deps:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        except Exception as e:
            print(f"Warning: Could not install {dep}: {e}")
            print("Continuing anyway...")
    
    # Build the executable using python -m pyinstaller
    print("Building executable...")
    
    pyinstaller_cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name=Astraea",
        "--onefile",
        "--windowed",  # No console window
        "--add-data=src/commands.json;src/",
        "--add-data=models;models/", 
    ]
    
    # Add icon if available
    if icon:
        pyinstaller_cmd.append(f"--icon={icon}")
    
    # Add main script
    pyinstaller_cmd.append("main.py")
    
    # Run PyInstaller
    print("Running command:", " ".join(pyinstaller_cmd))
    subprocess.check_call(pyinstaller_cmd)
    
    print("Build complete! Executable is in the 'dist' folder.")
    print("You can distribute the Astraea.exe file to your users.")

if __name__ == "__main__":
    main() 