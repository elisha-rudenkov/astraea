import os
import sys
import subprocess
import platform
import shutil

def main():
    print("Building Astraea PC Control Application...")
    
    # Make sure all dependencies are installed
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
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
    
    # Build the executable using python -m pyinstaller instead of direct command
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