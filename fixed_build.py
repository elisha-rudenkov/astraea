import os
import sys
import subprocess
import platform
import shutil
import time
import psutil

def main():
    print("Building Astraea PC Control Application (with DLL handling)...")
    
    # Make sure PyInstaller is installed
    print("Installing PyInstaller...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Try to install psutil for process handling
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    except:
        print("Warning: Could not install psutil. Process termination may not work correctly.")
    
    # Check if the executable is running and try to terminate it
    exe_path = os.path.join("dist", "Astraea.exe")
    if os.path.exists(exe_path):
        print(f"Checking if {exe_path} is running...")
        try:
            for proc in psutil.process_iter(['pid', 'name', 'exe']):
                if proc.info['exe'] and os.path.normcase(proc.info['exe']) == os.path.normcase(os.path.abspath(exe_path)):
                    print(f"Terminating running process (PID: {proc.info['pid']})")
                    proc.terminate()
                    proc.wait(timeout=3)
        except Exception as e:
            print(f"Warning: Could not check/terminate processes: {e}")
            print("Please close any running instances of Astraea.exe manually.")
            input("Press Enter to continue once all instances are closed...")
    
    # Give the system a moment to release any file locks
    time.sleep(2)
    
    # Clean previous build if it exists
    print("Cleaning previous build...")
    try:
        if os.path.exists("dist"):
            for root, dirs, files in os.walk("dist", topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except Exception as e:
                        print(f"Warning: Could not remove {name}: {e}")
            shutil.rmtree("dist", ignore_errors=True)
        if os.path.exists("build"):
            shutil.rmtree("build", ignore_errors=True)
    except Exception as e:
        print(f"Warning: Could not completely clean previous build: {e}")
        print("Will try to continue anyway.")
    
    # Install all required dependencies - order matters!
    print("Installing dependencies...")
    
    # Basic dependencies first
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy>=1.19.0"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python>=4.5.0"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyautogui>=0.9.53"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "keyboard"])
    
    # ONNX Runtime - Install CPU version first
    print("Installing ONNX Runtime...")
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime-gpu", "onnxruntime-directml"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime==1.16.3"])
    
    # Mediapipe - important for face tracking
    print("Installing Mediapipe...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapipe>=0.8.0"])
    
    # UI components
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyqt6"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sounddevice"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "jsonschema"])
    
    # Install QAI Hub with whisper-tiny-en model
    print("Installing QAI Hub Models with whisper-tiny-en - this may take a while...")
    try:
        # First install essential core packages
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper==20231117"])
        
        # Install the necessary PyTorch dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.1.2", "torchaudio==2.1.2", "torchvision==0.16.2"])
        
        # Install audio processing packages
        subprocess.check_call([sys.executable, "-m", "pip", "install", "audio2numpy==0.1.2", "samplerate==0.2.1"])
        
        # Install ffmpeg
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg"])
        
        # Install the QAI Hub packages with the whisper-tiny-en extras
        subprocess.check_call([sys.executable, "-m", "pip", "install", "qai-hub==0.21.0"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "qai-hub-models[whisper-tiny-en]==0.22.1"])
        
        print("QAI Hub Models with whisper-tiny-en installed successfully")
    except Exception as e:
        print(f"Warning: Could not install QAI Hub Models: {e}")
        print("Continuing anyway - voice functionality may not work.")
    
    # Determine icon path
    icon = ""
    if platform.system() == "Windows":
        icon_path = "src/ui/icons/app_icon.ico"
        if os.path.exists(icon_path):
            icon = f"--icon={icon_path}"
    
    # Generate a unique output name
    output_name = f"Astraea_{int(time.time())}"
    
    # Create a special helper for finding actual package paths
    print("Creating helper script to find module paths...")
    module_finder_path = "find_module_paths.py"
    module_finder_content = """
import sys
import os
import importlib

# List of modules to locate
modules = [
    'qai_hub_models.models.whisper_tiny_en',
    'qai_hub_models',
    'qai_hub',
    'whisper',
    'audio2numpy',
    'samplerate',
]

if __name__ == "__main__":
    print("Starting module path finder")
    with open('module_paths.txt', 'w') as f:
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, '__file__') and module.__file__:
                    module_path = os.path.dirname(module.__file__)
                    print(f"{module_name}: {module_path}")
                    f.write(f"{module_name}: {module_path}\\n")
                else:
                    print(f"{module_name}: Module found but no file path")
                    f.write(f"{module_name}: Module found but no file path\\n")
            except ImportError as e:
                print(f"{module_name}: Not found - {e}")
                f.write(f"{module_name}: Not found - {e}\\n")
    print("Module path finder complete")
"""
    with open(module_finder_path, 'w') as f:
        f.write(module_finder_content)
    
    # Run the module finder
    print("Finding actual module paths...")
    subprocess.check_call([sys.executable, module_finder_path])
    
    # Create a spec file to customize the build
    spec_content = f"""
# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_all, copy_metadata

# Collect all necessary data, binaries, and hidden imports
qai_datas, qai_binaries, qai_hiddenimports = collect_all('qai_hub_models')
whisper_datas, whisper_binaries, whisper_hiddenimports = collect_all('whisper')
torchaudio_datas, torchaudio_binaries, torchaudio_hiddenimports = collect_all('torchaudio')

# Also collect metadata for certain packages
metadata_datas = copy_metadata('qai_hub_models')

a = Analysis(['main.py'],
             pathex=[],
             binaries=qai_binaries + whisper_binaries + torchaudio_binaries,
             datas=[
                ('src/commands.json', 'src'), 
                ('models', 'models')
             ] + qai_datas + whisper_datas + torchaudio_datas + metadata_datas,
             hiddenimports=[
                'cv2', 
                'onnxruntime.capi', 
                'mediapipe', 
                'PyQt6.QtCore', 
                'PyQt6.QtGui', 
                'PyQt6.QtWidgets',
                'qai_hub_models',
                'qai_hub_models.models.whisper_tiny_en',
                'qai_hub_models.models',
                'qai_hub',
                'whisper',
                'torch',
                'torchvision',
                'torchaudio',
                'sounddevice',
                'numpy',
                'audio2numpy',
                'samplerate',
                'ffmpeg',
                'tiktoken',
                'openai_whisper',
                'qai_hub_models.models._shared.whisper.app',
                'qai_hub_models.models._shared.whisper',
                'qai_hub_models.models._shared',
             ] + qai_hiddenimports + whisper_hiddenimports + torchaudio_hiddenimports,
             hookspath=['.'],
             hooksconfig={{}},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=None,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='{output_name}',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None {', icon=r"'+icon_path+'"' if os.path.exists(icon_path) else ''})
"""
    
    with open('astraea.spec', 'w') as f:
        f.write(spec_content)
    
    # Create hook files for the key modules
    print("Creating additional hook files...")
    
    # Hook for qai_hub_models
    qai_hook_path = "hook-qai_hub_models.py"
    qai_hook_content = """
from PyInstaller.utils.hooks import collect_all, collect_submodules

# Collect all submodules
hiddenimports = collect_submodules('qai_hub_models')

# Collect all data
datas, binaries, _ = collect_all('qai_hub_models')
"""
    with open(qai_hook_path, 'w') as f:
        f.write(qai_hook_content)
    
    # Hook for whisper
    whisper_hook_path = "hook-whisper.py"
    whisper_hook_content = """
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all('whisper')
"""
    with open(whisper_hook_path, 'w') as f:
        f.write(whisper_hook_content)
    
    # Build using the spec file
    print("Building executable...")
    pyinstaller_cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "astraea.spec",
        "--clean",
    ]
    
    # Run PyInstaller
    print("Running command:", " ".join(pyinstaller_cmd))
    try:
        subprocess.check_call(pyinstaller_cmd)
        
        # Rename the final executable back to Astraea.exe if needed
        final_exe = os.path.join("dist", f"{output_name}.exe")
        target_exe = os.path.join("dist", "Astraea.exe")
        
        if os.path.exists(final_exe) and final_exe != target_exe:
            if os.path.exists(target_exe):
                try:
                    os.remove(target_exe)
                except:
                    print(f"Warning: Could not remove existing {target_exe}")
                    print(f"The new executable is available as {final_exe}")
                    print("You can manually rename it to Astraea.exe when possible.")
                    return
            try:
                os.rename(final_exe, target_exe)
                print(f"Renamed {final_exe} to {target_exe}")
            except:
                print(f"Warning: Could not rename {final_exe} to {target_exe}")
                print(f"The executable is available as {final_exe}")
        
        print("Build complete! Executable is in the 'dist' folder.")
        print("You can distribute the Astraea.exe file to your users.")
        
    except subprocess.CalledProcessError as e:
        print(f"Error during build: {e}")
        print("Try closing all Python applications and the Astraea.exe if it's running.")
        print("Then run this script again.")

if __name__ == "__main__":
    main() 