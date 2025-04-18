# Building Astraea into an Executable

This guide explains how to convert the Astraea application into a standalone executable for Windows users.

## Prerequisites

Make sure you have Python 3.8 or newer installed. This build process works on Windows systems.

## Build Options

You have two options for building the executable:

### Option 1: PyInstaller (Simpler, but larger output)

1. Clone or download this repository
2. Open a terminal (Command Prompt or PowerShell) in the project directory
3. Run the build script:

```
python build.py
```

4. Wait for the build process to complete (this may take several minutes)
5. Once finished, you'll find the executable in the `dist` folder:
   - `dist/Astraea.exe`

### Option 2: Nuitka (Faster execution, more optimized)

1. Clone or download this repository
2. Open a terminal (Command Prompt or PowerShell) in the project directory
3. Run the Nuitka build script:

```
python nuitka_build.py
```

4. Wait for the build process to complete (this may take 10-15 minutes)
5. Once finished, you'll find the executable in the `dist_nuitka` folder:
   - `dist_nuitka/Astraea.exe` (along with dependency files)

## Distribution

### PyInstaller Output
The standalone executable can be distributed to users who don't have Python installed. Simply share the `Astraea.exe` file.

### Nuitka Output
For the Nuitka build, you'll need to distribute the entire `dist_nuitka` folder. You can zip it for distribution.

## Important Notes:

- The executable contains all necessary Python dependencies and model files
- Initial startup may be slower than running the script normally
- Some antivirus software may flag the executable - this is a common false positive with packaged Python applications
- The size of the executable will be large (around 100-200MB) due to the inclusion of ML models and dependencies

## Troubleshooting

If the build fails:

1. Make sure all requirements are installed: `pip install -r requirements.txt`
2. Ensure you have administrative privileges on your system
3. Try running with a clean environment: `python -m build.py` or `python -m nuitka_build.py`

If the executable doesn't run:

1. Run it from command line to see any error messages: `dist\Astraea.exe` or `dist_nuitka\Astraea.exe`
2. Ensure the system has appropriate drivers for webcam access
3. For certain GPU acceleration features, appropriate GPU drivers must be installed 