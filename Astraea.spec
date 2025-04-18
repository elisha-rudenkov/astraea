
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
             hooksconfig={},
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
          name='Astraea_1744930419',
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
          entitlements_file=None )
