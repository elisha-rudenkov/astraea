
# This is a PyInstaller hook file to ensure whisper is included
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all('whisper')
