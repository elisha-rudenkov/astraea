
from PyInstaller.utils.hooks import collect_all, collect_submodules

# Collect all submodules
hiddenimports = collect_submodules('qai_hub_models')

# Collect all data
datas, binaries, _ = collect_all('qai_hub_models')
