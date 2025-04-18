
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
                    f.write(f"{module_name}: {module_path}\n")
                else:
                    print(f"{module_name}: Module found but no file path")
                    f.write(f"{module_name}: Module found but no file path\n")
            except ImportError as e:
                print(f"{module_name}: Not found - {e}")
                f.write(f"{module_name}: Not found - {e}\n")
    print("Module path finder complete")
