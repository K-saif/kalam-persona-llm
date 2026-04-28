import os
import ctypes

# Manually load the core C++ runtime that shm.dll needs
# Update the path to match your specific environment
dll_path = r"C:\Users\khans\miniconda3\envs\kalam2\Lib\site-packages\torch\lib\libiomp5md.dll"
if os.path.exists(dll_path):
    ctypes.CDLL(dll_path)

import torch
print(f"Success! GPU: {torch.cuda.get_device_name(0)}")