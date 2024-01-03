print("Let's get AuDiOrEaCtIvE!!!"

import os
import subprocess
import importlib.util
import sys

# scavenged install sequence from https://github.com/FizzleDorf/ComfyUI_FizzNodes/blob/main/__init__.py
def is_installed(package, package_overwrite=None):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        pass

    package = package_overwrite or package

    python = sys.executable
    if spec is None:
        print(f"Installing {package}...")
        command = f'"{python}" -m pip install {package}'
  
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=os.environ)

        if result.returncode != 0:
            print(f"Couldn't install\nCommand: {command}\nError code: {result.returncode}")

# to do: read from requirements.txt
is_installed("scipy")
is_installed("scikit-learn")
is_installed("librosa")
is_installed("loguru")


print(os.environ.get('COMFYUI_DEBUG_MODE'))

##############

# NB: nodes import needs to be after ensuring dependencies installed

#from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
#from .nodes.audio_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

from loguru import logger

logger.info(NODE_CLASS_MAPPINGS)
logger.info(NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
