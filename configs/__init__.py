import importlib
import os

if not importlib.util.find_spec("configs.paths"):
    raise ImportError(
        "configs/paths.py not found. Create it from the template:\n"
        "  cp configs/paths-template.py configs/paths.py\n"
        "Then edit the paths inside to match your setup."
    )
