import os as _os

if not _os.path.exists(_os.path.join(_os.path.dirname(__file__), "paths.py")):
    raise ImportError(
        "configs/paths.py not found. Create it from the template:\n"
        "  cp configs/paths-template.py configs/paths.py\n"
        "Then edit the paths inside to match your setup."
    )
