import os as _os
import warnings as _warnings

_warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but StandardScaler was fitted with feature names",
    category=UserWarning,
)
_warnings.filterwarnings(
    "ignore",
    message=r"X has feature names, but StandardScaler was fitted without feature names",
    category=UserWarning,
)

if not _os.path.exists(_os.path.join(_os.path.dirname(__file__), "paths.py")):
    raise ImportError(
        "configs/paths.py not found. Create it from the template:\n"
        "  cp configs/paths-template.py configs/paths.py\n"
        "Then edit the paths inside to match your setup."
    )
