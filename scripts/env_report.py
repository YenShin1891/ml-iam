#!/usr/bin/env python
"""Report the interpreter and library versions this process is running under.

`make check-env` runs this through the same interpreter resolution as every
other target, so it answers "which environment did make actually pick?" --
the question that a silently mismatched environment makes hard to answer.
"""

import importlib
import importlib.util
import sys

# The packages whose versions change results, plus the two that decide whether
# a target can run at all (streamlit for the dashboard, pytest for the tests).
REPORTED = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "xgboost",
    "torch",
    "lightning",
    "pytorch_forecasting",
    "streamlit",
    "pytest",
)


def main() -> int:
    print(f"interpreter   = {sys.executable}")
    print(f"python        = {sys.version.split()[0]}")
    for name in REPORTED:
        if importlib.util.find_spec(name) is None:
            print(f"  {name:20} -")
            continue
        try:
            module = importlib.import_module(name)
        except Exception as e:  # noqa: BLE001 - a broken install is worth showing
            print(f"  {name:20} present but fails to import ({type(e).__name__})")
            continue
        print(f"  {name:20} {getattr(module, '__version__', '?')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
