"""The XGBoost path must import without the deep-learning stack.

requirements.txt installs neither torch nor pytorch_forecasting; the README's
quick start runs XGBoost on CPU. Each check runs in a fresh interpreter with
those packages blocked, because the test session itself has them loaded.
"""

import subprocess
import sys
import textwrap

import pytest

BLOCKED = ("torch", "pytorch_forecasting", "lightning", "pytorch_lightning")

_BLOCKER = f'''
import sys

BLOCKED = {BLOCKED!r}


class _Blocker:
    """Make the deep-learning stack look uninstalled."""

    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {{name!r}}")
        return None


for _name in list(sys.modules):
    if _name.split(".")[0] in BLOCKED:
        del sys.modules[_name]
sys.meta_path.insert(0, _Blocker())
'''


def _run_without_torch(body: str) -> subprocess.CompletedProcess:
    script = _BLOCKER + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
    )


@pytest.mark.parametrize(
    "module",
    [
        "configs.models",
        "src.trainers.xgb_trainer",
        "src.trainers.evaluation",
        "scripts.train_xgb",
        "src.visualization",
        "src.visualization.whatif",
        "src.inference.whatif",
    ],
)
def test_module_imports_without_torch(module):
    result = _run_without_torch(f"""
        import {module}
        print("ok")
    """)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_xgb_config_objects_are_usable_without_torch():
    result = _run_without_torch("""
        from configs.models import XGBTrainerConfig, XGBSearchSpace
        print(XGBTrainerConfig().__class__.__name__, XGBSearchSpace().__class__.__name__)
    """)

    assert result.returncode == 0, result.stderr
    assert "XGBTrainerConfig XGBSearchSpace" in result.stdout


def test_touching_the_tft_config_still_needs_torch():
    """Guards the blocker itself: the lazy import must be a real import."""
    result = _run_without_torch("""
        import configs.models
        try:
            configs.models.TFTTrainerConfig
        except ModuleNotFoundError:
            print("raised")
    """)

    assert "raised" in result.stdout, result.stderr


def test_unknown_config_name_raises_attribute_error():
    result = _run_without_torch("""
        import configs.models
        try:
            configs.models.NoSuchConfig
        except AttributeError as e:
            print("AttributeError", e)
    """)

    assert "AttributeError" in result.stdout, result.stderr
