"""Search bookkeeping that the trainers read back."""

import pytest

pytest.importorskip("lightning")

from lightning.pytorch.callbacks import EarlyStopping

from src.trainers.tft_trainer import _get_best_score


class _Trainer:
    def __init__(self, completed_epochs, callbacks):
        self.current_epoch = completed_epochs
        self.callbacks = callbacks


def test_the_best_epoch_is_reported_zero_based():
    """Four epochs ran (0-3); the best was two before the last: index 1."""
    import torch

    early_stop = EarlyStopping(monitor="val_loss")
    early_stop.wait_count = 2
    early_stop.best_score = torch.tensor(0.25)

    best_epoch, best_val_loss = _get_best_score(_Trainer(4, [early_stop]))

    assert (best_epoch, best_val_loss) == (1, 0.25)


# ── the TFT two-stage search ──────────────────────────────────────────────
#
# A TFT trial costs hours, so the flow through the stages and the resume path
# are worth pinning down here rather than discovering them in a search that
# has already been running overnight.

import src.trainers.tft_trainer as tft_trainer
from configs.models import TFTSearchSpace


@pytest.fixture
def small_search(tmp_path, monkeypatch):
    """A 4-trial, top-2 TFT search whose trials are recorded, never run."""
    space = TFTSearchSpace(n_trials=4, stage2_top_k=2)
    monkeypatch.setattr(tft_trainer, "TFTSearchSpace", lambda **kw: space)
    monkeypatch.setattr(tft_trainer, "get_run_root", lambda _run_id: str(tmp_path))

    calls = []

    def fake_run_trials(train_ds, val_ds, n_targets, params_list, trainer_cfg, run_id, stage):
        calls.append({
            "stage": stage,
            "params": list(params_list),
            "max_epochs": trainer_cfg.max_epochs,
        })
        return [
            {
                **tft_trainer._canonicalize_search_params(params),
                # Stage 2 scores better than any stage-1 trial, and reverses
                # stage 1's order, so the assertions below can tell which
                # stage the winner came from.
                "val_loss": (0.01 * (i + 1)) if stage == "stage2" else float(len(params_list) - i),
                "best_epoch": 3,
                "signature": tft_trainer._params_signature(params),
                "status": "completed",
                "stage": stage,
            }
            for i, params in enumerate(params_list)
        ]

    monkeypatch.setattr(tft_trainer, "_run_trials_once", fake_run_trials)
    return space, calls


def _search(ledger=(), monkeypatch=None):
    monkeypatch.setattr(tft_trainer, "_read_trials_ledger", lambda _run_id: list(ledger))
    return tft_trainer.hyperparameter_search_tft(None, None, ["A"], "tft_01")


def test_stage_one_runs_every_trial_at_the_reduced_budget(small_search, monkeypatch):
    space, calls = small_search

    _search(monkeypatch=monkeypatch)

    stage1 = next(call for call in calls if call["stage"] == "stage1")
    assert len(stage1["params"]) == space.n_trials
    assert stage1["max_epochs"] == space.stage1_budget["max_epochs"]


def test_stage_two_refits_only_the_leaders_at_the_full_budget(small_search, monkeypatch):
    from configs.models import TFTTrainerConfig
    space, calls = small_search

    _search(monkeypatch=monkeypatch)

    stage2 = next(call for call in calls if call["stage"] == "stage2")
    assert len(stage2["params"]) == space.stage2_top_k
    assert stage2["max_epochs"] == TFTTrainerConfig().max_epochs


def test_the_winner_comes_from_stage_two(small_search, monkeypatch):
    """Stage 1 only ranks candidates; the full-schedule scores decide."""
    space, calls = small_search

    best = _search(monkeypatch=monkeypatch)

    stage2_params = next(call for call in calls if call["stage"] == "stage2")["params"]
    assert {k: best[k] for k in space.param_keys} == tft_trainer._canonicalize_search_params(
        stage2_params[0]
    )
    assert best["best_epoch"] == 3


def test_a_resumed_search_reruns_nothing_it_already_has(small_search, monkeypatch):
    space, calls = small_search
    done = [
        {
            **tft_trainer._canonicalize_search_params(params),
            "val_loss": float(i), "best_epoch": 3, "status": "completed", "stage": "stage1",
            "signature": tft_trainer._params_signature(params),
        }
        for i, params in enumerate(space.sample()[:3])
    ]

    _search(ledger=done, monkeypatch=monkeypatch)

    stage1 = next(call for call in calls if call["stage"] == "stage1")
    assert len(stage1["params"]) == space.n_trials - 3


def test_the_search_writes_the_report_the_paper_needs(small_search, tmp_path, monkeypatch):
    _search(monkeypatch=monkeypatch)

    written = sorted(p.name for p in (tmp_path / "search").iterdir())
    assert written == ["budget_curve.csv", "search_space.csv", "trials.csv"]
