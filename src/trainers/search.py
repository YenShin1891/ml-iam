"""Shared hyperparameter-search machinery: spaces, sampling and trial ranking.

The three models search very different parameters -- there is no sense in
which ``hidden_size`` and ``max_depth`` are the same knob -- so their spaces
cannot be made identical.  What *can* be made identical, and what a comparison
between the models depends on, is the procedure that turns a space into a
chosen configuration:

1. draw ``n_trials`` configurations from declared distributions with a fixed
   seed (one-shot random search, no staged coordinate descent),
2. train each under the reduced *stage 1* budget,
3. refit the ``stage2_top_k`` best under the full budget and take the winner.

Stage 2's budget is the trainer config's own -- longer than stage 1, and long
enough that a configuration is judged on where it converges rather than on how
fast it starts, which is the failure mode a truncated ranking has.

Only the budget differs per model, and it has to: a TFT trial costs hours
while an XGBoost trial costs seconds.  Stage 1 exists precisely so the
expensive model can be searched at all, and running the cheap models through
the same two stages costs almost nothing while making the protocol one rule
instead of three.

Distributions are continuous wherever the underlying parameter is continuous
(log-uniform for learning rates and regularisation strengths, uniform for
dropout), so coverage no longer depends on how finely someone happened to
discretise a range.
"""
from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "Distribution",
    "Uniform",
    "LogUniform",
    "IntUniform",
    "IntLogUniform",
    "Choice",
    "SearchSpace",
    "canonicalize_params",
    "params_signature",
    "is_completed_trial",
    "completed_trials",
    "rank_trials",
    "best_trial",
    "select_top_k_signatures",
    "budget_curve",
    "StagePlan",
    "plan_two_stage_search",
    "write_search_report",
]


# --------------------------------------------------------------------------
# Distributions
# --------------------------------------------------------------------------

class Distribution(ABC):
    """One searchable parameter.

    ``rvs`` matches what scikit-learn's ParameterSampler expects, so these
    objects also work anywhere a scipy frozen distribution would.
    """

    @abstractmethod
    def rvs(self, random_state: np.random.RandomState) -> Any:
        """Draw one value."""

    @abstractmethod
    def describe(self) -> str:
        """Human-readable range, for the search-space table in the paper."""

    @property
    def cardinality(self) -> Optional[int]:
        """Number of distinct values, or None when continuous."""
        return None


@dataclass(frozen=True)
class Uniform(Distribution):
    """Continuous, uniform on [low, high]."""

    low: float
    high: float

    def __post_init__(self):
        if not self.high >= self.low:
            raise ValueError(f"Uniform requires high >= low, got [{self.low}, {self.high}]")

    def rvs(self, random_state):
        return float(random_state.uniform(self.low, self.high))

    def describe(self) -> str:
        return f"uniform[{_num(self.low)}, {_num(self.high)}]"


@dataclass(frozen=True)
class LogUniform(Distribution):
    """Continuous, uniform on a log scale over [low, high]; both must be > 0.

    The right default for anything spanning orders of magnitude -- learning
    rates, weight decay, ``reg_alpha`` -- where a linear draw would spend
    almost every sample in the top decade.
    """

    low: float
    high: float

    def __post_init__(self):
        if self.low <= 0:
            raise ValueError(f"LogUniform requires low > 0, got {self.low}")
        if not self.high >= self.low:
            raise ValueError(f"LogUniform requires high >= low, got [{self.low}, {self.high}]")

    def rvs(self, random_state):
        return float(np.exp(random_state.uniform(math.log(self.low), math.log(self.high))))

    def describe(self) -> str:
        return f"log-uniform[{_num(self.low)}, {_num(self.high)}]"


@dataclass(frozen=True)
class IntUniform(Distribution):
    """Integer, uniform on [low, high] inclusive, optionally on a step grid."""

    low: int
    high: int
    step: int = 1

    def __post_init__(self):
        if self.step < 1:
            raise ValueError(f"IntUniform requires step >= 1, got {self.step}")
        if not self.high >= self.low:
            raise ValueError(f"IntUniform requires high >= low, got [{self.low}, {self.high}]")

    def rvs(self, random_state):
        n = self.cardinality
        return int(self.low + self.step * random_state.randint(n))

    @property
    def cardinality(self) -> Optional[int]:
        return (int(self.high) - int(self.low)) // int(self.step) + 1

    def describe(self) -> str:
        grid = f" step {self.step}" if self.step != 1 else ""
        return f"int-uniform[{self.low}, {self.high}]{grid}"


@dataclass(frozen=True)
class IntLogUniform(Distribution):
    """Integer, uniform on a log scale over [low, high] inclusive.

    For capacity parameters (``hidden_size``, ``max_depth``) where doubling
    matters more than adding a constant.  ``multiple_of`` snaps draws onto a
    grid: layer widths that are multiples of 8 keep the GPU kernels on their
    fast paths, and nothing is lost, because no two widths 3 apart differ in
    any way the search could resolve.
    """

    low: int
    high: int
    multiple_of: int = 1

    def __post_init__(self):
        if self.low < 1:
            raise ValueError(f"IntLogUniform requires low >= 1, got {self.low}")
        if not self.high >= self.low:
            raise ValueError(f"IntLogUniform requires high >= low, got [{self.low}, {self.high}]")
        if self.multiple_of < 1:
            raise ValueError(f"IntLogUniform requires multiple_of >= 1, got {self.multiple_of}")

    def rvs(self, random_state):
        # +1 in the upper log bound and floor below so the top value gets the
        # same width as the others; plain rounding would halve its probability.
        drawn = np.exp(random_state.uniform(math.log(self.low), math.log(self.high + 1)))
        value = min(int(math.floor(drawn)), int(self.high))
        if self.multiple_of > 1:
            value = int(round(value / self.multiple_of)) * self.multiple_of
            value = min(max(value, self._lowest_on_grid), self._highest_on_grid)
        return int(value)

    @property
    def _lowest_on_grid(self) -> int:
        return int(math.ceil(self.low / self.multiple_of)) * self.multiple_of

    @property
    def _highest_on_grid(self) -> int:
        return int(math.floor(self.high / self.multiple_of)) * self.multiple_of

    @property
    def cardinality(self) -> Optional[int]:
        if self.multiple_of > 1:
            span = self._highest_on_grid - self._lowest_on_grid
            if span < 0:
                raise ValueError(
                    f"No multiple of {self.multiple_of} lies in [{self.low}, {self.high}]"
                )
            return span // self.multiple_of + 1
        return int(self.high) - int(self.low) + 1

    def describe(self) -> str:
        grid = f" (multiples of {self.multiple_of})" if self.multiple_of > 1 else ""
        return f"int-log-uniform[{self.low}, {self.high}]{grid}"


@dataclass(frozen=True)
class Choice(Distribution):
    """A genuinely discrete parameter: unordered, or with very few settings.

    Use this only where interpolation is meaningless (``num_layers``,
    ``sequence_length``, an optimiser name) -- not as a way to discretise a
    continuous range.
    """

    values: Tuple[Any, ...]

    def __init__(self, values: Sequence[Any]):
        values = tuple(values)
        if not values:
            raise ValueError("Choice requires at least one value")
        object.__setattr__(self, "values", values)

    def rvs(self, random_state):
        return self.values[random_state.randint(len(self.values))]

    @property
    def cardinality(self) -> Optional[int]:
        return len(self.values)

    def describe(self) -> str:
        return "{" + ", ".join(_num(v) for v in self.values) + "}"


def _num(value: Any) -> str:
    """Compact rendering that keeps whole numbers whole."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e16:
        return str(int(value))
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


# --------------------------------------------------------------------------
# Search space
# --------------------------------------------------------------------------

_DistLike = Union[Distribution, Sequence[Any]]


@dataclass
class SearchSpace:
    """A model's search space plus the shared two-stage protocol's settings.

    Attributes
    ----------
    distributions:
        Parameter name -> Distribution.  A bare list is accepted and wrapped
        in :class:`Choice`, so existing discrete spaces port over unchanged.
    n_trials:
        Configurations drawn for stage 1.  The comparison's budget axis.
    stage1_budget:
        Trainer-config attributes overridden for stage-1 trials, e.g.
        ``{"max_epochs": 25, "patience": 6}``.  Model-specific by necessity:
        an epoch of TFT is not an epoch of LSTM, and XGBoost counts rounds.
    stage2_top_k:
        Stage-1 leaders refit under the trainer config's full budget.
    seed:
        Fixed so a search is reproducible and so the three models see draws
        from the same generator sequence.
    """

    distributions: Mapping[str, _DistLike]
    n_trials: int
    stage1_budget: Dict[str, Any] = field(default_factory=dict)
    stage2_top_k: int = 10
    seed: int = 0

    def __post_init__(self):
        if self.n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {self.n_trials}")
        if self.stage2_top_k < 1:
            raise ValueError(f"stage2_top_k must be >= 1, got {self.stage2_top_k}")
        normalized: Dict[str, Distribution] = {}
        for name, dist in self.distributions.items():
            if isinstance(dist, Distribution):
                normalized[name] = dist
            elif isinstance(dist, (list, tuple)):
                normalized[name] = Choice(dist)
            else:
                raise TypeError(
                    f"Parameter {name!r} must be a Distribution or a sequence, got {type(dist).__name__}"
                )
        self.distributions = normalized

    @property
    def param_keys(self) -> List[str]:
        """Searched parameter names, in a stable order."""
        return sorted(self.distributions)

    @property
    def cardinality(self) -> Optional[int]:
        """Distinct configurations the space can produce, or None if unbounded."""
        total = 1
        for dist in self.distributions.values():
            n = dist.cardinality
            if n is None:
                return None
            total *= n
        return total

    def sample(self, n: Optional[int] = None, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Draw ``n`` configurations (default ``n_trials``).

        Duplicates are rejected while the space is small enough for that to
        terminate; a space with any continuous parameter effectively never
        collides, so the check costs nothing there.
        """
        n = int(self.n_trials if n is None else n)
        if n < 0:
            raise ValueError(f"Cannot draw {n} configurations")
        rng = np.random.RandomState(self.seed if seed is None else seed)

        capacity = self.cardinality
        if capacity is not None and n > capacity:
            raise ValueError(
                f"Asked for {n} distinct configurations but the space holds only {capacity}. "
                "Widen the space or lower n_trials."
            )

        keys = self.param_keys
        drawn: List[Dict[str, Any]] = []
        seen = set()
        # Enough attempts to fill a nearly-exhausted discrete space by the
        # coupon-collector bound, without looping forever if one is malformed.
        max_attempts = 100 * max(n, 1) + 1000
        for _ in range(max_attempts):
            if len(drawn) >= n:
                break
            params = {key: self.distributions[key].rvs(rng) for key in keys}
            signature = params_signature(params, keys)
            if signature in seen:
                continue
            seen.add(signature)
            drawn.append(params)
        if len(drawn) < n:
            raise RuntimeError(
                f"Drew only {len(drawn)} distinct configurations out of {n} requested "
                f"in {max_attempts} attempts."
            )
        return drawn

    def describe(self) -> List[Dict[str, str]]:
        """Rows of (parameter, range) -- the search-space table, ready to print."""
        return [
            {"parameter": key, "range": self.distributions[key].describe()}
            for key in self.param_keys
        ]

    def summary(self) -> str:
        """One-line description for the search log."""
        capacity = self.cardinality
        size = "continuous" if capacity is None else f"{capacity} combinations"
        return (
            f"{len(self.distributions)} parameters ({size}), "
            f"{self.n_trials} trials at stage-1 budget {self.stage1_budget or '{}'}, "
            f"top {self.stage2_top_k} refit at full budget, seed {self.seed}"
        )


# --------------------------------------------------------------------------
# Trial bookkeeping
# --------------------------------------------------------------------------

def canonicalize_params(params: Mapping[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    """Params restricted to ``keys``, with floats rounded for stable equality.

    Trials round-trip through JSON ledgers and pandas frames, either of which
    can perturb the last bits of a float; comparing at 12 significant digits
    keeps a resumed search from re-running configurations it already has.  A
    whole float becomes an int for the same reason: collecting trials into a
    DataFrame widens every integer column to float64, so ``hidden_size=64``
    comes back as ``64.0`` and would otherwise no longer match its own ledger
    entry.
    """
    canonical: Dict[str, Any] = {}
    for key in keys:
        if key not in params:
            continue
        value = params[key]
        if isinstance(value, (np.integer,)):
            canonical[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            value = float(format(float(value), ".12g"))
            canonical[key] = int(value) if value.is_integer() else value
        else:
            canonical[key] = value
    return canonical


def params_signature(params: Mapping[str, Any], keys: Sequence[str]) -> str:
    """Stable identity for a configuration, used for dedup and resume."""
    canonical = canonicalize_params(params, keys)
    missing = [key for key in keys if key not in canonical]
    if missing:
        raise ValueError(
            f"Missing search params for signature: {missing}. Got keys={sorted(params)}"
        )
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def is_completed_trial(row: Mapping[str, Any], keys: Sequence[str], metric: str = "val_loss") -> bool:
    """True only for rows usable for ranking and resume.

    A trial that crashed, or one still being written, must not be mistaken for
    a bad-but-finished configuration: silently, that turns a lost trial into a
    shorter search.
    """
    if not isinstance(row, Mapping):
        return False
    if row.get("status") != "completed":
        return False
    if not all(key in row for key in keys):
        return False
    try:
        value = float(row[metric])
    except (TypeError, ValueError, KeyError):
        return False
    return math.isfinite(value)


def completed_trials(
    rows: Iterable[Mapping[str, Any]],
    keys: Sequence[str],
    metric: str = "val_loss",
) -> List[Dict[str, Any]]:
    """The subset of ``rows`` that :func:`is_completed_trial` accepts."""
    return [dict(row) for row in rows if is_completed_trial(row, keys, metric)]


def rank_trials(
    rows: Iterable[Mapping[str, Any]],
    metric: str = "val_loss",
    mode: str = "min",
) -> List[Dict[str, Any]]:
    """Trials sorted best-first by ``metric``."""
    if mode not in ("min", "max"):
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
    sign = 1.0 if mode == "min" else -1.0
    return sorted((dict(row) for row in rows), key=lambda row: sign * float(row[metric]))


def best_trial(
    rows: Iterable[Mapping[str, Any]],
    metric: str = "val_loss",
    mode: str = "min",
) -> Dict[str, Any]:
    """The single best trial; raises when there is none to choose from."""
    ranked = rank_trials(rows, metric, mode)
    if not ranked:
        raise RuntimeError(f"No completed trials to select a best {metric} from.")
    return ranked[0]


def select_top_k_signatures(
    rows: Iterable[Mapping[str, Any]],
    k: int,
    keys: Sequence[str],
    metric: str = "val_loss",
    mode: str = "min",
) -> List[str]:
    """Signatures of the ``k`` best distinct configurations, best-first."""
    signatures: List[str] = []
    seen = set()
    for row in rank_trials(rows, metric, mode):
        signature = row.get("signature") or params_signature(row, keys)
        if signature in seen:
            continue
        seen.add(signature)
        signatures.append(signature)
        if len(signatures) >= max(1, int(k)):
            break
    return signatures


def budget_curve(
    rows: Iterable[Mapping[str, Any]],
    metric: str = "val_loss",
    mode: str = "min",
    order_key: str = "trial",
) -> List[float]:
    """Running best score after each trial, in the order the trials ran.

    Plotted against the trial index this is the search-budget sensitivity
    curve: if it flattens well before the budget ends, more search would not
    have changed the winner, which is the evidence a reader needs to accept a
    comparison run at a fixed budget.
    """
    rows = list(rows)
    if order_key and all(order_key in row for row in rows):
        rows.sort(key=lambda row: float(row[order_key]))
    better = min if mode == "min" else max
    curve: List[float] = []
    for row in rows:
        value = float(row[metric])
        curve.append(value if not curve else better(curve[-1], value))
    return curve


# --------------------------------------------------------------------------
# The two-stage protocol
# --------------------------------------------------------------------------

@dataclass
class StagePlan:
    """What a (possibly resumed) two-stage search still has to run.

    ``stage1_pending`` and ``stage2_pending`` are parameter dicts; everything
    already in the ledger is excluded, so re-entering a search after a crash
    picks up where it stopped rather than repeating hours of trials.
    """

    all_params: List[Dict[str, Any]]
    stage1_pending: List[Dict[str, Any]]
    stage2_pending: List[Dict[str, Any]]
    stage1_done: int
    stage2_done: int

    @property
    def signature_to_params(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._sig_to_params)

    _sig_to_params: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)


def plan_two_stage_search(
    space: SearchSpace,
    ledger_rows: Iterable[Mapping[str, Any]],
    metric: str = "val_loss",
    mode: str = "min",
) -> StagePlan:
    """Work out what stage 1 and stage 2 still owe, given the trials on disk.

    Stage 2's candidates are only known once stage 1 has finished, so this
    returns stage 2's pending list computed from the trials *already*
    recorded.  Callers run stage 1, append its results, and call again (or use
    :func:`select_top_k_signatures` directly) to get the final stage-2 list.
    """
    keys = space.param_keys
    all_params = space.sample()
    sig_to_params = {params_signature(p, keys): p for p in all_params}

    done = completed_trials(ledger_rows, keys, metric)
    stage1_sigs = set()
    stage2_sigs = set()
    for row in done:
        signature = row.get("signature") or params_signature(row, keys)
        if row.get("stage") == "stage2":
            stage2_sigs.add(signature)
        else:
            stage1_sigs.add(signature)

    # A signature that reached stage 2 necessarily cleared stage 1.
    explored = stage1_sigs | stage2_sigs
    stage1_pending = [p for sig, p in sig_to_params.items() if sig not in explored]

    stage1_rows = [row for row in done if row.get("stage") != "stage2"]
    stage2_pending: List[Dict[str, Any]] = []
    if stage1_rows:
        for signature in select_top_k_signatures(
            stage1_rows, space.stage2_top_k, keys, metric, mode
        ):
            if signature in sig_to_params and signature not in stage2_sigs:
                stage2_pending.append(sig_to_params[signature])

    plan = StagePlan(
        all_params=all_params,
        stage1_pending=stage1_pending,
        stage2_pending=stage2_pending,
        stage1_done=len(stage1_sigs),
        stage2_done=len(stage2_sigs),
    )
    plan._sig_to_params = sig_to_params
    return plan


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def write_search_report(
    out_dir: str,
    space: "SearchSpace",
    rows: Iterable[Mapping[str, Any]],
    metric: str = "val_loss",
    mode: str = "min",
) -> Dict[str, str]:
    """Write the three artefacts every model's search now produces.

    ``search_space.csv`` is the declared space, ready to drop into the paper
    as the search-space table.  ``trials.csv`` is every completed trial.
    ``budget_curve.csv`` is the running best against trial count -- the
    evidence that the budget was large enough for the winner to have settled.

    The curve is drawn from stage-1 trials alone.  Stage 2 re-runs a handful
    of configurations under a different budget, so its scores are not
    comparable with stage 1's and appending them would show an improvement
    that came from training longer rather than from searching more.

    Returns the paths written, keyed by artefact name.
    """
    import csv
    import os

    os.makedirs(out_dir, exist_ok=True)
    rows = [dict(row) for row in rows]
    written: Dict[str, str] = {}

    space_path = os.path.join(out_dir, "search_space.csv")
    with open(space_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["parameter", "range"])
        writer.writeheader()
        writer.writerows(space.describe())
    written["search_space"] = space_path

    if rows:
        ranked = rank_trials(rows, metric, mode)
        fieldnames = sorted({key for row in rows for key in row})
        trials_path = os.path.join(out_dir, "trials.csv")
        with open(trials_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(ranked)
        written["trials"] = trials_path

        exploration = [row for row in rows if row.get("stage") == "stage1"] or rows
        curve_path = os.path.join(out_dir, "budget_curve.csv")
        with open(curve_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["n_trials", f"best_{metric}"])
            for index, value in enumerate(budget_curve(exploration, metric, mode), start=1):
                writer.writerow([index, value])
        written["budget_curve"] = curve_path

    return written
