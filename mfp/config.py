"""Project settings. Change values here rather than inside the code."""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ALL_SUBSETS = ("FD001", "FD002", "FD003", "FD004")


@dataclass(frozen=True)
class Config:
    """Settings for the NASA C-MAPSS pipeline (python -m mfp.pipeline)."""

    data_dir: Path = ROOT / "data"
    reports_dir: Path = ROOT / "reports"
    models_dir: Path = ROOT / "models"
    subset: str = "FD001"

    seed: int = 42
    # An engine is "at risk" if it will fail within this many cycles.
    failure_window: int = 30
    # Training engines are split three ways by whole engine:
    # fit (the rest, 60%), validation (20%) and lifecycle test (20%).
    val_fraction: float = 0.2
    lifecycle_fraction: float = 0.2
    # Past-only rolling window, in cycles.
    rolling_window: int = 10
    # Remaining useful life is capped here when training the regressor
    # (early in life, engines are effectively healthy).
    rul_cap: int = 125

    # Cost of one missed failure vs one false alarm. These are illustrative
    # assumptions: the dataset contains no real cost figures.
    cost_false_alarm: float = 1.0
    cost_ratio_main: float = 10.0
    cost_ratios_sensitivity: tuple = (5.0, 10.0, 20.0)

    # Lifecycle alarm simulation. These are assumptions, not facts from the data:
    # - an alarm is useful only if it rings between alert_min_lead and
    #   alert_max_lead cycles before failure (inclusive). Fewer than
    #   alert_min_lead cycles is too late to act; more than alert_max_lead
    #   means the engine is serviced long before it needed to be.
    # - the alarm rings once the failure probability has stayed at or above
    #   the threshold for confirm_cycles consecutive cycles; the pipeline tries
    #   each option and keeps the cheapest on validation engines.
    alert_min_lead: int = 10
    alert_max_lead: int = 50
    confirm_cycles_options: tuple = (1, 2, 3)

    # Operating regimes per subset (FD002 and FD004 run under six operating
    # conditions; FD001 and FD003 under one).
    regimes: dict = field(default_factory=lambda: {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6})

    # Warning-time option: the validation lifecycle sweep is also computed for
    # these minimum lead times, so the dashboard can show how the policy
    # changes when more warning is required. alert_min_lead stays the default.
    alert_min_lead_options: tuple = (10, 20, 30)

    # Small tuning grid for gradient boosting, scored by engine-grouped
    # cross-validation on fit engines.
    tuning_grid: dict = field(
        default_factory=lambda: {
            "learning_rate": [0.03, 0.1],
            "max_leaf_nodes": [15, 31],
            "max_iter": [200, 400],
        }
    )
    tuning_folds: int = 3

    @property
    def n_regimes(self) -> int:
        """Operating regimes of the configured subset."""
        return int(self.regimes.get(self.subset, 1))


@dataclass(frozen=True)
class ScaniaConfig:
    """Settings for the Scania Component X pipeline (python -m mfp.scania.pipeline)."""

    data_dir: Path = ROOT / "data" / "scania"
    reports_dir: Path = ROOT / "reports" / "scania"
    models_dir: Path = ROOT / "models" / "scania"
    seed: int = 42

    # Read CSVs once, then load from parquet copies in data_dir/cache/.
    use_cache: bool = True

    # Past-only window for rates of change and changes in histogram shares,
    # counted in readouts (the current readout and the 4 before it).
    change_window: int = 5

    # Use at most this many labelled readouts per training vehicle (a seeded
    # random sample). None = every labelled readout. Only for slow machines.
    max_rows_per_vehicle: int = None

    # Gradient boosting tuning: a small grid scored by multiclass log loss with
    # vehicle-grouped folds on a seeded random share of training vehicles.
    tuning_fraction: float = 0.3
    tuning_folds: int = 3
    tuning_grid: dict = field(
        default_factory=lambda: {
            "learning_rate": [0.05, 0.1],
            "max_leaf_nodes": [15, 31],
            "max_iter": [300],
        }
    )

    # Prior-shift correction: class 0 keeps weight 1; classes 1-4 each take a
    # weight from this grid, chosen on validation vehicles only.
    prior_weight_grid: tuple = (0.25, 0.5, 1.0, 2.0, 4.0)

    # Model selection: vehicle-grouped folds for the simulated validation set,
    # gradient boosting settings (from the grouped-CV tuning on training
    # vehicles), and the feature groups tried in this order.
    selection_folds: int = 5
    # Candidates are compared on a seeded random share of training rows (each
    # vehicle keeps its share of rows); the chosen model is then trained on
    # all rows. 1.0 compares on every row.
    selection_row_fraction: float = 0.4
    boosting_params: dict = field(
        default_factory=lambda: {"learning_rate": 0.05, "max_leaf_nodes": 15, "max_iter": 300}
    )
    feature_groups: tuple = ("intensity", "rate10", "fleet")

    # Workshop capacity: the shares of vehicles that may be flagged, and how
    # close to the unconstrained cost (on the simulated validation set) the
    # recommended capacity must be.
    capacities: tuple = (0.02, 0.05, 0.10, 0.20)
    capacity_tolerance: float = 0.05

    # Bootstrap of test vehicles for 95% intervals.
    bootstrap_samples: int = 1000
