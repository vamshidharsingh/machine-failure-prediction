"""Training labels from time-to-event records, with censoring handled honestly.

Classes by time steps left before the repair (Scania's definition):

    class 0: more than 48     class 1: 48 to 24     class 2: 24 to 12
    class 3: 12 to 6          class 4: 6 to 0

The paper does not say which class an exact edge value belongs to. Class 0
is "more than 48", so 48 itself is class 1, and every window is read the same
way: it includes its upper edge. So 48 -> 1, 24 -> 2, 12 -> 3, 6 -> 4.
"""

import numpy as np
import pandas as pd

# Upper-inclusive edges, from class 4 (closest to failure) to class 1.
CLASS_EDGES = ((6.0, 4), (12.0, 3), (24.0, 2), (48.0, 1))
HEALTHY_AFTER = 48.0


def time_left_to_class(time_left) -> np.ndarray:
    """Map time steps left before failure to classes 0-4.

    The one place the boundary rule lives.
    """
    t = np.asarray(time_left, dtype=float)
    conditions = [t <= edge for edge, _ in CLASS_EDGES]
    classes = [cls for _, cls in CLASS_EDGES]
    return np.select(conditions, classes, default=0).astype(int)


def label_training(readouts: pd.DataFrame, tte: pd.DataFrame):
    """Label every training readout, or drop it when its class is unknown.

    time_left = length_of_study_time_step - time_step.

    - Repaired vehicles (in_study_repair = 1): time_left is exact, so it maps
      straight to a class.
    - Vehicles not repaired (in_study_repair = 0): the failure, if any, comes
      after the study ended, so the true time left is at least time_left. Above
      48 the readout is certainly class 0; otherwise its class is unknown and
      the readout is dropped from training.
    - Readouts after the repair time step (time_left < 0) are dropped.

    Returns (labels, report). `labels` keeps the readouts' index and has the
    columns vehicle_id, time_step, time_left, class_label.
    """
    info = tte.set_index("vehicle_id")
    length = readouts["vehicle_id"].map(info["length_of_study_time_step"]).astype(float)
    repaired = readouts["vehicle_id"].map(info["in_study_repair"]).astype(int)
    time_left = length - readouts["time_step"].astype(float)

    after_repair = time_left < 0
    censored_unknown = ~after_repair & (repaired == 0) & (time_left <= HEALTHY_AFTER)
    keep = ~after_repair & ~censored_unknown

    labels = pd.DataFrame(
        {
            "vehicle_id": readouts["vehicle_id"],
            "time_step": readouts["time_step"],
            "time_left": time_left,
        }
    )[keep]
    labels["class_label"] = time_left_to_class(labels["time_left"])

    counts = labels["class_label"].value_counts().reindex(range(5), fill_value=0)
    report = {
        "readouts": int(len(readouts)),
        "dropped_after_repair": int(after_repair.sum()),
        "dropped_censored_unknown_class": int(censored_unknown.sum()),
        "labelled_readouts": int(keep.sum()),
        "labelled_vehicles": int(labels["vehicle_id"].nunique()),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
    }
    return labels, report
