"""Generated README sections and the cross-dataset comparison table."""

from mfp.report import COMPARISON_COLUMNS, comparison_markdown, comparison_rows, replace_section


def test_replace_section_only_touches_its_markers():
    text = (
        "A\n<!-- X:START -->\nold\n<!-- X:END -->\n"
        "B\n<!-- Y:START -->\nkeep\n<!-- Y:END -->\nFooter\n"
    )
    out = replace_section(text, "X", "new")
    assert (
        out == "A\n<!-- X:START -->\nnew\n<!-- X:END -->\n"
        "B\n<!-- Y:START -->\nkeep\n<!-- Y:END -->\nFooter\n"
    )
    assert replace_section(text, "MISSING", "new") is None


def _at_risk(pr_auc, precision, recall, fa):
    return {"pr_auc": pr_auc, "precision": precision, "recall": recall, "false_alarms": fa}


def test_comparison_rows_leave_cells_empty_where_metrics_do_not_apply():
    scania = {
        "test": {
            "by_model": {
                "gradient_boosting": {"total_cost": 100.0, "at_risk": _at_risk(0.1, 0.05, 0.6, 30)},
                "logistic_regression": {
                    "total_cost": 120.0,
                    "at_risk": _at_risk(0.08, 0.04, 0.5, 35),
                },
                "always_class_0": {"total_cost": 150.0, "at_risk": _at_risk(None, None, 0.0, 0)},
            }
        }
    }
    tl = {
        "pr_auc": 0.9,
        "at_chosen_threshold": {"precision": 0.8, "recall": 0.7, "false_alarms": 2},
    }
    nasa = {
        "FD001": {
            "test_last_cycle": tl,
            "lifecycle_test": {"chosen": {"cost": 3.0}},
            "rul": {
                "rmse_model": 18.0,
                "nasa_score_model": 700.0,
                "rmse_ridge": 22.0,
                "nasa_score_ridge": 1500.0,
            },
            "baseline_logistic_regression": {
                "test_last_cycle": {**tl, "pr_auc": 0.85},
                "lifecycle_test": {"chosen": {"cost": 4.0}},
            },
        }
    }
    rows = comparison_rows(nasa, scania)
    assert [(r["dataset"], r["model"]) for r in rows] == [
        ("Scania", "gradient_boosting"),
        ("Scania", "logistic_regression"),
        ("Scania", "always_class_0"),
        ("FD001", "gradient_boosting"),
        ("FD001", "logistic_regression"),
    ]
    assert all(set(r) == set(COMPARISON_COLUMNS) for r in rows)
    sc, base, gb, lr = rows[0], rows[2], rows[3], rows[4]
    assert sc["official_cost"] == 100 and sc["lifecycle_cost"] is None and sc["rmse"] is None
    assert base["pr_auc"] is None and base["precision"] is None and base["recall"] == 0.0
    assert gb["official_cost"] is None and gb["lifecycle_cost"] == 3 and gb["nasa_score"] == 700
    # The logistic NASA row carries the ridge remaining-life baseline.
    assert (lr["rmse"], lr["nasa_score"], lr["lifecycle_cost"], lr["pr_auc"]) == (22, 1500, 4, 0.85)
    md = comparison_markdown(rows)
    assert "| Scania | Always class 0 |  |  | 0.000 | 0 | 150 |  |  |  |" in md
    assert "Empty cells" in md
