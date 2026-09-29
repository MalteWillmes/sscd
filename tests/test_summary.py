"""results/scales_summary.csv."""

import math

import pandas as pd

import sscd


def test_scales_summary_counts_and_means():
    focus = pd.DataFrame({"scale_id": ["A", "B"], "score": [0.99, 0.2], "focus_method": ["standard", "low_threshold"]})
    # A: 3 transects, the one at 90 degrees without circuli; B: 2 transects, both empty; C: no focus
    transects = ["A_0", "A_45", "A_90", "B_0", "B_45"]
    circuli = pd.DataFrame({"scale_id": ["A"] * 7, "angle_deg": [0, 0, 0, 0, 45, 45, 45],
                            "spacing_px": [None, 10, 12, 11, None, 9, 13]})
    s = sscd.scales_summary(["A", "B", "C"], focus, transects, circuli).set_index("scale_id")

    assert s.loc["A", "focus_method"] == "standard" and s.loc["B", "focus_method"] == "low_threshold"
    assert s.loc["A", "n_transects"] == 3
    assert s.loc["A", "total_n_circuli"] == 7
    assert s.loc["A", "mean_n_circuli"] == 3.5          # 7 circuli / 2 transects with circuli
    assert s.loc["A", "median_spacing_px"] == 11.0
    assert s.loc["A", "transects_without_circuli"] == "90"
    assert math.isnan(s.loc["B", "mean_n_circuli"])      # only empty transects
    assert s.loc["B", "transects_without_circuli"] == "0;45"
    assert not s.loc["C", "focus_found"] and s.loc["C", "n_transects"] == 0


def test_scale_ids_with_underscores():
    focus = pd.DataFrame({"scale_id": ["N Esk_2018_1"], "score": [0.9], "focus_method": ["standard"]})
    circuli = pd.DataFrame({"scale_id": ["N Esk_2018_1"], "angle_deg": [45], "spacing_px": [None]})
    s = sscd.scales_summary(["N Esk_2018_1"], focus, ["N Esk_2018_1_45"], circuli)
    assert s.loc[0, "n_transects"] == 1 and s.loc[0, "total_n_circuli"] == 1
