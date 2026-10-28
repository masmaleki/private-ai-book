"""Leakage unit tests for the chapter 7 feature builder (run with pytest)."""

import numpy as np
import pandas as pd

from features import FORBIDDEN, build, load

T = pd.Timestamp("2025-03-01")
FEATURES = ["last_price", "med30", "med90", "n30", "trend", "lead_med30"]


def test_future_offers_do_not_change_past_features():
    tables = load()
    before = build(tables, slope=-0.05)
    # vandalise every offer observed after T: prices x10, lead times x3
    o = tables["demo_offers"]
    late = o["observed_at"] >= T
    o.loc[late, "unit_price"] *= 10
    o.loc[late, "lead_days"] *= 3
    after = build(tables, slope=-0.05)
    past = before["issued_at"] < T
    pd.testing.assert_frame_equal(before.loc[past, FEATURES], after.loc[past, FEATURES])


def test_last_offer_is_strictly_before_quote():
    q = build(load(), slope=-0.05)
    has = q["last_offer_at"].notna()
    assert (q.loc[has, "last_offer_at"] < q.loc[has, "issued_at"]).all()


def test_no_forbidden_columns_in_feature_list():
    from experiments import BASE

    assert not FORBIDDEN & set(BASE)


def test_temporal_split_has_no_overlap():
    from experiments import CUTOFF

    q = build(load(), slope=-0.05)
    train, test = q[q.issued_at < CUTOFF], q[q.issued_at >= CUTOFF]
    assert train["issued_at"].max() < test["issued_at"].min()
    assert np.isfinite(test["med30"]).all()
