import numpy as np
from refproj import db, price_model
from refproj.features import time_split


def test_time_split_has_no_future_rows(built):
    tr, te = time_split(db.offers_frame(db.connect()))
    assert tr.observed_at.max() < te.observed_at.min()


def test_defence_tenant_never_in_training(built):
    _, _, tr, te = price_model.train(db.offers_frame(db.connect()))
    assert "tenant-d" not in set(tr.tenant_id) | set(te.tenant_id)


def test_model_beats_baseline_and_recovers_discount(built):
    model, feat, card = price_model.load()
    m = card["metrics"]
    assert m["mape_model"] < m["mape_baseline"]
    w = dict(zip(model.columns, model.w[1:]))
    # Data recipe: x0.91 per decade of quantity -> log(0.91) = -0.094
    assert abs(w["log10_qty"] - np.log(0.91)) < 0.02
