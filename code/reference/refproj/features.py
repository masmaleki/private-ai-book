"""Turn offers into a numeric feature matrix. Fit on train data only."""

import numpy as np
import pandas as pd
from .config import SPLIT_DATE, TENANTS


def shared_rows(df):
    """Drop tenants whose data must never enter a shared model."""
    ok = [t for t, c in TENANTS.items() if c["shared_training"]]
    return df[df.tenant_id.isin(ok)]


def time_split(df, split=SPLIT_DATE):
    """Split by time, never randomly: the model must not see the future."""
    return df[df.observed_at < split], df[df.observed_at >= split]


class Featurizer:
    """Learns encodings from train rows; applies them to any rows."""

    def fit(self, train):
        # Median log price per part, learned from TRAIN rows only.
        logp = np.log(train.unit_price)
        self.part_level = logp.groupby(train.mpn).median().to_dict()
        self.global_level = float(np.median(logp))
        self.suppliers = sorted(train.supplier_id.unique().tolist())
        self.t0 = train.observed_at.min()
        return self

    def transform(self, df):
        X = pd.DataFrame(index=df.index)
        lvl = df.mpn.map(self.part_level)
        X["part_level"] = lvl.fillna(self.global_level)
        X["unknown_part"] = lvl.isna().astype(float)
        X["log10_qty"] = np.log10(df.qty)
        X["months"] = (df.observed_at - self.t0).dt.days / 30.44
        for s in self.suppliers[1:]:  # first one is the reference level
            X[f"sup_{s}"] = (df.supplier_id == s).astype(float)
        return X

    def row(self, mpn, qty, supplier_id, when):
        one = pd.DataFrame([dict(mpn=mpn, qty=qty, supplier_id=supplier_id,
                                 observed_at=pd.Timestamp(when))])
        return self.transform(one)

    def state(self):
        return dict(part_level=self.part_level, global_level=self.global_level,
                    suppliers=self.suppliers, t0=str(self.t0.date()))

    @classmethod
    def from_state(cls, st):
        f = cls()
        f.part_level, f.global_level = st["part_level"], st["global_level"]
        f.suppliers, f.t0 = st["suppliers"], pd.Timestamp(st["t0"])
        return f
