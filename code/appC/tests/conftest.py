import pytest
from refproj import data_gen, db, price_model


@pytest.fixture(scope="session")
def built():
    data_gen.build_db()
    model, feat, tr, te = price_model.train(db.offers_frame(db.connect()))
    price_model.save(model, feat, price_model.evaluate(model, feat, te, tr))
    return True


@pytest.fixture()
def tools(built):
    from refproj.tools import Tools
    return Tools(db.connect())
