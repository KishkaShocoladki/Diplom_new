from backend.extensions import db
from backend.models import Model


def get_by_id(model_id: int):
    return Model.query.get(model_id)


def get_by_name(name: str):
    return Model.query.filter_by(name=name).first()


def create_or_update_model(name: str, input_dim: int = None, dropout: float = None, path: str = None):
    m = get_by_name(name)
    if not m:
        m = Model(name=name, input_dim=input_dim, dropout=dropout, path=path)
        db.session.add(m)
    else:
        if input_dim is not None:
            m.input_dim = input_dim
        if dropout is not None:
            m.dropout = dropout
        if path is not None:
            m.path = path
    db.session.commit()
    return m
