from backend.extensions import db
from backend.models import Experiment


def get_by_id(exp_id: int):
    return Experiment.query.get(exp_id)


def create_experiment(**kwargs):
    e = Experiment(**kwargs)
    db.session.add(e)
    db.session.commit()
    return e


def update_experiment(exp: Experiment, **kwargs):
    for k, v in kwargs.items():
        setattr(exp, k, v)
    db.session.commit()
    return exp


