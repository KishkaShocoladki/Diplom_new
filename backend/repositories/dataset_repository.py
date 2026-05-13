from backend.extensions import db
from backend.models import Dataset


def get_by_id(dataset_id: int):
    return Dataset.query.get(dataset_id)


def get_by_hash(file_hash: str):
    return Dataset.query.filter_by(file_hash=file_hash).first()


def create_dataset(name: str, raw_path: str, file_hash: str):
    ds = Dataset(name=name, raw_path=raw_path, file_hash=file_hash)
    db.session.add(ds)
    db.session.commit()
    return ds


def delete_dataset(dataset_id: int):
    ds = Dataset.query.get(dataset_id)
    if ds:
        db.session.delete(ds)
        db.session.commit()
        return True
    return False
from typing import Optional
