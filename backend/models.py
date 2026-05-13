from datetime import datetime
from backend.extensions import db


class Model(db.Model):
    __tablename__ = "models"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False, unique=True)
    input_dim = db.Column(db.Integer, nullable=True)
    dropout = db.Column(db.Float, nullable=False, default=0.3)
    path = db.Column(db.String(512), nullable=True)
    version = db.Column(db.Integer, default=1)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    experiments = db.relationship("Experiment", backref="model", lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "input_dim": self.input_dim,
            "dropout": self.dropout,
            "path": self.path,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class Dataset(db.Model):
    __tablename__ = "datasets"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(256), nullable=False)
    raw_path = db.Column(db.String(512), nullable=False)
    file_hash = db.Column(db.String(64), nullable=True, unique=True)
    preprocessed_path = db.Column(db.String(512), nullable=True)
    artifacts_path = db.Column(db.String(512), nullable=True)
    has_label = db.Column(db.Boolean, default=False)
    row_count = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    experiments = db.relationship(
        "Experiment",
        backref="dataset",
        lazy=True,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def to_dict(self):
        training_kinds = {"train", "fine_tune", "retrain"}
        used_model_ids = list({e.model_id for e in (self.experiments or []) if getattr(e, 'model_id', None) is not None and getattr(e, 'kind', None) in training_kinds})
        return {
            "id": self.id,
            "name": self.name,
            "raw_path": self.raw_path,
            "file_hash": self.file_hash,
            "preprocessed_path": self.preprocessed_path,
            "artifacts_path": self.artifacts_path,
            "has_label": self.has_label,
            "row_count": self.row_count,
            "used_by_model_ids": used_model_ids,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class Experiment(db.Model):
    __tablename__ = "experiments"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(256), nullable=False)
    kind = db.Column(db.String(64), nullable=False)
    model_id = db.Column(db.Integer, db.ForeignKey("models.id", ondelete="SET NULL"), nullable=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=True)
    parent_experiment_id = db.Column(db.Integer, nullable=True)
    model_version_at_time = db.Column(db.Integer, nullable=True)
    status = db.Column(db.String(64), default="pending")
    message = db.Column(db.String(1024), nullable=True)
    result_data = db.Column(db.JSON, default=dict)
    comparison_with_parent = db.Column(db.JSON, nullable=True)
    predictions_path = db.Column(db.String(512), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "parent_experiment_id": self.parent_experiment_id,
            "model_version_at_time": self.model_version_at_time,
            "status": self.status,
            "message": self.message,
            "result_data": self.result_data,
            "comparison_with_parent": self.comparison_with_parent,
            "predictions_path": self.predictions_path,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class Metric(db.Model):
    __tablename__ = "metrics"
    id = db.Column(db.Integer, primary_key=True)
    experiment_id = db.Column(db.Integer, db.ForeignKey("experiments.id"), nullable=False)
    type = db.Column(db.String(32), nullable=False, default='final')
    value = db.Column(db.JSON, nullable=False)


class GraphData(db.Model):
    __tablename__ = "graph_data"
    id = db.Column(db.Integer, primary_key=True)
    experiment_id = db.Column(db.Integer, db.ForeignKey("experiments.id"), nullable=False)
    graph_type = db.Column(db.String(128), nullable=False)
    data = db.Column(db.JSON, nullable=False)
