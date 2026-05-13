import os
import uuid
import json
import threading
import time
import hashlib
import logging
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, Response

from flask_cors import CORS

from backend.extensions import db
from backend.logger_config import setup_logging, get_logger
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from werkzeug.utils import secure_filename
from werkzeug.exceptions import ClientDisconnected
import pandas as pd
import torch

from ml_module.preprocessing import (
    load_and_clean,
    prepare_for_training,
    load_and_clean_for_prediction,
    save_preprocessing_artifacts,
    load_preprocessing_artifacts,
)
from ml_module.model import SVModel, _roc_auc
from ml_module.metrics_utils import (
    compute_confusion_matrix,
    compute_roc_curve,
    compute_pr_curve,
    compute_calibration_curve,
    compute_prediction_distribution,
    compute_metrics_vs_threshold,
    compute_confidence_vs_accuracy,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_BASE = Path(os.environ.get("DATASETS_DIR", "D:/datasets"))
DATASETS_RAW = DATASETS_BASE / "raw"
DATASETS_PREP = DATASETS_BASE / "preprocessed"
MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(BASE_DIR / "models")))
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", str(BASE_DIR / "artifacts")))
PREDICTIONS_DIR = Path(os.environ.get("PREDICTIONS_DIR", "D:/predictions"))
RAW_DATA_DIR = Path(os.environ.get("RAW_DATA_DIR", str(DATASETS_BASE / "raw")))

for path in [DATASETS_RAW, DATASETS_PREP, MODELS_DIR, ARTIFACTS_DIR, PREDICTIONS_DIR]:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

setup_logging(level=logging.INFO)
app_logger = get_logger(__name__)

app.config["MAX_CONTENT_LENGTH"] = 7 * 1024 * 1024 * 1024
app.config["DATASETS_DIR"] = str(DATASETS_BASE)
app.config["RAW_DATASETS_DIR"] = str(DATASETS_RAW)
app.config["PREPROCESSED_DIR"] = str(DATASETS_PREP)
app.config["ARTIFACTS_DIR"] = str(ARTIFACTS_DIR)
app.config["PREDICTIONS_DIR"] = str(PREDICTIONS_DIR)
app.config["MODELS_DIR"] = str(MODELS_DIR)

app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/diplom",
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

from backend.models import Model, Dataset, Experiment, Metric, GraphData

from backend.services import task_service, training_service

from backend.controllers import (
    dataset_controller,
    experiment_controller,
    model_controller,
    task_controller,
)
dataset_controller.register_blueprint(app)
experiment_controller.register_blueprint(app)
model_controller.register_blueprint(app)
task_controller.register_blueprint(app)

create_task = task_service.create_task
update_task = task_service.update_task


def _ensure_dataset_file_hash_column():
    from sqlalchemy import text, inspect
    try:
        inspector = inspect(db.engine)
        columns = [col['name'] for col in inspector.get_columns('datasets')]
        if 'file_hash' not in columns:
            print("Adding file_hash column to datasets table...")
            with db.engine.connect() as conn:
                conn.execute(text("ALTER TABLE datasets ADD COLUMN file_hash VARCHAR(64)"))
                conn.commit()
            print("file_hash column added successfully")
    except Exception as exc:
        print(f"Error ensuring file_hash column: {exc}")


def _compute_file_hash(file_path):
    file_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                file_hash.update(chunk)
        return file_hash.hexdigest()
    except FileNotFoundError:
        return None


def _backfill_dataset_hashes():
    try:
        datasets = Dataset.query.filter(Dataset.file_hash.is_(None)).all()
        for dataset in datasets:
            if os.path.exists(dataset.raw_path):
                file_hash = _compute_file_hash(dataset.raw_path)
                if file_hash:
                    existing = Dataset.query.filter_by(file_hash=file_hash).first()
                    if not existing:
                        dataset.file_hash = file_hash
                    else:
                        print(f"Dataset {dataset.id} is a duplicate of {existing.id}")
                        continue
        try:
            db.session.commit()
            return True
        except Exception as exc:
            print(f"Error backfilling hashes: {exc}")
            db.session.rollback()
            return False
    except Exception as exc:
        print(f"Backfill skipped: {exc}")
        return False


def task_stream(task_id):
    return Response(task_service.task_stream(task_id), mimetype="text/event-stream")

def _add_graph_data(experiment_id, graph_type, data_dict):
    graph = GraphData(
        experiment_id=experiment_id,
        graph_type=graph_type,
        data=data_dict,
    )
    db.session.add(graph)
    db.session.commit()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
