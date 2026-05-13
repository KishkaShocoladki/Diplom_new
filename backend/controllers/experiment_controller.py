from flask import Blueprint, request, jsonify, current_app, send_file
from datetime import datetime
import os
import logging

from backend.extensions import db
from backend.logger_config import get_logger
from backend.models import Experiment, Metric, GraphData, Model, Dataset
from backend.services import training_service, task_service, prediction_service
from backend.utils.exporter import export_experiment
import io
import zipfile

logger = get_logger(__name__)
import tempfile
from flask import send_file

bp = Blueprint("experiments", __name__, url_prefix="/api/experiments")


@bp.route("/", methods=["GET"])
def list_experiments_controller():
    experiments = Experiment.query.order_by(Experiment.created_at.desc()).all()
    result = []
    for e in experiments:
        result.append({
            "id": e.id,
            "name": e.name,
            "kind": e.kind,
            "status": e.status,
            "message": e.message,
            "model_id": e.model_id,
            "dataset_id": e.dataset_id,
            "parent_experiment_id": e.parent_experiment_id,
            "model_version_at_time": e.model_version_at_time,
            "result_data": e.result_data,
            "predictions_path": e.predictions_path,
            "created_at": e.created_at.isoformat(),
            "updated_at": e.updated_at.isoformat(),
        })
    return jsonify(result)


@bp.route("/<int:experiment_id>", methods=["GET"])
def get_experiment_controller(experiment_id: int):
    e = Experiment.query.get_or_404(experiment_id)
    metric_rows = Metric.query.filter_by(experiment_id=e.id).order_by(Metric.id).all()
    per_epoch = []
    final_metrics = []
    for m in metric_rows:
        typ = getattr(m, 'type', None)
        if typ is None or typ == '':
            typ = 'final'
        if typ == 'per-epoch':
            per_epoch.append(m.value)
        else:
            final_metrics.append(m.value)

    metrics_out = per_epoch if per_epoch else final_metrics

    graphs = GraphData.query.filter_by(experiment_id=e.id).all()
    graph_dict = {g.graph_type: g.data for g in graphs}
    return jsonify({
        "id": e.id,
        "name": e.name,
        "kind": e.kind,
        "status": e.status,
        "message": e.message,
        "model_id": e.model_id,
        "dataset_id": e.dataset_id,
        "parent_experiment_id": e.parent_experiment_id,
        "model_version_at_time": e.model_version_at_time,
        "metrics": metrics_out,
        "result_data": e.result_data,
        "comparison_with_parent": e.comparison_with_parent,
        "graphs": graph_dict,
        "predictions_path": e.predictions_path,
        "created_at": e.created_at.isoformat(),
        "updated_at": e.updated_at.isoformat(),
    })


@bp.route("/<int:experiment_id>/graphs", methods=["GET"])
def get_experiment_graphs_controller(experiment_id: int):
    e = Experiment.query.get_or_404(experiment_id)
    graphs = GraphData.query.filter_by(experiment_id=e.id).all()
    graph_dict = {g.graph_type: g.data for g in graphs}
    return jsonify(graph_dict)


@bp.route("/<int:experiment_id>/predictions", methods=["GET"])
def get_experiment_predictions_controller(experiment_id: int):
    e = Experiment.query.get_or_404(experiment_id)
    if e.predictions_path is None or not os.path.isfile(e.predictions_path):
        return jsonify({"error": "Predictions file not found"}), 404

    page = request.args.get("page", default=1, type=int)
    per_page = request.args.get("per_page", default=50, type=int)

    try:
        import pandas as pd
        import numpy as np

        df = pd.read_csv(e.predictions_path, sep="\t")

        total = len(df)
        start = (page - 1) * per_page
        end = min(start + per_page, total)
        df_page = df.iloc[start:end].copy()

        df_page = df_page.replace([np.inf, -np.inf], None)
        df_page = df_page.where(pd.notna(df_page), None)

        records = df_page.to_dict(orient="records")
        for record in records:
            for key, value in record.items():
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    record[key] = None

        response_dict = {
            "total": total,
            "page": page,
            "per_page": per_page,
            "predictions": records
        }

        return jsonify(response_dict)

    except Exception as exc:
        logger.error(f"[GET_PREDICTIONS] Error: {exc}", exc_info=True)
        return jsonify({"error": str(exc)}), 500


@bp.route("/<int:experiment_id>/export", methods=["POST"])
def export_experiment_controller(experiment_id: int):
    logger.info(f"[EXPORT] Export request for experiment {experiment_id}")
    
    e = Experiment.query.get_or_404(experiment_id)
    payload = request.get_json(silent=True) or {}
    formats = payload.get("formats", None)

    if not formats:
        if e.kind == "predict":
            formats = ["pdf", "csv", "xlsx"]
        else:
            formats = ["pdf"]

    if isinstance(formats, str):
        formats = [f.strip().lower() for f in formats.split(',') if f.strip()]
    formats = [f.lower() for f in formats]
    formats = [f for f in formats if f in ("pdf", "csv", "xlsx", "json")]

    if not formats:
        logger.warning(f"[EXPORT] No valid formats for experiment {experiment_id}")
        return jsonify({"error": "No valid formats requested. Allowed: pdf, csv, xlsx, json"}), 400

    try:
        logger.info(f"[EXPORT] Calling export_experiment with formats={formats}")
        written = export_experiment(e.id, formats=formats)
    except Exception as exc:
        logger.error(f"[EXPORT] Exception during export: {str(exc)}", exc_info=True)
        return jsonify({"error": str(exc)}), 500

    if not written:
        logger.error(f"[EXPORT] Export produced no files for experiment {experiment_id}")
        return jsonify({"error": "Export produced no files"}), 500

    logger.info(f"[EXPORT] Export successful: {len(written)} files for experiment {experiment_id}")
    
    if len(written) == 1:
        fpath = written[0]
        ext = os.path.splitext(fpath)[1].lower()
        mime_map = {
            ".pdf":  "application/pdf",
            ".csv":  "text/csv",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".json": "application/json",
        }
        mime = mime_map.get(ext, "application/octet-stream")
        return send_file(
            fpath,
            mimetype=mime,
            as_attachment=True,
            download_name=os.path.basename(fpath),
        )

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in written:
            zf.write(fpath, arcname=os.path.basename(fpath))
    zip_buf.seek(0)

    zip_name = f"experiment_{experiment_id}_export.zip"
    return send_file(
        zip_buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=zip_name,
    )


@bp.route("/start", methods=["POST"])
def start_experiment_controller():
    payload = request.get_json(force=True)
    for key in ["dataset_id", "model_name", "mode"]:
        if key not in payload:
            return jsonify({"error": f"{key} required"}), 400
    dataset_id = int(payload["dataset_id"])
    model_name = payload["model_name"]
    mode = payload["mode"]
    if mode not in ["train", "predict", "fine_tune", "retrain"]:
        return jsonify({"error": f"Invalid mode: {mode}. Supported modes: train, predict, fine_tune, retrain"}), 400
    parent_experiment_id = payload.get("parent_experiment_id")
    if parent_experiment_id:
        parent_experiment_id = int(parent_experiment_id)

    architecture = str(payload.get("architecture", "cnn")).lower()
    if architecture not in {"cnn"}:
        return jsonify({"error": f"Unsupported architecture: {architecture}. Supported: cnn"}), 400

    optimizer = str(payload.get("optimizer", "adamw")).lower()
    if optimizer not in {"adamw", "adam", "sgd"}:
        return jsonify({"error": f"Unsupported optimizer: {optimizer}. Supported: adamw, adam, sgd"}), 400

    use_early_stopping = payload.get("use_early_stopping", False)
    if isinstance(use_early_stopping, str):
        use_early_stopping = use_early_stopping.lower() in {"1", "true", "yes", "on"}
    params = {
        "epochs": int(payload.get("epochs", 60)),
        "batch_size": int(payload.get("batch_size", 512)),
        "lr": float(payload.get("lr", 1e-3)),
        "dropout": float(payload.get("dropout", 0.3)),
        "optimizer": optimizer,
        "architecture": architecture,
        "use_early_stopping": bool(use_early_stopping),
        "early_stopping_patience": int(payload.get("early_stopping_patience", 10)),
        "verbose_every": int(payload.get("verbose_every", 5)),
    }
    exp = Experiment(name=f"{mode}_{model_name}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}", kind=mode, dataset_id=dataset_id, status="pending", message="queued")
    db.session.add(exp)
    db.session.commit()
    task_id = task_service.create_task(f"Experiment {mode} queued")
    training_service.model_training_worker(current_app._get_current_object(), task_id, exp.id, dataset_id, model_name, mode, params, parent_experiment_id=parent_experiment_id)
    return jsonify({"task_id": task_id, "experiment_id": exp.id})


@bp.route("/predict", methods=["POST"])
def start_prediction_controller():
    payload = request.get_json(force=True)
    for key in ["dataset_id", "model_id"]:
        if key not in payload:
            return jsonify({"error": f"{key} required"}), 400
    dataset_id = int(payload["dataset_id"])
    model_id = int(payload["model_id"])
    threshold = float(payload.get("threshold", 0.5))
    exp = Experiment(name=f"predict_{model_id}_{dataset_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}", kind="predict", dataset_id=dataset_id, model_id=model_id, status="pending", message="queued")
    db.session.add(exp)
    db.session.commit()
    task_id = task_service.create_task("Prediction queued")
    prediction_service.prediction_worker(current_app._get_current_object(), task_id, exp.id, dataset_id, model_id, threshold)
    return jsonify({"task_id": task_id, "experiment_id": exp.id})


@bp.route("/model/<int:model_id>/history", methods=["GET"])
def get_model_training_history_controller(model_id: int):
    training_kinds = {"train", "fine_tune", "retrain"}
    experiments = Experiment.query.filter(
        Experiment.model_id == model_id,
        Experiment.kind.in_(training_kinds),
        Experiment.status == "completed"
    ).order_by(Experiment.created_at.asc()).all()
    
    result = []
    for e in experiments:
        result.append({
            "id": e.id,
            "name": e.name,
            "kind": e.kind,
            "model_id": e.model_id,
            "model_version_at_time": e.model_version_at_time,
            "parent_experiment_id": e.parent_experiment_id,
            "status": e.status,
            "result_data": e.result_data,
            "comparison_with_parent": e.comparison_with_parent,
            "created_at": e.created_at.isoformat(),
        })
    return jsonify(result)


def register_blueprint(app):
    app.register_blueprint(bp)


@bp.route("/<int:experiment_id>", methods=["DELETE"])
def delete_experiment_controller(experiment_id: int):
    e = Experiment.query.get_or_404(experiment_id)
    try:
        try:
            for tid, t in list(task_service.TASKS.items()):
                try:
                    if t.get('experiment_id') == e.id or t.get('experiment_id') == str(e.id):
                        task_service.cancel_task(tid)
                except Exception:
                    continue
        except Exception:
            pass

        GraphData.query.filter_by(experiment_id=e.id).delete()
        Metric.query.filter_by(experiment_id=e.id).delete()

        if e.predictions_path and os.path.isfile(e.predictions_path):
            try:
                os.remove(e.predictions_path)
            except Exception:
                pass

        db.session.delete(e)
        db.session.commit()
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
