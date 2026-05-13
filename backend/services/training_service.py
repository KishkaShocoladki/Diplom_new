import os
import threading
import logging
from datetime import datetime

import pandas as pd
import torch

from backend.extensions import db
from backend.models import Model, Dataset, Experiment, Metric, GraphData
from backend.logger_config import get_logger
from ml_module.preprocessing import prepare_for_training
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

from backend.services import task_service
from backend.utils.exporter import export_experiment

logger = get_logger(__name__)


def _get_comparison_metrics(parent_exp_id: int, current_test_metrics: dict) -> dict:
    try:
        parent_exp = Experiment.query.get(parent_exp_id)
        if not parent_exp:
            logger.warning(f"Parent experiment with ID {parent_exp_id} not found")
            return None
        
        parent_result_data = parent_exp.result_data or {}
        parent_test_metrics = parent_result_data.get("test_metrics", {})
        
        if not parent_test_metrics:
            logger.warning(f"Parent experiment {parent_exp_id} has no test metrics")
            return None
        
        comparison = {}
        for metric_name in ["accuracy", "precision", "recall", "f1", "auc"]:
            parent_val = parent_test_metrics.get(metric_name)
            current_val = current_test_metrics.get(metric_name)
            
            if parent_val is not None and current_val is not None:
                delta = float(current_val) - float(parent_val)
                percent_improvement = (delta / float(parent_val) * 100) if parent_val != 0 else 0
                comparison[f"{metric_name}_parent"] = round(float(parent_val), 4)
                comparison[f"{metric_name}_current"] = round(float(current_val), 4)
                comparison[f"{metric_name}_delta"] = round(delta, 4)
                comparison[f"{metric_name}_improvement_percent"] = round(percent_improvement, 2)
        
        if comparison:
            logger.info(f"Metrics comparison calculated for experiment {parent_exp_id}: {comparison}")
        return comparison if comparison else None
    except Exception as e:
        logger.error(f"Error in _get_comparison_metrics: {str(e)}", exc_info=True)
        return None


def _append_metric(experiment_id, step, value_dict):
    try:
        metric = Metric(experiment_id=experiment_id, value=value_dict, type='per-epoch')
        db.session.add(metric)
        db.session.commit()
    except Exception:
        db.session.rollback()


def _add_graph_data(experiment_id, graph_type, data_dict):
    graph = GraphData(experiment_id=experiment_id, graph_type=graph_type, data=data_dict)
    db.session.add(graph)
    db.session.commit()


def model_training_worker(app, tx_id, experiment_id, dataset_id, model_name, mode, params, parent_experiment_id=None):
    def _worker():
        with app.app_context():
            try:
                logger.info(f"[TRAINING] Starting {mode} for experiment {experiment_id}, model '{model_name}'")
                
                exp = Experiment.query.get(experiment_id)
                exp.status = "running"
                exp.message = f"{mode} started"
                exp.parent_experiment_id = parent_experiment_id
                db.session.commit()

                dataset = Dataset.query.get(dataset_id)
                if dataset is None:
                    logger.error(f"Dataset with ID {dataset_id} not found")
                    raise RuntimeError(f"Dataset with ID {dataset_id} not found in database")
                if dataset.preprocessed_path is None:
                    logger.error(f"Dataset '{dataset.name}' (ID: {dataset_id}) not preprocessed")
                    raise RuntimeError(f"Dataset '{dataset.name}' (ID: {dataset_id}) has not been preprocessed yet. Please preprocess it first.")

                logger.info(f"[DATA] Loading preprocessed dataset: {dataset.preprocessed_path}")
                df = pd.read_csv(dataset.preprocessed_path, sep="\t")
                logger.info(f"[DATA] Loaded {len(df):,} rows, {len(df.columns)} columns")
                
                if "label" not in df.columns:
                    logger.error("Dataset missing 'label' column")
                    raise RuntimeError("Dataset has no label column for training")

                def _prep_progress(pct, message):
                    try:
                        if pct is None:
                            task_service.update_task(tx_id, status="running", message=message)
                        else:
                            task_service.update_task(tx_id, status="running", progress=int(pct), message=message)
                    except Exception:
                        pass

                logger.info(f"[DATA] Preparing train/val/test splits...")
                X_train, X_val, X_test, y_train, y_val, y_test = prepare_for_training(df, progress_callback=_prep_progress)
                logger.info(f"[DATA] Splits: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}")
                
                sv_ids = None
                if "sv_id" in df.columns and "gene_id" in df.columns:
                    sv_ids = (df["sv_id"].astype(str) + "__" + df["gene_id"].astype(str)).values

                model = None
                existing_model = Model.query.filter_by(name=model_name).first()
                if existing_model and existing_model.path and os.path.isfile(existing_model.path):
                    logger.info(f"[MODEL] Loading existing model from {existing_model.path}")
                    model = SVModel.load(existing_model.path)
                    existing_model.input_dim = model.input_dim
                    existing_model.dropout = model.dropout
                    existing_model.updated_at = datetime.utcnow()
                    db.session.commit()
                else:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    logger.info(f"[MODEL] Creating new model (device={device})")
                    model = SVModel(input_dim=X_train.shape[1], dropout=params.get("dropout", 0.3), device=device)
                    if not existing_model:
                        existing_model = Model(name=model_name, input_dim=X_train.shape[1], dropout=params.get("dropout", 0.3))
                        db.session.add(existing_model)
                        db.session.commit()

                model_id = existing_model.id
                exp.model_version_at_time = existing_model.version
                db.session.commit()

                def progress_cb(epoch_info):
                    payload = {
                        "status": "running",
                        "progress": int(100 * epoch_info["epoch"] / params["epochs"]),
                        "message": f"Epoch {epoch_info['epoch']} / {epoch_info['epochs']}",
                        "epoch": epoch_info.get("epoch"),
                        "epochs": epoch_info.get("epochs"),
                        "train_loss": epoch_info.get("train_loss"),
                        "val_loss": epoch_info.get("val_loss"),
                        "val_acc": epoch_info.get("val_acc"),
                        "val_auc": epoch_info.get("val_auc"),
                        "lr": epoch_info.get("lr"),
                    }
                    task_service.update_task(tx_id, **payload)
                    try:
                        _append_metric(experiment_id=exp.id, step=epoch_info.get("epoch", 0), value_dict=epoch_info)
                    except Exception:
                        db.session.rollback()

                if mode == "train":
                    logger.info(f"[TRAINING] Starting train mode: epochs={params['epochs']}, batch_size={params['batch_size']}, lr={params['lr']}, optimizer={params.get('optimizer', 'adamw')}")
                    model.train(
                        X_train, y_train, X_val, y_val,
                        epochs=params["epochs"], batch_size=params["batch_size"], lr=params["lr"],
                        sv_ids=sv_ids, verbose_every=params.get("verbose_every", 5), progress_callback=progress_cb,
                        use_early_stopping=params.get("use_early_stopping", False),
                        early_stopping_patience=params.get("early_stopping_patience", 10),
                        optimizer_name=params.get("optimizer", "adamw"),
                    )
                elif mode == "fine_tune":
                    logger.info(f"[TRAINING] Starting fine_tune mode: epochs={params['epochs']}, batch_size={params['batch_size']}, lr={params['lr']}, optimizer={params.get('optimizer', 'adamw')}, parent_exp={parent_experiment_id}")
                    model.fine_tune(
                        X_train, y_train, X_val, y_val,
                        sv_ids=sv_ids, epochs=params["epochs"], batch_size=params["batch_size"], lr=params["lr"],
                        verbose_every=params.get("verbose_every", 5), progress_callback=progress_cb,
                        use_early_stopping=params.get("use_early_stopping", False),
                        early_stopping_patience=params.get("early_stopping_patience", 5),
                        optimizer_name=params.get("optimizer", "adamw"),
                    )
                elif mode == "retrain":
                    logger.info(f"[TRAINING] Starting retrain mode: epochs={params['epochs']}, batch_size={params['batch_size']}, lr={params['lr']}, optimizer={params.get('optimizer', 'adamw')}, parent_exp={parent_experiment_id}")
                    model.retrain(
                        X_train, y_train, X_val, y_val,
                        sv_ids=sv_ids, epochs=params["epochs"], batch_size=params["batch_size"], lr=params["lr"],
                        verbose_every=params.get("verbose_every", 5), progress_callback=progress_cb,
                        use_early_stopping=params.get("use_early_stopping", False),
                        early_stopping_patience=params.get("early_stopping_patience", 10),
                        optimizer_name=params.get("optimizer", "adamw"),
                    )
                    existing_model.version += 1
                    db.session.commit()
                    logger.info(f"[MODEL] Model version incremented to {existing_model.version}")
                else:
                    logger.error(f"Unsupported training mode: {mode}")
                    raise RuntimeError("Unsupported training mode")

                model_filename = f"{model_id}_{model_name.replace(' ', '_')}.pt"
                model_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "models", model_filename)
                logger.info(f"[MODEL] Saving model to {model_path}")
                model.save(model_path)

                existing_model.path = str(model_path)
                existing_model.input_dim = X_train.shape[1]
                existing_model.updated_at = datetime.utcnow()
                db.session.commit()

                exp.model_id = model_id
                exp.status = "completed"
                exp.message = "Training completed"
                
                logger.info(f"[EVALUATION] Evaluating on test set ({len(X_test):,} samples)")
                final_test_metrics = model.evaluate(X_test, y_test)
                logger.info(f"[RESULTS] Test metrics: accuracy={final_test_metrics.get('accuracy', 'N/A'):.4f}, auc={final_test_metrics.get('auc', 'N/A'):.4f}")
                
                exp.result_data = {
                    "test_metrics": final_test_metrics,
                    "train_rows": len(X_train),
                    "val_rows": len(X_val),
                    "test_rows": len(X_test),
                }
                
                if parent_experiment_id:
                    logger.info(f"[COMPARISON] Computing metrics comparison with parent experiment {parent_experiment_id}")
                    exp.comparison_with_parent = _get_comparison_metrics(parent_experiment_id, final_test_metrics)
                
                db.session.commit()

                try:
                    m = Metric(experiment_id=exp.id, value={
                        "test_metrics": final_test_metrics,
                        "train_rows": len(X_train),
                        "val_rows": len(X_val),
                        "test_rows": len(X_test),
                    }, type='final')
                    db.session.add(m)
                    db.session.commit()
                except Exception:
                    db.session.rollback()

                try:
                    logger.info(f"[GRAPHS] Generating training graphs for experiment {exp.id}")
                    
                    y_pred_proba = model.predict_proba(X_test)
                    y_pred = (y_pred_proba >= 0.5).astype(int).flatten()
                    
                    cm = compute_confusion_matrix(y_test, y_pred)
                    _add_graph_data(exp.id, "confusion_matrix", cm)
                    
                    fpr, tpr, roc_auc = compute_roc_curve(y_test, y_pred_proba.flatten())
                    _add_graph_data(exp.id, "roc_curve", {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc)})
                    
                    recall, precision, pr_auc = compute_pr_curve(y_test, y_pred_proba.flatten())
                    _add_graph_data(exp.id, "pr_curve", {"recall": recall.tolist(), "precision": precision.tolist(), "auc": float(pr_auc)})
                    
                    cal_probs, cal_fracs = compute_calibration_curve(y_test, y_pred_proba.flatten())
                    _add_graph_data(exp.id, "calibration", {"probs": cal_probs.tolist(), "fracs": cal_fracs.tolist()})
                    
                    metrics_vs_thresh = compute_metrics_vs_threshold(y_test, y_pred_proba.flatten())
                    _add_graph_data(exp.id, "metrics_vs_threshold", metrics_vs_thresh)
                    
                    conf_vs_acc = compute_confidence_vs_accuracy(y_test, y_pred, y_pred_proba.flatten())
                    _add_graph_data(exp.id, "confidence_vs_accuracy", conf_vs_acc)
                    
                    logger.info(f"[GRAPHS] Training graphs saved for experiment {exp.id}")
                except Exception as e:
                    logger.error(f"[GRAPHS] Failed to generate training graphs for experiment {exp.id}: {str(e)}", exc_info=True)

                task_service.update_task(tx_id, status="completed", progress=100, message="Training completed")
                try:
                    try:
                        export_experiment(exp.id, formats=["csv", "json", "pdf"])
                    except Exception:
                        pass
                except Exception:
                    pass

            except Exception as exc:
                logger.error(f"[ERROR] Training failed for experiment {experiment_id}: {str(exc)}", exc_info=True)
                exp = Experiment.query.get(experiment_id)
                if exp:
                    exp.status = "failed"
                    exp.message = str(exc)
                    db.session.commit()
                task_service.update_task(tx_id, status="failed", progress=100, message=str(exc))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

def run_training(*args, **kwargs):
    raise NotImplementedError("Move training worker logic from app.py into this function when ready.")
