import os
import threading
import logging
import pandas as pd
import numpy as np

from backend.extensions import db
from backend.logger_config import get_logger
from backend.models import Model, Dataset, Experiment, GraphData, Metric
from ml_module.model import SVModel, _roc_auc
from ml_module.preprocessing import load_preprocessing_artifacts, load_and_clean_for_prediction
from ml_module.metrics_utils import (
    compute_confusion_matrix,
    compute_roc_curve,
    compute_pr_curve,
    compute_calibration_curve,
    compute_prediction_distribution,
    compute_metrics_vs_threshold,
    compute_confidence_vs_accuracy,
)
from ml_module.interpretation import (
    generate_better_explanations,
    create_sv_statistics,
    create_sv_summary_text,
)

from backend.services import task_service
from backend.utils.exporter import export_experiment

logger = get_logger(__name__)


def _add_graph_data(experiment_id, graph_type, data_dict):
    graph = GraphData(experiment_id=experiment_id, graph_type=graph_type, data=data_dict)
    db.session.add(graph)
    db.session.commit()


def prediction_worker(app, tx_id, experiment_id, dataset_id, model_id, threshold):
    def _worker():
        with app.app_context():
            try:
                logger.info(f"[PREDICTION] Starting prediction for experiment {experiment_id}, model {model_id}, dataset {dataset_id}")
                
                exp = Experiment.query.get(experiment_id)
                exp.status = "running"
                exp.message = "Predicting"
                db.session.commit()

                dataset = Dataset.query.get(dataset_id)
                if dataset is None:
                    logger.error(f"Dataset with ID {dataset_id} not found")
                    raise RuntimeError(f"Dataset with ID {dataset_id} not found in database")
                if dataset.preprocessed_path is None:
                    logger.error(f"Dataset '{dataset.name}' not preprocessed")
                    raise RuntimeError(f"Dataset '{dataset.name}' (ID: {dataset_id}) has not been preprocessed yet")
                if dataset.artifacts_path is None:
                    logger.error(f"Dataset '{dataset.name}' artifacts not found")
                    raise RuntimeError(f"Dataset '{dataset.name}' (ID: {dataset_id}) artifacts not found. Please preprocess it again.")

                if model_id is None:
                    raise RuntimeError("Model id is required for prediction")

                model_object = Model.query.get(model_id)
                if model_object is None or model_object.path is None or not os.path.isfile(model_object.path):
                    raise RuntimeError("Model not found for prediction")

                df_pre = None
                sv_ids = []
                if dataset.preprocessed_path and os.path.isfile(dataset.preprocessed_path):
                    try:
                        task_service.update_task(tx_id, progress=5, message="Loading preprocessed dataset")
                    except Exception:
                        pass
                    df_pre = pd.read_csv(dataset.preprocessed_path, sep="\t")
                    if "sv_id" in df_pre.columns:
                        sv_ids = df_pre["sv_id"].astype(str).values
                    if "label" in df_pre.columns:
                        df_for_pred = df_pre.drop(columns=["label"]).copy()
                    else:
                        df_for_pred = df_pre.copy()
                    X = df_for_pred.values.astype("float32") if not df_for_pred.empty else pd.DataFrame().values
                else:
                    scaler, encoders = load_preprocessing_artifacts(dataset.artifacts_path)
                    try:
                        task_service.update_task(tx_id, progress=5, message="Preprocessing artifacts loaded")
                    except Exception:
                        pass

                    def _cb(pct, msg):
                        try:
                            pct_val = pct if pct is not None else None
                            task_service.update_task(tx_id, progress=pct_val if pct_val is not None else 10, message=msg)
                        except Exception:
                            pass

                    X, sv_ids, feature_names = load_and_clean_for_prediction(dataset.preprocessed_path, scaler, encoders, progress_callback=_cb)

                    try:
                        task_service.update_task(tx_id, progress=50, message="Data cleaned and transformed")
                    except Exception:
                        pass

                model = SVModel.load(model_object.path)
                try:
                    task_service.update_task(tx_id, progress=65, message="Model loaded")
                except Exception:
                    pass
                labels, probs = model.predict(X, threshold=threshold)
                try:
                    task_service.update_task(tx_id, progress=85, message=f"Predicted {len(X)} rows")
                except Exception:
                    pass

                n = len(labels)
                needed = ["sv_id", "sv_type", "svtype", "chrom", "chromosome", "name", "gene_name", "gene_id", 
                         "gene_type", "label", "median_tpm", "mean_tpm", "max_tpm", "log2_tpm", "tau", 
                         "pli", "lof_oe", "mis_z", "expressed_tissues", "af", "svlen"]
                try:
                    df_raw = pd.read_csv(dataset.raw_path, sep="\t", usecols=lambda c: c in needed, nrows=n)
                except Exception:
                    df_raw = pd.read_csv(dataset.raw_path, sep="\t", nrows=n)

                info_map = {}
                if "sv_id" in df_raw.columns:
                    info_map["sv_id"] = df_raw["sv_id"].astype(str).values
                if "sv_type" in df_raw.columns:
                    info_map["sv_type"] = df_raw["sv_type"].values
                elif "svtype" in df_raw.columns:
                    info_map["sv_type"] = df_raw["svtype"].values
                if "chrom" in df_raw.columns:
                    info_map["chrom"] = df_raw["chrom"].values
                elif "chromosome" in df_raw.columns:
                    info_map["chrom"] = df_raw["chromosome"].values
                if "gene_name" in df_raw.columns:
                    info_map["gene_name"] = df_raw["gene_name"].values
                if "gene_id" in df_raw.columns:
                    info_map["gene_id"] = df_raw["gene_id"].values
                if "gene_type" in df_raw.columns:
                    info_map["gene_type"] = df_raw["gene_type"].values
                
                bio_features = ["median_tpm", "mean_tpm", "max_tpm", "log2_tpm", "tau", 
                               "pli", "lof_oe", "mis_z", "expressed_tissues", "af", "svlen"]
                for feat in bio_features:
                    if feat in df_raw.columns:
                        info_map[feat] = df_raw[feat].values

                sv_series = sv_ids if len(sv_ids) > 0 else (info_map.get("sv_id", [None] * n))

                raw_probs = np.array(list(probs)[:n], dtype=float).flatten()
                pred_confidence = np.where(raw_probs >= 0.5, raw_probs, 1.0 - raw_probs)

                df_result = pd.DataFrame({
                    "sv_id": list(sv_series)[:n],
                    "prediction": [int(x) for x in list(labels)[:n]],
                    "confidence": [float(x) for x in pred_confidence],
                    "positive_probability": [float(x) for x in raw_probs],
                })

                try:
                    if 'feature_names' not in locals():
                        feature_names = df_for_pred.columns.tolist() if 'df_for_pred' in locals() else None
                except Exception:
                    feature_names = None

                genes_list = None
                if "gene_name" in info_map:
                    genes_list = list(info_map["gene_name"])[:n]

                BIO_FEATURES = [
                    "log2_tpm", "median_tpm", "mean_tpm", "max_tpm",
                    "tau", "pli", "lof_oe", "mis_z",
                    "expressed_tissues", "af", "svlen",
                ]
                raw_df_for_explanations = None
                try:
                    available_bio = [f for f in BIO_FEATURES if f in df_raw.columns]
                    if available_bio:
                        raw_df_for_explanations = df_raw[available_bio].head(n)
                except Exception:
                    raw_df_for_explanations = None

                explanations = generate_better_explanations(
                    raw_df_for_explanations,
                    probs,
                    labels,
                    threshold,
                    feat_names=feature_names,
                    genes=genes_list,
                )
                df_result["explanation"] = explanations

                for k, arr in info_map.items():
                    df_result[k] = list(arr)[:n]

                if "label" in df_raw.columns:
                    df_result["true_label"] = df_raw["label"].values[:n]

                from pathlib import Path
                PREDICTIONS_DIR = Path(app.config.get("PREDICTIONS_DIR", str(Path(__file__).resolve().parent.parent / "predictions")))
                try:
                    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.error(f"Failed to create predictions dir {PREDICTIONS_DIR}: {e}")
                    pass
                predictions_file = PREDICTIONS_DIR / f"pred_{dataset.id}_{experiment_id}.tsv"
                logger.info(f"[PREDICTION] Saving {len(df_result)} predictions to {predictions_file}")
                try:
                    df_result.to_csv(predictions_file, sep="\t", index=False)
                    logger.info(f"[PREDICTION] Predictions file saved successfully")
                except Exception as e:
                    logger.error(f"[PREDICTION] Failed to save predictions file: {e}")
                    raise

                stats = {
                    "total": len(df_result),
                    "pred_1": int((labels == 1).sum()),
                    "pred_0": int((labels == 0).sum()),
                }

                try:
                    sv_stats_dict = create_sv_statistics(df_result)
                    stats["sv_statistics"] = sv_stats_dict.get("sv_statistics", [])
                except Exception as e:
                    stats["sv_statistics"] = []
                    print(f"Warning: Could not generate SV statistics: {e}")

                graph_data = {}
                if "true_label" in df_result.columns:
                    y_true = df_result["true_label"].astype(int).values
                    y_pred = df_result["prediction"].astype(int).values
                    y_score = probs

                    stats["accuracy"] = float((y_pred == y_true).mean())
                    stats["auc"] = float(_roc_auc(y_true, y_score))

                    cm = compute_confusion_matrix(y_true, y_pred)
                    stats.update(cm)
                    graph_data["confusion_matrix"] = cm

                    fpr, tpr, roc_auc = compute_roc_curve(y_true, y_score)
                    graph_data["roc_curve"] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}

                    recall, precision, pr_auc = compute_pr_curve(y_true, y_score)
                    graph_data["pr_curve"] = {"recall": recall, "precision": precision, "auc": pr_auc}

                    cal_probs, cal_fracs = compute_calibration_curve(y_true, y_score)
                    graph_data["calibration"] = {"probs": cal_probs, "fracs": cal_fracs}

                    graph_data["prediction_dist"] = compute_prediction_distribution(y_score, y_true)

                    graph_data["metrics_vs_threshold"] = compute_metrics_vs_threshold(y_true, y_score)

                    graph_data["confidence_vs_accuracy"] = compute_confidence_vs_accuracy(y_true, y_pred, y_score)

                    for graph_type, data in graph_data.items():
                        _add_graph_data(experiment_id, graph_type, data)

                exp.status = "completed"
                exp.message = "Prediction completed"
                exp.predictions_path = str(predictions_file)
                exp.result_data = stats
                db.session.commit()
                try:
                    try:
                        export_experiment(exp.id, formats=["csv", "json", "pdf"])
                    except Exception:
                        pass

                    task_service.update_task(tx_id, status="completed", progress=100, message="Prediction completed")
                except Exception:
                    pass

            except Exception as exc:
                exp = Experiment.query.get(experiment_id)
                if exp:
                    exp.status = "failed"
                    exp.message = str(exc)
                    db.session.commit()
                try:
                    task_service.update_task(tx_id, status="failed", progress=100, message=str(exc))
                except Exception:
                    pass

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

def run_prediction(*args, **kwargs):
    raise NotImplementedError("Move prediction worker logic from app.py into this function when ready.")
