import threading
from datetime import datetime
from pathlib import Path
import traceback

from backend.extensions import db
from backend.models import Dataset
from backend.services import task_service
from ml_module.preprocessing import load_and_clean, save_preprocessing_artifacts


def start_preprocessing(app, task_id, dataset_id, raw_path):
    def _worker():
        with app.app_context():
            try:
                task_service.update_task(task_id, status="running", progress=0, message="loading")
                result = load_and_clean(str(raw_path), fit=True)
                if isinstance(result, tuple):
                    try:
                        df, sv_gene_pairs, scaler, encoders = result
                    except Exception:
                        df = result if not isinstance(result, (list,)) else None
                        sv_gene_pairs = None
                        scaler = None
                        encoders = None
                else:
                    df = result
                    sv_gene_pairs = None
                    scaler = None
                    encoders = None

                pre_dir = Path(app.config.get("PREPROCESSED_DIR", "datasets/preprocessed"))
                pre_dir.mkdir(parents=True, exist_ok=True)
                pre_path = pre_dir / f"preprocessed_{dataset_id}.tsv"
                df.to_csv(pre_path, sep="\t", index=False)

                ds_obj = Dataset.query.get(dataset_id)
                ds_obj.preprocessed_path = str(pre_path)
                ds_obj.updated_at = datetime.utcnow()
                db.session.commit()
                task_service.update_task(task_id, status="running", progress=97, message="Preprocessed dataset saved")

                artifact_folder = Path(app.config.get("ARTIFACTS_DIR", "artifacts")) / f"dataset_{dataset_id}"
                artifact_folder.mkdir(parents=True, exist_ok=True)
                try:
                    save_preprocessing_artifacts(scaler, encoders, save_dir=str(artifact_folder))
                except Exception:
                    traceback.print_exc()

                ds_obj = Dataset.query.get(dataset_id)
                ds_obj.preprocessed_path = str(pre_path)
                ds_obj.artifacts_path = str(artifact_folder)
                ds_obj.has_label = "label" in df.columns
                ds_obj.row_count = len(df)
                ds_obj.updated_at = datetime.utcnow()
                db.session.commit()

                task_service.update_task(task_id, status="completed", progress=100, message="Dataset preprocessing completed")
            except Exception as exc:
                traceback.print_exc()
                ds = Dataset.query.get(dataset_id)
                if ds:
                    ds.updated_at = datetime.utcnow()
                    db.session.commit()
                task_service.update_task(task_id, status="failed", progress=100, message=str(exc))
            finally:
                task_service.TASK_DATASET_MAP.pop(task_id, None)
                task_service.CANCEL_FLAGS.pop(task_id, None)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

def preprocess_dataset(path: str, save_dir: str, progress_callback=None):
    raise NotImplementedError("Move preprocessing logic from app.py into this function when ready.")
