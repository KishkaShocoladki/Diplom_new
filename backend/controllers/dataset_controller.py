from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from pathlib import Path
import shutil

from backend.extensions import db
from backend.models import Dataset, Experiment, Metric, GraphData
from backend.services import task_service
from backend.ml_module.preprocessing import load_preprocessing_artifacts, NUMERIC_COLS

bp = Blueprint("datasets", __name__, url_prefix="/api/datasets")

SV_COLUMNS = [
    "sv_id", "svtype", "svlen", "af",
    "gene_id", "gene_name", "gene_type",
    "overlap_type", "overlap_frac", "gene_ov_frac", "distance",
    "median_tpm", "mean_tpm", "max_tpm", "log2_tpm", "expressed_tissues",
    "tau", "pli", "lof_oe", "mis_z", "label",
]

SORTABLE_COLUMNS = set(SV_COLUMNS)

FILTER_NUMERIC_COLS = {
    "svlen", "af", "overlap_frac", "gene_ov_frac", "distance",
    "median_tpm", "mean_tpm", "max_tpm", "log2_tpm", "expressed_tissues",
    "tau", "pli", "lof_oe", "mis_z",
}

FILTER_CATEGORY_COLS = {"svtype", "gene_type", "overlap_type", "label"}


def _compute_hash(file_path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_file_type(filename: str) -> str:
    ext = Path(filename).suffix.lower().lstrip('.')
    if ext == 'csv':
        return 'csv'
    if ext == 'tsv':
        return 'tsv'
    raise ValueError('Unsupported file type. Supported: CSV, TSV.')


def _validate_uploaded_dataset(file_path: Path, filename: str):
    import pandas as pd

    errors = []
    try:
        file_type = _detect_file_type(filename)
    except ValueError as exc:
        raise ValueError(str(exc))

    sep = '\t' if file_type == 'tsv' else ','
    try:
        df = pd.read_csv(file_path, sep=sep, nrows=20)
        required = {'sv_id', 'label'}
        if not required.issubset(set(df.columns)):
            missing = required - set(df.columns)
            errors.append('CSV/TSV missing required columns: ' + ', '.join(sorted(missing)))
    except Exception as exc:
        errors.append(f'CSV/TSV parse error: {exc}')

    if errors:
        raise ValueError('; '.join(errors))

    return True


def _load_df_raw_for_stats(ds: Dataset):
    import pandas as pd

    if ds.raw_path.endswith(".parquet"):
        df = pd.read_parquet(ds.raw_path)
    else:
        df = pd.read_csv(ds.raw_path, sep="\t")

    present = [c for c in SV_COLUMNS if c in df.columns]
    return df[present]


def _load_df(ds: Dataset):
    import pandas as pd

    if ds.preprocessed_path.endswith(".parquet"):
        df = pd.read_parquet(ds.preprocessed_path)
    else:
        df = pd.read_csv(ds.preprocessed_path, sep="\t")

    present = [c for c in SV_COLUMNS if c in df.columns]
    return df[present]


def _load_dataset_scaler(ds: Dataset):
    if not ds.artifacts_path:
        return None
    try:
        scaler, _ = load_preprocessing_artifacts(ds.artifacts_path)
        return scaler
    except Exception:
        return None


def _denormalize_numeric_columns(df, scaler):
    if scaler is None or df is None:
        return df

    cols = [c for c in NUMERIC_COLS if c in df.columns]
    if not cols:
        return df

    try:
        numeric_data = df[cols].astype(float).to_numpy()
        df[cols] = scaler.inverse_transform(numeric_data)
    except Exception:
        pass

    return df


def _compute_sv_stats(df) -> dict:
    import numpy as np

    total = len(df)
    stats: dict = {"total_sv": total}

    def safe_int(x):
        try:
            return int(x)
        except Exception:
            return 0

    def safe_float(x):
        try:
            v = float(x)
            return None if (v != v) else round(v, 6)   # NaN → None
        except Exception:
            return None

    has_label = "label" in df.columns
    stats["has_label"] = has_label
    if has_label:
        lbl = df["label"].dropna()
        pos = safe_int((lbl == 1).sum())
        neg = safe_int((lbl == 0).sum())
        stats["label_positive"]     = pos
        stats["label_negative"]     = neg
        stats["label_positive_pct"] = round(pos / total * 100, 2) if total else 0

    if "svtype" in df.columns:
        stats["sv_per_type"] = {
            k: safe_int(v)
            for k, v in df["svtype"].value_counts().items()
        }

    if "svlen" in df.columns:
        svlen = df["svlen"].dropna()
        stats["size_categories"] = {
            "Крошечные (<100 bp)":      safe_int((svlen < 100).sum()),
            "Малые (100 bp–1 кб)":      safe_int(((svlen >= 100)   & (svlen < 1_000)).sum()),
            "Средние (1–10 кб)":        safe_int(((svlen >= 1_000)  & (svlen < 10_000)).sum()),
            "Крупные (10–100 кб)":      safe_int(((svlen >= 10_000) & (svlen < 100_000)).sum()),
            "Очень крупные (>100 кб)":  safe_int((svlen >= 100_000).sum()),
        }

        if "svtype" in df.columns:
            size_by_type = {}
            for svtype, grp in df.groupby("svtype"):
                s = grp["svlen"].dropna()
                if len(s):
                    size_by_type[svtype] = {
                        "median": safe_float(s.median()),
                        "mean":   safe_float(s.mean()),
                        "min":    safe_float(s.min()),
                        "max":    safe_float(s.max()),
                        "p25":    safe_float(s.quantile(0.25)),
                        "p75":    safe_float(s.quantile(0.75)),
                    }
            stats["size_by_type"] = size_by_type

    if "af" in df.columns:
        af = df["af"].dropna()
        stats["af_categories"] = {
            "Редкие (AF < 1%)":        safe_int((af < 0.01).sum()),
            "Нечастые (AF 1–5%)":      safe_int(((af >= 0.01) & (af < 0.05)).sum()),
            "Частые (AF > 5%)":        safe_int((af >= 0.05).sum()),
        }
        stats["af_median"] = safe_float(af.median()) if len(af) else None

    if "overlap_type" in df.columns:
        stats["overlap_types"] = {
            k: safe_int(v)
            for k, v in df["overlap_type"].value_counts().items()
        }
    if "gene_type" in df.columns:
        stats["gene_types"] = {
            k: safe_int(v)
            for k, v in df["gene_type"].value_counts().items()
        }

    if has_label and "median_tpm" in df.columns:
        expr = {}
        for lbl_val, key in [(1, "positive"), (0, "neutral")]:
            grp = df[df["label"] == lbl_val]["median_tpm"].dropna()
            if len(grp):
                expr[key] = {
                    "median_tpm": safe_float(grp.median()),
                    "mean_tpm":   safe_float(grp.mean()),
                }
        if expr:
            stats["expression_by_label"] = expr

    if has_label and "expressed_tissues" in df.columns:
        tis = {}
        for lbl_val, key in [(1, "positive"), (0, "neutral")]:
            grp = df[df["label"] == lbl_val]["expressed_tissues"].dropna()
            if len(grp):
                tis[key] = {
                    "median": safe_float(grp.median()),
                    "mean":   safe_float(grp.mean()),
                }
        if tis:
            stats["tissues_by_label"] = tis

    if "pli" in df.columns:
        pli = df["pli"].dropna()
        stats["pli_categories"] = {
            "Высокое (pLI > 0.9)":   safe_int((pli > 0.9).sum()),
            "Среднее (0.5–0.9)":      safe_int(((pli >= 0.5) & (pli <= 0.9)).sum()),
            "Низкое (< 0.5)":        safe_int((pli < 0.5).sum()),
        }

    if "lof_oe" in df.columns:
        loe = df["lof_oe"].dropna()
        stats["lof_oe_categories"] = {
            "Под отбором (≤ 0.35)":       safe_int((loe <= 0.35).sum()),
            "Промежуточный (0.35–0.7)":   safe_int(((loe > 0.35) & (loe <= 0.7)).sum()),
            "Толерантный (> 0.7)":        safe_int((loe > 0.7).sum()),
        }

    if "mis_z" in df.columns:
        mz = df["mis_z"].dropna()
        stats["mis_z_categories"] = {
            "Под давлением (mis_z > 3)":   safe_int((mz > 3).sum()),
            "Умеренный (1–3)":             safe_int(((mz >= 1) & (mz <= 3)).sum()),
            "Нейтральный (< 1)":           safe_int((mz < 1).sum()),
        }

    if "tau" in df.columns:
        tau = df["tau"].dropna()
        stats["tau_categories"] = {
            "Убиквитарные (tau < 0.3)":       safe_int((tau < 0.3).sum()),
            "Умеренные (0.3–0.7)":            safe_int(((tau >= 0.3) & (tau <= 0.7)).sum()),
            "Тканеспецифичные (tau > 0.7)":   safe_int((tau > 0.7).sum()),
        }

    if "gene_name" in df.columns:
        top = df["gene_name"].value_counts().head(10)
        stats["top_genes"] = [
            {"gene": str(g), "count": safe_int(c)} for g, c in top.items()
        ]

    if "chrom" in df.columns:
        stats["available_chroms"] = sorted(
            str(c) for c in df["chrom"].dropna().unique()
        )
    else:
        try:
            chroms = (
                df["sv_id"]
                .str.extract(r"_(chr\w+)_")[0]
                .dropna()
                .value_counts()
                .head(30)
                .index.tolist()
            )
            stats["available_chroms"] = sorted(set(chroms))
        except Exception:
            stats["available_chroms"] = []

    return stats


def _load_dataset_encoders(ds: Dataset):
    if not ds.artifacts_path:
        return {}
    try:
        _, encoders = load_preprocessing_artifacts(ds.artifacts_path)
        return encoders or {}
    except Exception:
        return {}


def _normalize_category_values(col, values, df, encoders):
    if not values:
        return values
    if col not in FILTER_CATEGORY_COLS or not encoders:
        return values

    encoder = encoders.get(col)
    if encoder is None:
        return values

    normalized = []
    for raw in values:
        if raw == "":
            continue
        try:
            normalized.append(int(raw))
            continue
        except Exception:
            pass

        if raw in encoder.classes_:
            try:
                normalized.append(int(encoder.transform([raw])[0]))
                continue
            except Exception:
                pass

        normalized.append(raw)

    return normalized


def _apply_filters(df, args, ds=None):
    import pandas as pd

    encoders = _load_dataset_encoders(ds) if ds is not None else {}
    filtered = df.copy()

    for col in FILTER_CATEGORY_COLS:
        raw = args.get(f"filter_{col}", "").strip()
        if not raw or col not in filtered.columns:
            continue
        vals_str = [v.strip() for v in raw.split(",") if v.strip()]
        if not vals_str:
            continue
        vals = _normalize_category_values(col, vals_str, filtered, encoders)
        filtered = filtered[filtered[col].isin(vals)]

    for col in FILTER_NUMERIC_COLS:
        if col not in filtered.columns:
            continue
        mn = args.get(f"filter_{col}_min", "").strip()
        mx = args.get(f"filter_{col}_max", "").strip()
        try:
            if mn:
                filtered = filtered[filtered[col] >= float(mn)]
        except (ValueError, TypeError):
            pass
        try:
            if mx:
                filtered = filtered[filtered[col] <= float(mx)]
        except (ValueError, TypeError):
            pass

    search_q = args.get("search", "").strip()
    if search_q:
        import pandas as pd
        mask = pd.Series(False, index=filtered.index)
        for tcol in ("sv_id", "gene_id", "gene_name"):
            if tcol in filtered.columns:
                mask |= filtered[tcol].astype(str).str.contains(
                    search_q, case=False, na=False, regex=False
                )
        filtered = filtered[mask]

    return filtered


@bp.route("/", methods=["GET"])
def list_datasets_controller():
    datasets = Dataset.query.order_by(Dataset.created_at.desc()).all()
    return jsonify([d.to_dict() for d in datasets])


@bp.route("/<int:dataset_id>", methods=["GET"])
def get_dataset_controller(dataset_id: int):
    ds = Dataset.query.get_or_404(dataset_id)
    return jsonify(ds.to_dict())


@bp.route("/<int:dataset_id>", methods=["DELETE"])
def delete_dataset_controller(dataset_id: int):
    ds = Dataset.query.get_or_404(dataset_id)
    try:    
        experiments = Experiment.query.filter_by(dataset_id=ds.id).all()
        for e in experiments:
            Metric.query.filter_by(experiment_id=e.id).delete()
            GraphData.query.filter_by(experiment_id=e.id).delete()
            if e.predictions_path and Path(e.predictions_path).exists():
                try:
                    Path(e.predictions_path).unlink()
                except Exception:
                    pass
            db.session.delete(e)

        for attr in ("raw_path", "preprocessed_path"):
            p = getattr(ds, attr, None)
            if p and Path(p).exists():
                Path(p).unlink()
        if ds.artifacts_path and Path(ds.artifacts_path).exists():
            shutil.rmtree(ds.artifacts_path)
    except Exception:
        pass
    db.session.delete(ds)
    db.session.commit()
    return jsonify({"status": "deleted"})


@bp.route("/upload", methods=["POST"])
def upload_dataset_controller():
    from werkzeug.exceptions import BadRequest

    if "file" not in request.files:
        raise BadRequest("No file provided")
    file = request.files["file"]
    filename = secure_filename(file.filename)
    raw_dir = Path(current_app.config.get("RAW_DATA_DIR", "D:/datasets/raw"))
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / filename
    file.save(raw_path)

    file_hash = _compute_hash(raw_path)

    existing = Dataset.query.filter_by(file_hash=file_hash).first()
    if existing:
        raw_path.unlink(missing_ok=True)
        return jsonify({"error": "duplicate", "dataset": existing.to_dict()}), 400

    try:
        _validate_uploaded_dataset(raw_path, filename)
    except ValueError as exc:
        ds = Dataset(
            name=filename,
            raw_path=str(raw_path),
            file_hash=file_hash,
        )
        db.session.add(ds)
        db.session.commit()
        return jsonify({
            "error": "validation_failed",
            "validation_errors": str(exc),
            "dataset_id": ds.id,
        }), 400

    ds = Dataset(
        name=filename,
        raw_path=str(raw_path),
        file_hash=file_hash,
    )
    db.session.add(ds)
    db.session.commit()

    task_id = task_service.create_task("preprocessing started")
    task_service.TASK_DATASET_MAP[task_id] = ds.id

    from backend.services import preprocessing_service
    preprocessing_service.start_preprocessing(
        current_app._get_current_object(), task_id, ds.id, raw_path
    )

    return jsonify({"task_id": task_id, "dataset_id": ds.id}), 201


@bp.route("/<int:dataset_id>/sv-info", methods=["GET"])
def dataset_sv_info_controller(dataset_id: int):
    ds = Dataset.query.get_or_404(dataset_id)

    if not ds.preprocessed_path or not Path(ds.preprocessed_path).exists():
        return jsonify({"error": "not preprocessed"}), 404

    try:
        df_raw = _load_df_raw_for_stats(ds)
        df = _load_df(ds)
        scaler = _load_dataset_scaler(ds)
        df = _denormalize_numeric_columns(df, scaler)
    except Exception as exc:
        return jsonify({"error": f"Failed to load dataset: {exc}"}), 500

    sv_info = _compute_sv_stats(df_raw)

    sv_info["last_prediction"] = None
    try:
        from backend.models import Prediction
        pred = (
            Prediction.query
            .filter_by(dataset_id=dataset_id)
            .order_by(Prediction.predicted_at.desc())
            .first()
        )
        if pred:
            sv_info["last_prediction"] = {
                "experiment_id": pred.experiment_id,
                "predicted_at": pred.predicted_at.isoformat(),
                "stats": pred.stats_json,
            }
    except Exception:
        pass
    
    filtered = _apply_filters(df, request.args, ds=ds)

    sort_by  = request.args.get("sort_by",  "").strip()
    sort_dir = request.args.get("sort_dir", "asc").strip().lower()
    if sort_by in SORTABLE_COLUMNS and sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=(sort_dir != "desc"))

    total_filtered = len(filtered)
    try:
        page     = max(1, int(request.args.get("page",     1)))
        per_page = min(200, max(1, int(request.args.get("per_page", 50))))
    except ValueError:
        page, per_page = 1, 50

    start    = (page - 1) * per_page
    page_df  = filtered.iloc[start: start + per_page]
    rows     = page_df.where(page_df.notna(), other=None).to_dict(orient="records")

    return jsonify({
        "sv_info": sv_info,
        "rows": rows,
        "pagination": {
            "page":        page,
            "per_page":    per_page,
            "total":       total_filtered,
            "total_pages": max(1, (total_filtered + per_page - 1) // per_page),
        },
        "columns": list(page_df.columns),
    })


def register_blueprint(app):
    app.register_blueprint(bp)