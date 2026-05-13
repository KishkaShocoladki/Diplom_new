import os
import warnings
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

ID_COLS: list[str] = ["sv_id", "gene_id", "gene_name"]
LEAKAGE_COLS: list[str] = [
    "overlap_frac",  
    "gene_ov_frac",  
    "overlap_type",   
    "distance",       
]
CATEGORICAL_COLS: list[str] = ["svtype", "gene_type"]
NUMERIC_COLS: list[str] = [
    "svlen",              # variant size
    "af",                 # allele frequency in population
    "pli",                # LoF-intolerance probability
    "lof_oe",             # observed/expected LoF ratio
    "mis_z",              # Z-score of missense mutations
    "expressed_tissues",  # number of tissues with TPM > 0.5
    "log2_tpm",           # log2(median_tpm + 1)
    "median_tpm",         # median TPM by tissues
    "mean_tpm",           # mean TPM by tissues
    "max_tpm",            # max TPM by tissues
    "tau",                # tissue specificity index
]

TARGET_COL: str = "label"

def load_and_clean(
    path: str,
    scaler: Optional[StandardScaler] = None,
    encoders: Optional[dict] = None,
    fit: bool = True,
    chunksize: int = 2_000_000,
    progress_callback=None,
) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler, dict]:
    import os 
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    drop_cols = [c for c in ID_COLS + LEAKAGE_COLS]

    approx_nchunks = 1
    if fit:
        if progress_callback:
            progress_callback(0, "Сбор статистики (этап 1 из 2)")
        print(f"[load_and_clean] Проход 1/2: сбор статистик (chunksize={chunksize:,})...")
        numeric_accum  = {col: [] for col in NUMERIC_COLS}
        cat_values     = {col: set() for col in CATEGORICAL_COLS}
        total_rows = 0

        chunk_index = 0
        for chunk in pd.read_csv(path, sep="\t", low_memory=False, chunksize=chunksize):
            chunk_index += 1
            if TARGET_COL in chunk.columns:
                chunk = chunk[chunk[TARGET_COL] != -1]
            chunk.drop(columns=[c for c in drop_cols if c in chunk.columns],
                       inplace=True, errors="ignore")
            chunk.replace("", np.nan, inplace=True)

            for col in NUMERIC_COLS:
                if col in chunk.columns:
                    vals = pd.to_numeric(chunk[col], errors="coerce").dropna()
                    numeric_accum[col].extend(vals.tolist())

            for col in CATEGORICAL_COLS:
                if col in chunk.columns:
                    cat_values[col].update(chunk[col].fillna("unknown").astype(str).unique())

            total_rows += len(chunk)
            if progress_callback:
                progress_callback(min(20, 5 + int(15 * chunk_index / max(1, math.ceil(total_rows / chunksize)))), f"Сбор статистики: {chunk_index} чанков")

        approx_nchunks = max(1, math.ceil(total_rows / chunksize))

        col_medians = {}
        for col in NUMERIC_COLS:
            vals = numeric_accum[col]
            col_medians[col] = float(np.median(vals)) if vals else 0.0
        del numeric_accum

        if encoders is None:
            encoders = {}
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            classes = sorted(cat_values[col] | {"unknown"})
            le.fit(classes)
            encoders[col] = le

        if progress_callback:
            progress_callback(25, "Статистики собраны")
        print(f"[load_and_clean] Статистики собраны. Строк (без -1): {total_rows:,}")
    else:
        col_medians = None
        if encoders is None:
            raise ValueError("encoders не переданы при fit=False")
        if scaler is None:
            raise ValueError("scaler не передан при fit=False")

    print(f"[load_and_clean] Проход {'2/2' if fit else '1/1'}: трансформация...")
    chunks_out   = []
    sv_ids_list  = []
    pairs_list   = []

    if fit:
        pass
    else:
        file_size = os.path.getsize(path) if os.path.exists(path) else 0
        approx_nchunks = max(1, math.ceil(file_size / (100 * 1024 * 1024)))

    chunk_index = 0
    for chunk in pd.read_csv(path, sep="\t", low_memory=False, chunksize=chunksize):
        chunk_index += 1
        if progress_callback:
            pct = min(90, 30 + int(60 * chunk_index / max(1, approx_nchunks)))
            progress_callback(pct, f"Трансформация: чанк {chunk_index} / {approx_nchunks}")

        if TARGET_COL in chunk.columns:
            chunk = chunk[chunk[TARGET_COL] != -1].copy()
        else:
            chunk = chunk.copy()

        if "sv_id" in chunk.columns:
            sv_ids_list.append(chunk["sv_id"].values)
        if "sv_id" in chunk.columns and "gene_id" in chunk.columns:
            pairs_list.append(
                (chunk["sv_id"].astype(str) + "__" + chunk["gene_id"].astype(str)).values
            )

        chunk.drop(columns=[c for c in drop_cols if c in chunk.columns],
                   inplace=True, errors="ignore")

        chunk.replace("", np.nan, inplace=True)

        present_numeric = [c for c in NUMERIC_COLS if c in chunk.columns]
        for col in present_numeric:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
            if fit:
                chunk[col] = chunk[col].fillna(col_medians.get(col, 0.0))
            else:
                chunk[col] = chunk[col].fillna(chunk[col].median()
                                               if not chunk[col].isna().all() else 0.0)

        present_cat = [c for c in CATEGORICAL_COLS if c in chunk.columns]
        for col in present_cat:
            le = encoders[col]
            chunk[col] = chunk[col].fillna("unknown").astype(str)
            known = set(le.classes_)
            chunk[col] = chunk[col].apply(lambda x: x if x in known else "unknown")
            chunk[col] = le.transform(chunk[col])

        if TARGET_COL in chunk.columns:
            chunk[TARGET_COL] = pd.to_numeric(chunk[TARGET_COL], errors="coerce")
            chunk.dropna(subset=[TARGET_COL], inplace=True)
            chunk[TARGET_COL] = chunk[TARGET_COL].astype(int)

        chunks_out.append(chunk)

    print(f"[load_and_clean] Объединяем чанки...")
    df = pd.concat(chunks_out, ignore_index=True)
    del chunks_out

    sv_gene_pairs = np.concatenate(pairs_list) if pairs_list else np.array([])
    sv_ids        = np.concatenate(sv_ids_list) if sv_ids_list else np.array([])

    dropped_leakage = [c for c in LEAKAGE_COLS if c in df.columns]
    if dropped_leakage:
        df.drop(columns=dropped_leakage, inplace=True)
        print(f"[load_and_clean] Удалены leakage-признаки: {dropped_leakage}")

    scale_cols = [c for c in NUMERIC_COLS if c in df.columns]
    if fit:
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
    else:
        df[scale_cols] = scaler.transform(df[scale_cols])

    print(f"[load_and_clean] Готово. Строк: {len(df):,} | Признаков: {len(scale_cols) + len([c for c in CATEGORICAL_COLS if c in df.columns])}")
    if progress_callback:
        progress_callback(95, "Предобработка данных готова")
    return df, sv_gene_pairs, scaler, encoders


def prepare_for_training(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    progress_callback=None,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:

    if TARGET_COL not in df.columns:
        raise ValueError(f"Колонка '{TARGET_COL}' отсутствует в датасете.")

    before_filter = len(df)
    df = df[df[TARGET_COL] != -1].copy()
    filtered = before_filter - len(df)
    if filtered > 0:
        msg = f"[prepare_for_training] Отфильтровано label=-1: {filtered:,} строк"
        print(msg)
        if progress_callback:
            try:
                progress_callback(None, msg)
            except Exception:
                pass

    X_all = df.drop(columns=[TARGET_COL])
    y_all = df[TARGET_COL]

    counts = y_all.value_counts()
    min_count = counts.min()
    msg = f"[prepare_for_training] Распределение классов до балансировки: {counts.to_dict()}"
    print(msg)
    if progress_callback:
        try:
            progress_callback(None, msg)
        except Exception:
            pass

    balanced_parts = []
    for label_val in [0, 1]:
        if label_val not in counts.index:
            raise ValueError(f"Класс {label_val} отсутствует в датасете.")
        subset = df[y_all == label_val]
        sampled = subset.sample(n=min_count, random_state=random_state)
        balanced_parts.append(sampled)

    df_balanced = (
        pd.concat(balanced_parts)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    X = df_balanced.drop(columns=[TARGET_COL]).values.astype(np.float32)
    y = df_balanced[TARGET_COL].values.astype(np.float32)

    msg = (f"[prepare_for_training] После балансировки: {len(df_balanced)} строк "
           f"({min_count} каждого класса)")
    print(msg)
    if progress_callback:
        try:
            progress_callback(None, msg)
        except Exception:
            pass

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val, random_state=random_state, stratify=y_temp
    )

    msg = (f"[prepare_for_training] Train: {len(X_train)} | "
           f"Val: {len(X_val)} | Test: {len(X_test)}")
    print(msg)
    if progress_callback:
        try:
            progress_callback(None, msg)
        except Exception:
            pass

    return X_train, X_val, X_test, y_train, y_val, y_test

def save_preprocessing_artifacts(
    scaler: StandardScaler,
    encoders: dict,
    save_dir: str = "artifacts",
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(save_dir, "encoders.pkl"))
    print(f"[save_preprocessing_artifacts] Артефакты сохранены в '{save_dir}/'")


def load_preprocessing_artifacts(
    save_dir: str = "artifacts",
) -> Tuple[StandardScaler, dict]:
    scaler = joblib.load(os.path.join(save_dir, "scaler.pkl"))
    encoders = joblib.load(os.path.join(save_dir, "encoders.pkl"))
    print(f"[load_preprocessing_artifacts] Артефакты загружены из '{save_dir}/'")
    return scaler, encoders

def load_and_clean_for_prediction(
    path: str,
    scaler: StandardScaler,
    encoders: dict,
    progress_callback=None,
) -> Tuple[np.ndarray, np.ndarray]:
    df, sv_ids, _, _ = load_and_clean(
        path=path,
        scaler=scaler,
        encoders=encoders,
        fit=False,
        progress_callback=progress_callback,
    )
    
    df.drop(columns=[TARGET_COL], inplace=True, errors="ignore")

    X = df.values.astype(np.float32)
    feature_names = list(df.columns)
    return X, sv_ids, feature_names
