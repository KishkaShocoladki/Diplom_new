from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, List


def _explain_svtype(val) -> tuple[str, str]:

    sv_map = {
        "del": "делеция (DEL)", "dup": "дупликация (DUP)",
        "inv": "инверсия (INV)", "ins": "вставка (INS)",
        "bnd": "межхромосомная перестройка (BND)",
    }
    s = str(val).lower().strip()
    label = sv_map.get(s, f"тип СВ «{val}»")

    impact_map = {
        "del": ("high",   f"Тип варианта - {label}; делеции часто нарушают ORF и регуляторные элементы"),
        "dup": ("high",   f"Тип варианта - {label}; дупликации изменяют дозу гена"),
        "inv": ("medium", f"Тип варианта - {label}; инверсии могут нарушить регуляторные домены"),
        "ins": ("medium", f"Тип варианта - {label}; вставки могут прерывать экзоны"),
        "bnd": ("high",   f"Тип варианта - {label}; транслокации часто создают химерные транскрипты"),
    }
    wt, txt = impact_map.get(s, ("low", f"Тип варианта - {label}"))
    return txt, wt


def _explain_svlen(val) -> tuple[str, str]:
    try:
        bp = abs(float(val))
    except (TypeError, ValueError):
        return "Размер СВ неизвестен", "low"

    if bp >= 1_000_000:
        return f"СВ очень крупный ({bp/1e6:.1f} Мбп) - затрагивает крупные геномные домены", "high"
    if bp >= 100_000:
        return f"СВ крупный ({bp/1e3:.0f} кбп) - высокая вероятность захвата регуляторных элементов", "high"
    if bp >= 10_000:
        return f"СВ среднего размера ({bp/1e3:.1f} кбп)", "medium"
    if bp >= 1_000:
        return f"СВ небольшой ({bp:.0f} бп) - возможно затрагивает один экзон", "medium"
    return f"СВ очень маленький ({bp:.0f} бп) - минимальное влияние на ген", "low"


def _explain_pli(val) -> tuple[str, str]:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "Индекс PLI неизвестен", "low"

    if v >= 0.9:
        return f"Ген строго нетолерантен к потере функции (PLI={v:.2f}≥0.9) - любые LoF-варианты крайне опасны", "high"
    if v >= 0.5:
        return f"Ген умеренно нетолерантен к LoF (PLI={v:.2f})", "medium"
    return f"Ген относительно толерантен к потере функции (PLI={v:.2f}<0.5)", "low"


def _explain_lof_oe(val) -> tuple[str, str]:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "Соотношение LoF O/E неизвестно", "low"

    if v < 0.1:
        return f"Крайне низкое LOF O/E ({v:.3f}) - ген нетолерантен к LoF, мутации негативно отбираются", "high"
    if v < 0.35:
        return f"Низкое LOF O/E ({v:.2f}) - ген с умеренным LoF-ограничением", "medium"
    return f"Высокое LOF O/E ({v:.2f}) - ген толерантен к потере функции", "low"


def _explain_mis_z(val) -> tuple[str, str]:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "Миссенс Z-score неизвестен", "low"

    if v >= 3.09:
        return f"Ген сильно ограничен по миссенс-мутациям (Z={v:.2f}≥3.09) - высокая функциональная чувствительность", "high"
    if v >= 1.0:
        return f"Ген умеренно ограничен по миссенс-мутациям (Z={v:.2f})", "medium"
    if v < 0:
        return f"Ген толерантен к миссенс-мутациям (Z={v:.2f}<0) - мутации встречаются чаще ожидаемого", "low"
    return f"Нейтральный миссенс-constraint (Z={v:.2f})", "low"


def _explain_log2_tpm(val) -> tuple[str, str]:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "Уровень экспрессии неизвестен", "low"

    if v >= 4.0:   # ~16 TPM
        return f"Ген сильно экспрессирован (log2_TPM={v:.1f}) - функциональная активность высока", "high"
    if v >= 1.0:   # ~2 TPM
        return f"Ген умеренно экспрессирован (log2_TPM={v:.1f})", "medium"
    if v > 0:
        return f"Ген слабо экспрессирован (log2_TPM={v:.1f}) - может быть тканеспецифичным", "low"
    return f"Ген практически не экспрессирован (log2_TPM={v:.1f}) - маловероятно функциональное влияние", "low"


def _explain_expressed_tissues(val) -> tuple[str, str]:
    try:
        v = int(float(val))
    except (TypeError, ValueError):
        return "Число экспрессирующих тканей неизвестно", "low"

    if v >= 30:
        return f"Ген экспрессируется в {v} тканях - повсеместный, влияние варианта широкое", "high"
    if v >= 10:
        return f"Ген экспрессируется в {v} тканях", "medium"
    if v >= 3:
        return f"Ген экспрессируется в {v} тканях - умеренно тканеспецифичен", "low"
    return f"Ген экспрессируется лишь в {v} тканях - высокая тканевая специфичность", "low"


def _explain_tau(val) -> tuple[str, str]:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "Индекс тканевой специфичности (tau) неизвестен", "low"

    if v >= 0.85:
        return f"Ген строго тканеспецифичен (tau={v:.2f}) - влияние ограничено одной-двумя тканями", "medium"
    if v >= 0.5:
        return f"Ген умеренно тканеспецифичен (tau={v:.2f})", "medium"
    return f"Ген экспрессируется широко (tau={v:.2f}) - потенциально системное влияние", "low"


def _explain_af(val) -> tuple[str, str]:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "Частота аллеля неизвестна", "low"

    if v < 1e-4:
        return f"Редкий вариант (AF={v:.2e}) - вероятно патогенный или de novo", "high"
    if v < 0.01:
        return f"Вариант низкой частоты (AF={v:.4f}) - возможна клиническая значимость", "medium"
    return f"Частый полиморфизм (AF={v:.3f}) - скорее всего доброкачественный", "low"


def _explain_gene_type(val) -> tuple[str, str]:
    s = str(val).lower()
    if "protein_coding" in s:
        return "Ген кодирует белок - наиболее значимая категория для потери функции", "high"
    if "lncrna" in s or "lincrna" in s:
        return "lncRNA - может регулировать экспрессию соседних генов", "medium"
    if "mirna" in s:
        return "miRNA - регуляторная РНК, влияние затрагивает несколько генов-мишеней", "medium"
    return f"Тип гена: {val}", "low"


_FEATURE_EXPLAINERS = {
    "svtype":             _explain_svtype,
    "svlen":              _explain_svlen,
    "pli":                _explain_pli,
    "lof_oe":             _explain_lof_oe,
    "mis_z":              _explain_mis_z,
    "log2_tpm":           _explain_log2_tpm,
    "expressed_tissues":  _explain_expressed_tissues,
    "tau":                _explain_tau,
    "af":                 _explain_af,
    "gene_type":          _explain_gene_type,
}

_WEIGHT_ORDER = {"high": 0, "medium": 1, "low": 2}


def _build_explanation(row: dict, pred: int, prob: float, gene: Optional[str]) -> str:
    facts: list[tuple[str, str]] = []

    for feat, explainer in _FEATURE_EXPLAINERS.items():
        if feat in row and row[feat] is not None and not _is_nan(row[feat]):
            txt, wt = explainer(row[feat])
            facts.append((txt, wt))

    facts.sort(key=lambda x: _WEIGHT_ORDER.get(x[1], 3))
    top = facts[:4]

    gene_str = f"{gene}" if gene else ""
    if pred == 1:
        verdict = f"СВ ВЛИЯЕТ на ген {gene_str}"
        intro   = "Основные факторы, указывающие на влияние:"
    else:
        verdict = f"СВ НЕ ВЛИЯЕТ на ген {gene_str}"
        intro   = "Основные факторы, указывающие на отсутствие влияния:"

    pct = prob * 100
    if pct >= 85:
        conf_phrase = f"высокая уверенность модели ({pct:.0f}%)"
    elif pct >= 65:
        conf_phrase = f"уверенность модели {pct:.0f}%"
    elif pct >= 50:
        conf_phrase = f"слабая уверенность модели ({pct:.0f}%)"
    else:
        neutral_pct = (1 - prob) * 100
        conf_phrase = f"уверенность в нейтральности {neutral_pct:.0f}%"

    if not top:
        return f"{verdict} [{conf_phrase}] - биологические данные недоступны."

    reasons = "; ".join(t for t, _ in top)
    return f"{verdict} [{conf_phrase}]. {intro} {reasons}."


def _is_nan(v) -> bool:
    try:
        return np.isnan(float(v))
    except (TypeError, ValueError):
        return False

def generate_better_explanations(
    raw_df: Optional[pd.DataFrame],
    probs_arr,
    predictions,
    threshold_val: float,
    feat_names=None,
    genes: Optional[List[str]] = None,
) -> List[str]:

    if raw_df is not None and not isinstance(raw_df, pd.DataFrame):
        try:
            raw_df = pd.DataFrame(raw_df, columns=feat_names) if feat_names is not None else pd.DataFrame(raw_df)
        except Exception:
            raw_df = pd.DataFrame(raw_df)

    n = len(probs_arr)
    explanations = []

    for i in range(n):
        prob = float(probs_arr[i])
        pred = int(predictions[i]) if i < len(predictions) else 0
        gene = genes[i] if genes and i < len(genes) else None

        row: dict = {}
        if raw_df is not None and not raw_df.empty and i < len(raw_df):
            try:
                row = raw_df.iloc[i].to_dict()
            except Exception:
                row = {}

        try:
            explanation = _build_explanation(row, pred, prob, gene)
        except Exception:
            pct = prob * 100
            label_str = "влияет" if pred == 1 else "не влияет"
            explanation = f"СВ {label_str} (уверенность {pct:.0f}%)"

        explanations.append(explanation)

    return explanations


def create_sv_statistics(df_predictions: pd.DataFrame) -> dict:
    required_cols = ["sv_id", "prediction", "confidence"]
    for col in required_cols:
        if col not in df_predictions.columns:
            return {"sv_statistics": []}

    sv_groups = df_predictions.groupby("sv_id")
    statistics = []

    for sv_id, group in sv_groups:
        row_dict = group.iloc[0].to_dict()
        sv_type = row_dict.get("sv_type") or row_dict.get("svtype", "UNKNOWN")

        group = group.copy()
        group["abs_confidence"] = np.abs(group["confidence"] - 0.5) * 2
        group = group.sort_values("abs_confidence", ascending=False)

        affected_genes: list = []
        uncertain_genes: list = []
        neutral_genes: list = []
        gene_details: list = []

        for _, row in group.iterrows():
            gene_name = row.get("gene_name") or row.get("gene_id", "UNKNOWN")
            pred = int(row["prediction"])
            conf = float(row["confidence"])

            if conf >= 0.75:
                conf_level = "very_high"
            elif conf >= 0.60:
                conf_level = "high"
            elif conf >= 0.40:
                conf_level = "medium"
            else:
                conf_level = "low"

            if pred == 1:
                if conf >= 0.7:
                    category = "affected"
                    affected_genes.append(gene_name)
                else:
                    category = "uncertain_affected"
                    uncertain_genes.append(gene_name)
            else:
                if conf >= 0.7:
                    category = "neutral"
                    neutral_genes.append(gene_name)
                else:
                    category = "uncertain_neutral"
                    uncertain_genes.append(gene_name)

            gene_details.append({
                "gene_name": str(gene_name),
                "prediction": pred,
                "confidence": float(conf),
                "confidence_pct": float(conf) * 100,
                "confidence_level": conf_level,
                "category": category,
            })

        statistics.append({
            "sv_id": str(sv_id),
            "sv_type": str(sv_type),
            "num_genes": len(group),
            "num_affected": len(affected_genes),
            "num_uncertain": len(uncertain_genes),
            "num_neutral": len(neutral_genes),
            "gene_details": gene_details,
        })

    return {"sv_statistics": statistics}


def create_sv_summary_text(sv_stat: dict) -> str:
    parts = [
        f"Structural variant: {sv_stat['sv_id']}",
        f"Type: {sv_stat['sv_type']}",
        f"Genes tested: {sv_stat['num_genes']}",
    ]
    summary: list[str] = []
    if sv_stat["num_affected"] > 0:
        summary.append(f"{sv_stat['num_affected']} gene(s) affected with high confidence")
    if sv_stat["num_uncertain"] > 0:
        summary.append(f"{sv_stat['num_uncertain']} gene(s) with uncertain effect")
    if sv_stat["num_neutral"] > 0:
        summary.append(f"{sv_stat['num_neutral']} gene(s) unaffected")
    if summary:
        parts.append("Summary: " + ", ".join(summary))
    return " | ".join(parts)
