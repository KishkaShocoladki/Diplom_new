import React, { useEffect, useState, useCallback, useRef } from 'react';
import api from '../api';
import { useTaskContext } from '../TaskContext';

const SV_COLORS = {
  DEL: '#e74c3c',
  DUP: '#3498db',
  INV: '#f39c12',
  INS: '#2ecc71',
  CNV: '#9b59b6',
  BND: '#1abc9c',
  CPX: '#e67e22',
};

const SVTYPE_DECODE = {
  0: 'DEL', 1: 'DUP', 2: 'INS', 3: 'INV', 4: 'unknown',
};

const GENE_TYPE_DECODE = {
  0:  'IG_D_gene',
  1:  'IG_V_gene',
  2:  'IG_V_pseudogene',
  3:  'TEC',
  4:  'TR_D_gene',
  5:  'TR_J_gene',
  6:  'TR_V_gene',
  7:  'TR_V_pseudogene',
  8:  'lncRNA',
  9:  'miRNA',
  10: 'misc_RNA',
  11: 'processed_pseudogene',
  12: 'protein_coding',
  13: 'rRNA_pseudogene',
  14: 'sRNA',
  15: 'scaRNA',
  16: 'snRNA',
  17: 'snoRNA',
  18: 'transcribed_processed_pseudogene',
  19: 'transcribed_unitary_pseudogene',
  20: 'transcribed_unprocessed_pseudogene',
  21: 'unitary_pseudogene',
  22: 'unknown',
  23: 'unprocessed_pseudogene',
  24: 'vault_RNA',
};

function decodeKey(key, decodeMap) {
  const n = Number(key);
  if (!isNaN(n) && decodeMap[n] !== undefined) return decodeMap[n];
  return key;
}

function decodeDict(obj, decodeMap) {
  if (!obj) return obj;
  const result = {};
  for (const [k, v] of Object.entries(obj)) {
    result[decodeKey(k, decodeMap)] = v;
  }
  return result;
}

const COL_META = {
  sv_id:             { label: 'SV ID',           type: 'text',     w: 210, desc: 'Уникальный идентификатор структурного варианта' },
  svtype:            { label: 'Тип',             type: 'category', w: 75,  desc: 'Тип SV: DEL, DUP, INS, INV, BND, CNV' },
  svlen:             { label: 'Длина (bp)',       type: 'number',   w: 110, desc: 'Длина варианта в парах оснований' },
  af:                { label: 'AF',              type: 'number',   w: 95,  desc: 'Частота аллеля в популяции gnomAD' },
  gene_id:           { label: 'Gene ID',         type: 'text',     w: 170, desc: 'Ensembl ID гена' },
  gene_name:         { label: 'Ген',            type: 'text',     w: 110, desc: 'Символ гена (HGNC)' },
  gene_type:         { label: 'Биотип гена',    type: 'category', w: 160, desc: 'GENCODE биотип: protein_coding, lncRNA, pseudogene …' },
  overlap_type:      { label: 'Тип перекрытия', type: 'category', w: 120, desc: 'direct — SV перекрывает ген; proximal — SV рядом с геном' },
  overlap_frac:      { label: 'Ov SV',          type: 'number',   w: 85,  desc: 'Доля длины SV, перекрывающая ген (0–1)' },
  gene_ov_frac:      { label: 'Ov ген',         type: 'number',   w: 85,  desc: 'Доля длины гена, перекрытая SV (0–1)' },
  distance:          { label: 'Дист. (bp)',      type: 'number',   w: 110, desc: 'Расстояние от ближайшего края SV до гена (0 = перекрытие)' },
  median_tpm:        { label: 'TPM мед.',       type: 'number',   w: 90,  desc: 'Медианная экспрессия гена по GTEx (TPM)' },
  mean_tpm:          { label: 'TPM ср.',        type: 'number',   w: 90,  desc: 'Средняя экспрессия гена по GTEx (TPM)' },
  max_tpm:           { label: 'TPM макс.',      type: 'number',   w: 90,  desc: 'Максимальная экспрессия гена (TPM)' },
  log2_tpm:          { label: 'log₂(TPM)',      type: 'number',   w: 90,  desc: 'log₂-трансформация медианной TPM' },
  expressed_tissues: { label: 'Ткани',          type: 'number',   w: 75,  desc: 'Число GTEx-тканей, где TPM > 1' },
  tau:               { label: 'Tau',            type: 'number',   w: 70,  desc: 'Индекс тканевой специфичности (0 — убиквитарный, 1 — строго специфичный)' },
  pli:               { label: 'pLI',            type: 'number',   w: 70,  desc: 'P(loss-of-function intolerant): > 0.9 — ген чувствителен к гаплонедостаточности' },
  lof_oe:            { label: 'LoF O/E',        type: 'number',   w: 85,  desc: 'Отношение наблюдаемых/ожидаемых LoF-вариантов: ≤ 0.35 — ген под сильным очищающим отбором' },
  mis_z:             { label: 'Mis Z',          type: 'number',   w: 75,  desc: 'Z-score миссенс-вариантов: > 3 — ген чувствителен к миссенсам' },
  label:             { label: 'Влияние',        type: 'category', w: 95,  desc: '1 — SV влияет на экспрессию гена, 0 — нет' },
};

function svlenFmt(bp) {
  if (bp === null || bp === undefined) return null;
  bp = Number(bp);
  if (bp >= 1_000_000) return `${(bp / 1_000_000).toFixed(2)} Мб`;
  if (bp >= 1_000)     return `${(bp / 1_000).toFixed(1)} кб`;
  return `${bp.toLocaleString()} bp`;
}

function afFmt(v) {
  if (v === null || v === undefined) return null;
  const n = Number(v);
  if (Number.isNaN(n)) return String(v);
  if (n === 0) return '0';

  const str = n.toFixed(8).replace(/\.?(?:0)+$/, '');
  return str;
}

function numFmt(v, d = 3) {
  if (v === null || v === undefined) return null;
  const n = Number(v);
  if (isNaN(n)) return String(v);
  if (Number.isInteger(n)) return n.toLocaleString();
  return n.toFixed(d);
}

const GENE_TYPE_DISPLAY = {
  'protein_coding': 'Кодирующие белки',
  'lncRNA': 'Длинная некодирующая РНК (lncRNA)',
  'miRNA': 'микроРНК',
  'snoRNA': 'малая ядрышковая РНК (snoRNA)',
  'snRNA': 'малая ядерная РНК (snRNA)',
  'pseudogene': 'Псевдоген',
  'unprocessed_pseudogene': 'Необработанный псевдоген',
  'processed_pseudogene': 'Обработанный псевдоген',
  'transcribed_pseudogene': 'Транскрибируемый псевдоген',
  'transcribed_processed_pseudogene': 'Транскрибируемый обработанный псевдоген',
  'transcribed_unprocessed_pseudogene': 'Транскрибируемый необработанный псевдоген',
  'IG_V_gene': 'Ген вариабельного домена иммуноглобулина',
  'IG_D_gene': 'Ген D сегмента иммуноглобулина',
  'IG_J_gene': 'Ген J сегмента иммуноглобулина',
  'IG_C_gene': 'Ген константного домена иммуноглобулина',
  'TR_V_gene': 'Ген вариабельного домена TCR',
  'TR_D_gene': 'Ген D сегмента TCR',
  'TR_J_gene': 'Ген J сегмента TCR',
  'TR_C_gene': 'Ген константного домена TCR',
};

const OVERLAP_TYPE_DISPLAY = {
  'direct': 'Прямое перекрытие (SV внутри гена)',
  'proximal': 'Рядом (SV рядом с геном)',
};

const SIZE_CATEGORY_DISPLAY = {
  'Малые (<1 кб)': 'Малые (<1 кб)',
  'Мелкие (1-10 кб)': 'Мелкие (1-10 кб)',
  'Крупные (10-100 кб)': 'Крупные (10-100 кб)',
  'Очень крупные (>100 кб)': 'Очень крупные (>100 кб)',
  'Крупные (> 10 кб)': 'Крупные (> 10 кб)',
  'Средние (100 bp–1 kb)': 'Средние (100 bp–1 kb)',
  'Мелкие (100 bp–1 kb)': 'Мелкие (100 bp–1 kb)',
  'Маленькие (<100 bp)': 'Маленькие (<100 bp)',
};

function displayLabel(key, mapperObj) {
  return mapperObj[key] || key;
}

function CellValue({ col, value: v }) {
  if (v === null || v === undefined) return <span style={{ opacity: 0.22 }}>—</span>;

  switch (col) {
    case 'svlen': {
      const s = svlenFmt(v);
      return s ? <span>{s}</span> : <span style={{ opacity: 0.22 }}>—</span>;
    }
    case 'af': {
      const s = afFmt(v);
      return s ? <span>{s}</span> : <span style={{ opacity: 0.22 }}>—</span>;
    }
    case 'label':
      return (v === 1 || v === '1')
        ? <span style={{ color: '#e74c3c', fontWeight: 700 }}>● Влияет</span>
        : <span style={{ color: '#6c757d' }}>○ Нет</span>;
    case 'svtype': {
      const name = decodeKey(v, SVTYPE_DECODE);
      const c = SV_COLORS[name] || '#aaa';
      return (
        <span style={{
          display: 'inline-block', padding: '1px 7px', borderRadius: 4,
          border: `1px solid ${c}`, color: c, fontWeight: 700, fontSize: 11, letterSpacing: 0.5
        }}>{name}</span>
      );
    }
    case 'gene_type': {
      const raw = decodeKey(v, GENE_TYPE_DECODE);
      const label = displayLabel(raw, GENE_TYPE_DISPLAY);
      return <span style={{ fontSize: 11 }}>{label}</span>;
    }
    case 'pli': {
      const n = parseFloat(v);
      const c = n > 0.9 ? '#e74c3c' : n > 0.5 ? '#f39c12' : '#aaa';
      return <span style={{ color: c }}>{numFmt(v)}</span>;
    }
    case 'lof_oe': {
      const n = parseFloat(v);
      const c = n <= 0.35 ? '#e74c3c' : n <= 0.7 ? '#f39c12' : '#aaa';
      return <span style={{ color: c }}>{numFmt(v)}</span>;
    }
    case 'mis_z': {
      const n = parseFloat(v);
      const c = n > 3 ? '#e74c3c' : n > 1 ? '#f39c12' : '#aaa';
      return <span style={{ color: c }}>{numFmt(v)}</span>;
    }
    case 'tau': {
      const n = parseFloat(v);
      const c = n > 0.7 ? '#9b59b6' : n > 0.3 ? '#f39c12' : '#3498db';
      return <span style={{ color: c }}>{numFmt(v)}</span>;
    }
    case 'distance':
      return v === 0 || v === '0'
        ? <span style={{ color: '#e74c3c', fontWeight: 600 }}>0</span>
        : <span>{Number(v).toLocaleString()}</span>;
    case 'overlap_frac':
    case 'gene_ov_frac': return <span>{numFmt(v, 4)}</span>;
    default:
      if (typeof v === 'number') return <span>{numFmt(v)}</span>;
      return <span>{String(v)}</span>;
  }
}

function BarRow({ label, count, total, color = 'var(--blue)' }) {
  const w = total > 0 ? Math.max(2, (count / total) * 100) : 0;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 5, fontSize: 12 }}>
      <div style={{ minWidth: 180, maxWidth: 180, opacity: 0.82, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{label}</div>
      <div style={{ flex: 1, background: 'rgba(255,255,255,0.08)', borderRadius: 3, height: 15, overflow: 'hidden' }}>
        <div style={{ width: `${w}%`, height: '100%', background: color, borderRadius: 3, transition: 'width 0.4s ease' }} />
      </div>
      <div style={{ minWidth: 90, textAlign: 'right', opacity: 0.9 }}>
        {count.toLocaleString()} <span style={{ opacity: 0.4, fontSize: 11 }}>({w.toFixed(1)}%)</span>
      </div>
    </div>
  );
}

function StatCard({ title, iconClass, note, children, style }) {
  return (
    <div className="sv-panel" style={{ marginBottom: 0, ...style }}>
      <div style={{ fontWeight: 700, fontSize: 13, marginBottom: note ? 6 : 12, display: 'flex', alignItems: 'center', gap: 8 }}>
        {iconClass ? <i className={iconClass} style={{ fontSize: '1.12rem', opacity: 0.88 }} aria-hidden /> : null}
        <span>{title}</span>
      </div>
      {note && <div style={{ fontSize: 11, opacity: 0.55, marginBottom: 12, lineHeight: 1.5 }}>{note}</div>}
      {children}
    </div>
  );
}

function MetricPill({ label, value, color, sub }) {
  return (
    <div style={{
      background: 'rgba(255,255,255,0.055)', borderRadius: 10,
      padding: '12px 18px', textAlign: 'center', flex: '1 1 100px', minWidth: 100,
    }}>
      <div style={{ fontSize: 24, fontWeight: 700, color: color || 'inherit', lineHeight: 1.1 }}>{value}</div>
      {sub && <div style={{ fontSize: 12, color: color || 'inherit', opacity: 0.65, marginTop: 2 }}>{sub}</div>}
      <div style={{ fontSize: 11, opacity: 0.5, marginTop: 4 }}>{label}</div>
    </div>
  );
}

function StatsTab({ info }) {
  if (!info) return null;
  const total = info.total_sv || 0;
  const G2 = { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 };
  const G3 = { display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginBottom: 12 };

  const svPerType   = decodeDict(info.sv_per_type,  SVTYPE_DECODE);
  const sizeByType  = decodeDict(info.size_by_type, SVTYPE_DECODE);
  const geneTypes   = decodeDict(info.gene_types,   GENE_TYPE_DECODE);

  return (
    <div>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 12 }}>
        <MetricPill label="Всего SV" value={total.toLocaleString()} color="var(--blue)" />
        {info.has_label && (
          <>
            <MetricPill
              label="Влияют на ген (label=1)"
              value={`${(info.label_positive || 0).toLocaleString()}`}
              sub={`${(info.label_positive_pct || 0).toFixed(1)}% от всех`}
              color="#e74c3c"
            />
            <MetricPill
              label="Нет влияния (label=0)"
              value={(info.label_negative || 0).toLocaleString()}
              color="#6c757d"
            />
          </>
        )}
        {info.af_median != null && (
          <MetricPill label="Медианный AF" value={afFmt(info.af_median)} color="#f39c12" />
        )}
        {info.sv_per_type && (
          <MetricPill label="Типов SV" value={Object.keys(info.sv_per_type).length} color="#9b59b6" />
        )}
      </div>

      <div style={{ ...G2, marginBottom: 12 }}>
        {svPerType && Object.keys(svPerType).length > 0 && (
          <StatCard title="Типы структурных вариантов" iconClass="bi bi-diagram-3"
            note="DEL — делеция; DUP — дупликация; INS — инсерция; INV — инверсия; BND — транслокация; CNV — вариация числа копий">
            {Object.entries(svPerType)
              .sort((a, b) => b[1] - a[1])
              .map(([t, c]) => (
                <BarRow key={t} label={t} count={c} total={total} color={SV_COLORS[t] || '#aaa'} />
              ))}
          </StatCard>
        )}
        {info.size_categories && (
          <StatCard title="Распределение по длине SV" iconClass="bi bi-rulers"
            note="Крупные SV (> 10 кб) с большей вероятностью затрагивают регуляторные элементы и несколько генов одновременно">
            {Object.entries(info.size_categories).map(([cat, c]) => (
              <BarRow key={cat} label={displayLabel(cat, SIZE_CATEGORY_DISPLAY)} count={c} total={total} color="#3498db" />
            ))}
          </StatCard>
        )}
      </div>

      {sizeByType && Object.keys(sizeByType).length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <StatCard title="Размер SV по типам" iconClass="bi bi-arrows-angle-expand"
            note="Медиана, среднее и квантили длины. INS обычно короткие; CNV/DUP — крупнее.">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr>
                    {['Тип', 'Медиана', 'Среднее', 'Мин.', 'Макс.', 'P25', 'P75'].map(h => (
                      <th key={h} style={{
                        padding: '5px 10px', textAlign: h === 'Тип' ? 'left' : 'right',
                        opacity: 0.55, fontWeight: 700, fontSize: 11,
                        borderBottom: '1px solid rgba(255,255,255,0.08)', whiteSpace: 'nowrap'
                      }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(sizeByType).map(([t, s]) => (
                    <tr key={t}>
                      <td style={{ padding: '5px 10px', fontWeight: 700, color: SV_COLORS[t] || '#aaa' }}>{t}</td>
                      {['median', 'mean', 'min', 'max', 'p25', 'p75'].map(k => (
                        <td key={k} style={{ padding: '5px 10px', textAlign: 'right' }}>
                          {s[k] != null ? svlenFmt(s[k]) : <span style={{ opacity: 0.3 }}>—</span>}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </StatCard>
        </div>
      )}

      <div style={{ ...G2, marginBottom: 12 }}>
        {info.overlap_types && Object.keys(info.overlap_types).length > 0 && (
          <StatCard title="Тип перекрытия SV с геном" iconClass="bi bi-link-45deg"
            note="direct — SV перекрывает тело гена напрямую (высокий риск). proximal — SV находится рядом, воздействуя через регуляторные элементы.">
            {Object.entries(info.overlap_types)
              .sort((a, b) => b[1] - a[1])
              .map(([t, c]) => (
                <BarRow key={t} label={displayLabel(t, OVERLAP_TYPE_DISPLAY)} count={c} total={total} color="#9b59b6" />
              ))}
          </StatCard>
        )}
        {geneTypes && Object.keys(geneTypes).length > 0 && (
          <StatCard title="Биотип затронутых генов" iconClass="bi bi-heart-pulse"
            note="protein_coding — приоритетны для клинической интерпретации. lncRNA и псевдогены — менее изучены, но могут влиять на регуляцию.">
            {Object.entries(geneTypes)
              .sort((a, b) => b[1] - a[1])
              .slice(0, 8)
              .map(([t, c]) => (
                <BarRow key={t} label={displayLabel(t, GENE_TYPE_DISPLAY)} count={c} total={total} color="#1abc9c" />
              ))}
          </StatCard>
        )}
      </div>

      <div style={{ ...G2, marginBottom: 12 }}>
        {info.af_categories && (
          <StatCard title="Популяционная частота (AF)" iconClass="bi bi-people"
            note="Редкие варианты (AF < 1%) статистически ассоциированы с патогенностью. Частые (> 5%) — скорее всего, нейтральные полиморфизмы.">
            {Object.entries(info.af_categories).map(([cat, c]) => (
              <BarRow key={cat} label={cat} count={c} total={total} color="#f39c12" />
            ))}
          </StatCard>
        )}
        {info.expression_by_label && (
          <StatCard title="Экспрессия гена vs влияние SV" iconClass="bi bi-graph-up-arrow"
            note="Гены с высокой экспрессией (высокий TPM) чаще демонстрируют аномалии при наличии SV. Сравните медианный уровень между классами.">
            {[
              { key: 'positive', label: 'Влияет (label = 1)', color: '#e74c3c' },
              { key: 'neutral',  label: 'Не влияет (label = 0)', color: '#6c757d' },
            ].map(({ key, label, color }) => {
              const d = info.expression_by_label[key];
              if (!d) return null;
              return (
                <div key={key} style={{ marginBottom: 16, paddingBottom: 12, borderBottom: '1px solid rgba(255,255,255,0.07)' }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color, marginBottom: 8 }}>{label}</div>
                  <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
                    {[
                      ['Медиана TPM', d.median_tpm],
                      ['Среднее TPM', d.mean_tpm],
                    ].map(([lbl, val]) => (
                      <div key={lbl}>
                        <div style={{ fontSize: 11, opacity: 0.55 }}>{lbl}</div>
                        <div style={{ fontSize: 20, fontWeight: 700 }}>{numFmt(val)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
            {info.tissues_by_label && (
              <div style={{ marginTop: 4 }}>
                <div style={{ fontSize: 11, opacity: 0.55, marginBottom: 6 }}>Медианное число экспрессирующих тканей</div>
                <div style={{ display: 'flex', gap: 20 }}>
                  {[
                    { key: 'positive', label: 'Влияет', color: '#e74c3c' },
                    { key: 'neutral',  label: 'Не влияет', color: '#6c757d' },
                  ].map(({ key, label, color }) => {
                    const d = info.tissues_by_label[key];
                    if (!d) return null;
                    return (
                      <div key={key}>
                        <div style={{ fontSize: 11, color, opacity: 0.8 }}>{label}</div>
                        <div style={{ fontSize: 18, fontWeight: 700 }}>{numFmt(d.median, 1)}</div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </StatCard>
        )}
      </div>

      <div style={{ ...G3, marginBottom: 12 }}>
        {info.pli_categories && (
          <StatCard title="Гаплонедостаточность (pLI)" iconClass="bi bi-exclamation-triangle"
            note="pLI > 0.9: потеря одной копии, вероятно, патогенна. Источник: gnomAD.">
            {[
              ['Высокое (pLI > 0.9)', '#e74c3c'],
              ['Среднее (0.5–0.9)', '#f39c12'],
              ['Низкое (< 0.5)', '#6c757d'],
            ].map(([k, c]) => info.pli_categories[k] != null && (
              <BarRow key={k} label={k} count={info.pli_categories[k]} total={total} color={c} />
            ))}
          </StatCard>
        )}
        {info.lof_oe_categories && (
          <StatCard title="Очищающий отбор LoF (O/E)" iconClass="bi bi-microscope"
            note="LoF O/E ≤ 0.35: ген под сильным давлением — LoF-варианты крайне редки в популяции.">
            {[
              ['Под отбором (≤ 0.35)', '#e74c3c'],
              ['Промежуточный (0.35–0.7)', '#f39c12'],
              ['Толерантный (> 0.7)', '#6c757d'],
            ].map(([k, c]) => info.lof_oe_categories[k] != null && (
              <BarRow key={k} label={k} count={info.lof_oe_categories[k]} total={total} color={c} />
            ))}
          </StatCard>
        )}
        {info.tau_categories && (
          <StatCard title="Тканевая специфичность (Tau)" iconClass="bi bi-activity"
            note="Tau ≈ 1: ген строго специфичен для одной ткани — SV может иметь узконаправленный эффект.">
            {[
              ['Убиквитарные (tau < 0.3)', '#3498db'],
              ['Умеренные (0.3–0.7)', '#f39c12'],
              ['Тканеспецифичные (tau > 0.7)', '#9b59b6'],
            ].map(([k, c]) => info.tau_categories[k] != null && (
              <BarRow key={k} label={k} count={info.tau_categories[k]} total={total} color={c} />
            ))}
          </StatCard>
        )}
      </div>

      {info.top_genes && info.top_genes.length > 0 && (
        <StatCard title="Топ-10 генов по числу SV" iconClass="bi bi-trophy"
          note="Гены, затронутые наибольшим числом структурных вариантов в датасете.">
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {info.top_genes.map(({ gene, count }, i) => (
              <div key={gene} style={{
                background: 'rgba(255,255,255,0.07)', borderRadius: 8,
                padding: '8px 14px', fontSize: 13,
              }}>
                <span style={{ opacity: 0.4, fontSize: 11, marginRight: 5 }}>#{i + 1}</span>
                <strong>{gene}</strong>
                <span style={{ opacity: 0.45, marginLeft: 8, fontSize: 11 }}>{count} SV</span>
              </div>
            ))}
          </div>
        </StatCard>
      )}
    </div>
  );
}

const INITIAL_FILTERS = {
  search: '',
  filter_svtype: '',
  filter_gene_type: '',
  filter_overlap_type: '',
  filter_label: '',
  filter_svlen_min: '',
  filter_svlen_max: '',
  filter_af_min: '',
  filter_af_max: '',
  filter_pli_min: '',
  filter_lof_oe_max: '',
};

function FilterBar({ info, filters, onChange, onReset }) {
  const sel = {
    background: 'var(--input-bg, rgba(255,255,255,0.07))',
    border: '1px solid var(--border, rgba(255,255,255,0.15))',
    color: 'inherit', borderRadius: 6, padding: '5px 9px', fontSize: 12, cursor: 'pointer',
  };
  const inp = { ...sel, cursor: 'text', width: 80 };

  const svTypeEntries   = info?.sv_per_type
    ? Object.keys(info.sv_per_type).map(k => ({ value: k, label: decodeKey(k, SVTYPE_DECODE) }))
    : [];
  const geneTypeEntries = info?.gene_types
    ? Object.keys(info.gene_types).slice(0, 20).map(k => ({ value: k, label: displayLabel(decodeKey(k, GENE_TYPE_DECODE), GENE_TYPE_DISPLAY) }))
    : [];
  const overlapTypes = info?.overlap_types ? Object.keys(info.overlap_types) : [];

  function field(key) {
    return {
      value: filters[key] || '',
      onChange: e => onChange({ ...filters, [key]: e.target.value }),
    };
  }

  const hasFilters = Object.entries(filters).some(([, v]) => v !== '');

  return (
    <div className="sv-panel" style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'flex-end' }}>

        <div>
          <div style={{ fontSize: 11, opacity: 0.5, marginBottom: 3 }}>Поиск (ID, ген)</div>
          <input style={{ ...inp, width: 160 }} placeholder="sv_id, ген…" {...field('search')} />
        </div>

        {svTypeEntries.length > 0 && (
          <div>
            <div style={{ fontSize: 11, opacity: 0.5, marginBottom: 3 }}>Тип SV</div>
            <select style={sel} {...field('filter_svtype')}>
              <option value="">Все типы</option>
              {svTypeEntries.map(({ value, label }) => <option key={value} value={value}>{label}</option>)}
            </select>
          </div>
        )}

        {geneTypeEntries.length > 0 && (
          <div>
            <div style={{ fontSize: 11, opacity: 0.5, marginBottom: 3 }}>Биотип гена</div>
            <select style={sel} {...field('filter_gene_type')}>
              <option value="">Все биотипы</option>
              {geneTypeEntries.map(({ value, label }) => <option key={value} value={value}>{label}</option>)}
            </select>
          </div>
        )}

        {overlapTypes.length > 0 && (
          <div>
            <div style={{ fontSize: 11, opacity: 0.5, marginBottom: 3 }}>Перекрытие</div>
            <select style={sel} {...field('filter_overlap_type')}>
              <option value="">Все</option>
              {overlapTypes.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
        )}

        {info?.has_label && (
          <div>
            <div style={{ fontSize: 11, opacity: 0.5, marginBottom: 3 }}>Влияние</div>
            <select style={sel} {...field('filter_label')}>
              <option value="">Все</option>
              <option value="1">● Влияет</option>
              <option value="0">○ Нет</option>
            </select>
          </div>
        )}

        <div>
          <div style={{ fontSize: 11, opacity: 0.5, marginBottom: 3 }}>Длина SV (bp)</div>
          <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
            <input style={inp} placeholder="мин" type="number" min="0" {...field('filter_svlen_min')} />
            <span style={{ opacity: 0.4 }}>–</span>
            <input style={inp} placeholder="макс" type="number" min="0" {...field('filter_svlen_max')} />
          </div>
        </div>

        <div>
          <div style={{ fontSize: 11, opacity: 0.5, marginBottom: 3 }}>AF (0–1)</div>
          <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
            <input style={inp} placeholder="мин" type="number" step="0.0001" min="0" max="1" {...field('filter_af_min')} />
            <span style={{ opacity: 0.4 }}>–</span>
            <input style={inp} placeholder="макс" type="number" step="0.0001" min="0" max="1" {...field('filter_af_max')} />
          </div>
        </div>

        <div>
          <div style={{ fontSize: 11, opacity: 0.5, marginBottom: 3 }}>pLI ≥</div>
          <input style={inp} placeholder="напр. 0.9" type="number" step="0.1" min="0" max="1" {...field('filter_pli_min')} />
        </div>

        <div>
          <div style={{ fontSize: 11, opacity: 0.5, marginBottom: 3 }}>LoF O/E ≤</div>
          <input style={inp} placeholder="напр. 0.35" type="number" step="0.05" min="0" {...field('filter_lof_oe_max')} />
        </div>

        {hasFilters && (
          <button
            onClick={onReset}
            style={{
              ...sel, cursor: 'pointer', alignSelf: 'flex-end',
              border: '1px solid var(--red-btn, #e74c3c)',
              color: 'var(--red-btn, #e74c3c)', background: 'transparent',
            }}
          >
            Сбросить
          </button>
        )}
      </div>
    </div>
  );
}

function Tooltip({ text, children }) {
  const [show, setShow] = useState(false);
  return (
    <span
      style={{ position: 'relative', cursor: 'help' }}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      {show && (
        <div style={{
          position: 'absolute', zIndex: 99, bottom: '100%', left: '50%', transform: 'translateX(-50%)',
          background: '#1e2535', border: '1px solid rgba(255,255,255,0.15)', borderRadius: 6,
          padding: '6px 10px', fontSize: 11, whiteSpace: 'normal', width: 220,
          boxShadow: '0 4px 12px rgba(0,0,0,0.4)', lineHeight: 1.5, opacity: 0.97,
          pointerEvents: 'none', marginBottom: 4, color: '#d0d6e0',
        }}>
          {text}
        </div>
      )}
    </span>
  );
}

function SvTable({ columns, rows, sortBy, sortDir, onSort }) {
  if (!rows.length) {
    return (
      <div className="sv-panel sv-empty" style={{ marginTop: 0, borderRadius: '0 0 8px 8px' }}>
        Нет строк, соответствующих фильтрам.
      </div>
    );
  }

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr>
            {columns.map(col => {
              const meta = COL_META[col] || { label: col, type: 'text', w: 100 };
              const sorted = sortBy === col;
              return (
                <th
                  key={col}
                  onClick={() => onSort(col)}
                  style={{
                    padding: '7px 10px', whiteSpace: 'nowrap', cursor: 'pointer',
                    userSelect: 'none', fontWeight: 700, fontSize: 11, opacity: sorted ? 1 : 0.65,
                    borderBottom: '1px solid rgba(255,255,255,0.1)',
                    textAlign: meta.type === 'number' ? 'right' : 'left',
                    minWidth: meta.w,
                    background: sorted ? 'rgba(52,152,219,0.08)' : 'transparent',
                    color: sorted ? 'var(--blue, #3498db)' : 'inherit',
                  }}
                >
                  <Tooltip text={meta.desc || meta.label}>
                    <span>{meta.label}</span>
                  </Tooltip>
                  {' '}
                  <span style={{ fontSize: 10, opacity: sorted ? 0.9 : 0.3 }}>
                    {sorted ? (sortDir === 'asc' ? '↑' : '↓') : '↕'}
                  </span>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={i}
              style={{
                background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.022)',
                transition: 'background 0.1s',
              }}
              onMouseEnter={e => (e.currentTarget.style.background = 'rgba(52,152,219,0.07)')}
              onMouseLeave={e => (e.currentTarget.style.background = i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.022)')}
            >
              {columns.map(col => {
                const meta = COL_META[col] || { type: 'text' };
                return (
                  <td
                    key={col}
                    style={{
                      padding: '6px 10px',
                      textAlign: meta.type === 'number' ? 'right' : 'left',
                      borderBottom: '1px solid rgba(255,255,255,0.05)',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    <CellValue col={col} value={row[col]} />
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Pagination({ page, totalPages, total, perPage, onPage }) {
  if (totalPages <= 1) return null;

  const pages = [];
  const delta = 2;
  for (let i = 1; i <= totalPages; i++) {
    if (i === 1 || i === totalPages || (i >= page - delta && i <= page + delta)) {
      pages.push(i);
    } else if (pages[pages.length - 1] !== '…') {
      pages.push('…');
    }
  }

  const btn = (active, disabled) => ({
    padding: '4px 10px', borderRadius: 4, fontSize: 12, cursor: disabled ? 'default' : 'pointer',
    border: active ? '1px solid var(--blue, #3498db)' : '1px solid rgba(255,255,255,0.12)',
    background: active ? 'var(--blue, #3498db)' : 'transparent',
    color: active ? '#fff' : 'inherit', opacity: disabled ? 0.35 : 1,
  });

  const start = (page - 1) * perPage + 1;
  const end = Math.min(page * perPage, total);

  return (
    <div style={{ display: 'flex', gap: 6, alignItems: 'center', justifyContent: 'center', marginTop: 14, flexWrap: 'wrap' }}>
      <span style={{ fontSize: 12, opacity: 0.45, marginRight: 8 }}>
        {start.toLocaleString()}–{end.toLocaleString()} из {total.toLocaleString()}
      </span>
      <button style={btn(false, page === 1)} disabled={page === 1} onClick={() => onPage(page - 1)}>←</button>
      {pages.map((p, i) =>
        p === '…'
          ? <span key={`e${i}`} style={{ padding: '4px 2px', opacity: 0.35 }}>…</span>
          : <button key={p} style={btn(p === page, false)} onClick={() => onPage(p)}>{p}</button>
      )}
      <button style={btn(false, page === totalPages)} disabled={page === totalPages} onClick={() => onPage(page + 1)}>→</button>
    </div>
  );
}

export default function DatasetInfoPage() {
  const { datasetsVersion } = useTaskContext();

  const [datasets,   setDatasets]   = useState([]);
  const [selected,   setSelected]   = useState('');
  const [svInfo,     setSvInfo]     = useState(null);
  const [rows,       setRows]       = useState([]);
  const [columns,    setColumns]    = useState([]);
  const [pagination, setPagination] = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [error,      setError]      = useState(null);
  const [activeTab,  setActiveTab]  = useState('stats');

  const [sortBy,  setSortBy]  = useState('');
  const [sortDir, setSortDir] = useState('asc');
  const [page,    setPage]    = useState(1);
  const [filters, setFilters] = useState(INITIAL_FILTERS);
  const [filterDraft, setFilterDraft] = useState(INITIAL_FILTERS);
  const filterDebounceTimeout = useRef(null);
  const FILTER_DEBOUNCE_MS = 800;

  useEffect(() => {
    api.listDatasets()
      .then(res => setDatasets(res.data || []))
      .catch(() => {});
  }, [datasetsVersion]);

  const buildParams = useCallback((overrides = {}) => {
    const state = { sortBy, sortDir, page, filters, ...overrides };
    const p = {
      page:     state.page,
      per_page: 50,
      ...(state.sortBy ? { sort_by: state.sortBy, sort_dir: state.sortDir } : {}),
    };
    Object.entries(state.filters).forEach(([k, v]) => {
      if (v !== '' && v !== null && v !== undefined) {
        p[k] = v;
      }
    });
    return p;
  }, [sortBy, sortDir, page, filters]);

  const fetchData = useCallback((datasetId, overrides = {}) => {
    if (!datasetId) return;
    setLoading(true);
    setError(null);
    api.getDatasetSvInfo(datasetId, buildParams(overrides))
      .then(res => {
        const d = res.data;
        setSvInfo(d.sv_info || null);
        setRows(d.rows || []);
        setColumns(d.columns || []);
        setPagination(d.pagination || null);
      })
      .catch(e => setError(e?.response?.data?.error || e?.message || 'Ошибка загрузки'))
      .finally(() => setLoading(false));
  }, [buildParams]);

  useEffect(() => {
    if (selected) {
      fetchData(selected);
    } else {
      setSvInfo(null); setRows([]); setColumns([]); setPagination(null);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, page, sortBy, sortDir, filters]);

  function handleSort(col) {
    if (sortBy === col) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(col);
      setSortDir('asc');
    }
    setPage(1);
  }

  function handleFiltersChange(newFilters) {
    setFilterDraft(newFilters);
    setPage(1);
    if (filterDebounceTimeout.current) {
      clearTimeout(filterDebounceTimeout.current);
    }
    filterDebounceTimeout.current = setTimeout(() => {
      setFilters(newFilters);
      filterDebounceTimeout.current = null;
    }, FILTER_DEBOUNCE_MS);
  }

  useEffect(() => {
    return () => {
      if (filterDebounceTimeout.current) {
        clearTimeout(filterDebounceTimeout.current);
      }
    };
  }, []);

  function cancelPendingFilterDebounce() {
    if (filterDebounceTimeout.current) {
      clearTimeout(filterDebounceTimeout.current);
      filterDebounceTimeout.current = null;
    }
  }

  function handleReset() {
    cancelPendingFilterDebounce();
    setFilterDraft(INITIAL_FILTERS);
    setFilters(INITIAL_FILTERS);
    setSortBy('');
    setSortDir('asc');
    setPage(1);
  }

  function handleDatasetChange(e) {
    cancelPendingFilterDebounce();
    setSelected(e.target.value);
    setPage(1);
    setFilterDraft(INITIAL_FILTERS);
    setFilters(INITIAL_FILTERS);
    setSortBy('');
    setSortDir('asc');
    setSvInfo(null);
    setRows([]);
  }

  const tabBtn = (key) => ({
    padding: '8px 22px', borderRadius: '6px 6px 0 0', fontSize: 13, fontWeight: 600,
    border: '1px solid rgba(255,255,255,0.1)', borderBottom: activeTab === key ? 'none' : '1px solid rgba(255,255,255,0.1)',
    cursor: 'pointer', transition: 'background 0.15s',
    background: activeTab === key ? 'var(--panel-bg, rgba(255,255,255,0.04))' : 'transparent',
    color: activeTab === key ? 'var(--blue, #3498db)' : 'inherit',
    marginBottom: -1,
  });

  const hasActiveFilters = Object.values(filters).some(v => v !== '');

  return (
    <div>
      <h2 className="sv-page-title">Структурные варианты</h2>

      <div className="sv-panel">
        <label className="sv-label">Выберите датасет</label>
        <select className="sv-select" value={selected} onChange={handleDatasetChange}>
          <option value="">Выберите датасет…</option>
          {datasets.map(d => (
            <option key={d.id} value={d.id}>
              {d.name}{d.row_count ? ` — ${d.row_count.toLocaleString()} строк` : ''}
            </option>
          ))}
        </select>
      </div>

      {loading && (
        <div className="sv-panel sv-empty">
          <span className="sv-spinner" style={{ marginBottom: 8 }} />
          <div>Загрузка…</div>
        </div>
      )}

      {error && !loading && (
        <div className="sv-alert sv-alert-danger">
          <strong>Ошибка:</strong> {error}
        </div>
      )}

      {svInfo && !loading && (
        <>
          <div style={{ display: 'flex', gap: 0, marginBottom: 0, marginTop: 4 }}>
            <button style={tabBtn('stats')} type="button" onClick={() => setActiveTab('stats')}>
              <i className="bi bi-bar-chart-line me-1" aria-hidden />
              Статистика
            </button>
            <button style={tabBtn('data')} type="button" onClick={() => setActiveTab('data')}>
              <i className="bi bi-table me-1" aria-hidden />
              Данные
              {hasActiveFilters && (
                <span style={{
                  marginLeft: 6, background: 'var(--blue, #3498db)', color: '#fff',
                  borderRadius: 10, fontSize: 10, padding: '1px 6px',
                }}>
                  ф
                </span>
              )}
            </button>
          </div>

          <div className="sv-panel" style={{ borderRadius: '0 6px 6px 6px', marginTop: 0 }}>
            {activeTab === 'stats' && <StatsTab info={svInfo} />}

            {activeTab === 'data' && (
              <>
                <FilterBar
                  info={svInfo}
                  filters={filterDraft}
                  onChange={handleFiltersChange}
                  onReset={handleReset}
                />

                <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10, fontSize: 12, opacity: 0.65 }}>
                  <span>
                    {pagination
                      ? `${pagination.total.toLocaleString()} строк (стр. ${pagination.page} из ${pagination.total_pages})`
                      : `${rows.length} строк`}
                  </span>
                  {hasActiveFilters && svInfo?.total_sv && pagination && (
                    <span style={{ color: '#f39c12' }}>
                      ≈ {((pagination.total / svInfo.total_sv) * 100).toFixed(1)}% от датасета
                    </span>
                  )}
                  {sortBy && (
                    <span>
                      Сортировка: <strong>{COL_META[sortBy]?.label || sortBy}</strong> {sortDir === 'asc' ? '↑' : '↓'}
                      <button
                        onClick={() => { setSortBy(''); setSortDir('asc'); setPage(1); }}
                        style={{ marginLeft: 6, background: 'none', border: 'none', cursor: 'pointer', color: 'var(--red-btn, #e74c3c)', fontSize: 11 }}
                        type="button"
                        aria-label="Сбросить сортировку"
                      >
                        <i className="bi bi-x-lg" aria-hidden />
                      </button>
                    </span>
                  )}
                </div>

                <div style={{ border: '1px solid rgba(255,255,255,0.08)', borderRadius: 6, overflow: 'hidden' }}>
                  <SvTable
                    columns={columns}
                    rows={rows}
                    sortBy={sortBy}
                    sortDir={sortDir}
                    onSort={handleSort}
                  />
                </div>

                {pagination && (
                  <Pagination
                    page={pagination.page}
                    totalPages={pagination.total_pages}
                    total={pagination.total}
                    perPage={pagination.per_page}
                    onPage={setPage}
                  />
                )}
              </>
            )}
          </div>
        </>
      )}

      {!selected && !loading && (
        <div className="sv-panel sv-empty">
          Выберите датасет, чтобы просмотреть структурные варианты и статистику по ним.
        </div>
      )}
    </div>
  );
}