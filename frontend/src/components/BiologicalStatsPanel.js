import React, { useState } from 'react';

const C = {
  affectedHigh:    { bg: '#c0122c', border: '#f6225a', text: '#fff',   dot: '#f6225a'  },
  affectedLow:     { bg: '#6b1a2e', border: '#d06080', text: '#f0b0c0', dot: '#d06080' },
  neutralHigh:     { bg: '#0e4a20', border: '#2ff849', text: '#d0ffd8', dot: '#2ff849' },
  neutralLow:      { bg: '#1a3d24', border: '#3a9455', text: '#a0d8b0', dot: '#3a9455' },
  uncertain:       { bg: '#3a3010', border: '#c9a227', text: '#f5e89a', dot: '#c9a227' },
};

function categoryColors(cat) {
  if (cat === 'affected')          return C.affectedHigh;
  if (cat === 'uncertain_affected') return C.affectedLow;
  if (cat === 'neutral')           return C.neutralHigh;
  if (cat === 'uncertain_neutral') return C.neutralLow;
  return C.uncertain;
}

function categoryLabel(cat, geneName) {
  if (cat === 'affected')           return `${geneName} — влияет (высокая уверенность)`;
  if (cat === 'uncertain_affected') return `${geneName} — вероятно влияет`;
  if (cat === 'neutral')            return `${geneName} — нейтрально (высокая уверенность)`;
  if (cat === 'uncertain_neutral')  return `${geneName} — вероятно нейтрально`;
  return geneName;
}

function svTypeBadgeColor(svType) {
  const t = (svType || '').toUpperCase();
  if (t === 'DEL') return { bg: '#5a0e1e', border: '#f6225a', color: '#f6225a' };
  if (t === 'DUP') return { bg: '#0a3060', border: '#2B72FB', color: '#7ab8ff' };
  if (t === 'INV') return { bg: '#3a2800', border: '#c9a227', color: '#f5d97a' };
  if (t === 'BND') return { bg: '#2a0a4a', border: '#9b59b6', color: '#d7a8f0' };
  if (t === 'INS') return { bg: '#0a3a2a', border: '#2ff849', color: '#7fffb0' };
  return { bg: '#2a2a2a', border: '#888', color: '#ccc' };
}

function ConfBar({ value, category }) {
  const col = categoryColors(category);
  const pct = Math.round((value ?? 0) * 100);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <div style={{ flex: 1, height: 6, borderRadius: 3, background: 'rgba(255,255,255,0.08)', overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', borderRadius: 3, background: col.border, transition: 'width .4s ease' }} />
      </div>
      <span style={{ fontSize: 11, fontVariantNumeric: 'tabular-nums', color: col.dot, minWidth: 36, textAlign: 'right' }}>
        {pct}%
      </span>
    </div>
  );
}

function GeneRow({ gene, rank }) {
  const col = categoryColors(gene.category);
  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: '22px 1fr 200px',
      alignItems: 'center',
      gap: 10,
      padding: '6px 10px',
      borderRadius: 6,
      background: col.bg,
      border: `1px solid ${col.border}22`,
      marginBottom: 4,
    }}>
      <span style={{ fontSize: 10, opacity: 0.5, fontVariantNumeric: 'tabular-nums', textAlign: 'center' }}>{rank}</span>

      <div>
        <span style={{ fontWeight: 700, fontSize: 13, color: col.dot, letterSpacing: '0.02em' }}>
          {gene.gene_name}
        </span>
        <span style={{ fontSize: 11, opacity: 0.75, marginLeft: 8, color: col.text }}>
          {categoryLabel(gene.category, '').trim()}
        </span>
      </div>

      <ConfBar value={gene.confidence} category={gene.category} />
    </div>
  );
}

function LegendPill({ color, label }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11 }}>
      <div style={{ width: 10, height: 10, borderRadius: 2, background: color, flexShrink: 0 }} />
      <span style={{ opacity: 0.8 }}>{label}</span>
    </div>
  );
}

function SVCard({ sv }) {
  const [expanded, setExpanded] = useState(sv.gene_details.length <= 6);

  const badgeStyle = svTypeBadgeColor(sv.sv_type);
  const genes = sv.gene_details; // already sorted by abs_confidence desc from backend

  const displayGenes = expanded ? genes : genes.slice(0, 5);
  const hidden = genes.length - 5;

  const summaryParts = [];
  if (sv.num_affected > 0)
    summaryParts.push(
      <span key="a" style={{ color: '#f6225a', fontWeight: 700 }}>
        {sv.num_affected} ген{sv.num_affected > 1 ? 'а' : ''} затронут{sv.num_affected > 1 ? 'о' : ''}
      </span>
    );
  if (sv.num_uncertain > 0)
    summaryParts.push(
      <span key="u" style={{ color: '#c9a227' }}>
        {sv.num_uncertain} неопределён{sv.num_uncertain > 1 ? 'о' : ''}
      </span>
    );
  if (sv.num_neutral > 0)
    summaryParts.push(
      <span key="n" style={{ color: '#2ff849' }}>
        {sv.num_neutral} нейтральн{sv.num_neutral > 1 ? 'ых' : 'ый'}
      </span>
    );

  return (
    <div style={{
      border: '1px solid rgba(201,162,39,0.35)',
      borderRadius: 10,
      marginBottom: 14,
      overflow: 'hidden',
      background: 'rgba(0,0,0,0.18)',
    }}>
      <div
        style={{
          display: 'flex', alignItems: 'center', gap: 12,
          padding: '10px 16px',
          background: 'rgba(201,162,39,0.07)',
          borderBottom: '1px solid rgba(201,162,39,0.15)',
          cursor: genes.length > 5 ? 'pointer' : 'default',
        }}
        onClick={() => genes.length > 5 && setExpanded(e => !e)}
      >
        <span style={{
          fontSize: 11, fontWeight: 800, letterSpacing: '0.08em',
          padding: '2px 8px', borderRadius: 4,
          background: badgeStyle.bg, border: `1px solid ${badgeStyle.border}`,
          color: badgeStyle.color, flexShrink: 0,
        }}>
          {sv.sv_type || 'SV'}
        </span>

        <code style={{ fontSize: 12, color: 'var(--cream)', opacity: 0.9, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {sv.sv_id}
        </code>

        <div style={{ display: 'flex', gap: 6, flexShrink: 0 }}>
          <span style={{ fontSize: 11, padding: '1px 7px', borderRadius: 4, background: '#5a0e1e', color: '#f6225a', border: '1px solid #f6225a44' }}>
            {sv.num_affected} затронуто
          </span>
          {sv.num_uncertain > 0 && (
            <span style={{ fontSize: 11, padding: '1px 7px', borderRadius: 4, background: '#3a2800', color: '#c9a227', border: '1px solid #c9a22744' }}>
              {sv.num_uncertain} неопр.
            </span>
          )}
          <span style={{ fontSize: 11, padding: '1px 7px', borderRadius: 4, background: '#0e4a20', color: '#2ff849', border: '1px solid #2ff84944' }}>
            {sv.num_neutral} нейтр.
          </span>
        </div>

        {genes.length > 5 && (
          <i className={expanded ? 'bi bi-chevron-up' : 'bi bi-chevron-down'} style={{ fontSize: 14, opacity: 0.5, flexShrink: 0 }} aria-hidden />
        )}
      </div>

      <div style={{ padding: '8px 16px 4px', fontSize: 12, opacity: 0.85 }}>
        Всего генов: <strong>{sv.num_genes}</strong>.&nbsp;
        {summaryParts.reduce((acc, el, i) => i === 0 ? [el] : [...acc, <span key={`s${i}`}>, </span>, el], [])}
        {summaryParts.length > 0 && '.'}
      </div>

      <div style={{ padding: '8px 16px 12px' }}>
        {displayGenes.map((g, idx) => (
          <GeneRow key={g.gene_name + idx} gene={g} rank={idx + 1} />
        ))}
        {!expanded && hidden > 0 && (
          <button
            onClick={() => setExpanded(true)}
            style={{
              background: 'none', border: '1px dashed rgba(201,162,39,0.4)',
              color: 'var(--gold, #c9a227)', borderRadius: 6, padding: '5px 14px',
              cursor: 'pointer', fontSize: 12, width: '100%', marginTop: 4,
            }}
          >
            + ещё {hidden} ген{hidden > 4 ? 'ов' : hidden > 1 ? 'а' : ''}
          </button>
        )}
      </div>
    </div>
  );
}

export default function BiologicalStatsPanel({ svStatistics = [] }) {
  const [search, setSearch] = useState('');
  const [filterType, setFilterType] = useState('all');

  if (!svStatistics || svStatistics.length === 0) return null;

  let filtered = svStatistics;
  if (search.trim()) {
    const q = search.trim().toLowerCase();
    filtered = filtered.filter(sv =>
      sv.sv_id.toLowerCase().includes(q) ||
      (sv.sv_type || '').toLowerCase().includes(q) ||
      sv.gene_details.some(g => g.gene_name.toLowerCase().includes(q))
    );
  }
  if (filterType === 'affected')  filtered = filtered.filter(s => s.num_affected > 0);
  if (filterType === 'uncertain') filtered = filtered.filter(s => s.num_uncertain > 0);
  if (filterType === 'neutral')   filtered = filtered.filter(s => s.num_neutral > 0 && s.num_affected === 0);

  return (
    <div className="sv-panel" style={{ marginBottom: 20 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16, flexWrap: 'wrap', gap: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <i className="bi bi-diagram-3" style={{ fontSize: '1.25rem', opacity: 0.9 }} aria-hidden />
          <strong style={{ fontSize: 16, letterSpacing: '0.03em' }}>Биологическая интерпретация результатов</strong>
          <span style={{ fontSize: 12, opacity: 0.5, marginLeft: 4 }}>
            ({svStatistics.length} СВ)
          </span>
        </div>

        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {[
            { key: 'all',       label: 'Все', icon: null },
            { key: 'affected',  label: 'Влияют', icon: 'bi-circle-fill', iconColor: '#f6225a' },
            { key: 'uncertain', label: 'Неопред.', icon: 'bi-circle-fill', iconColor: '#c9a227' },
            { key: 'neutral',   label: 'Нейтр.', icon: 'bi-circle-fill', iconColor: '#2ff849' },
          ].map(f => (
            <button
              key={f.key}
              type="button"
              onClick={() => setFilterType(f.key)}
              style={{
                fontSize: 12, padding: '4px 12px', borderRadius: 6, cursor: 'pointer',
                border: filterType === f.key ? '1px solid var(--gold, #c9a227)' : '1px solid rgba(255,255,255,0.15)',
                background: filterType === f.key ? 'rgba(201,162,39,0.15)' : 'transparent',
                color: filterType === f.key ? 'var(--gold, #c9a227)' : 'rgba(255,255,255,0.65)',
                transition: 'all .15s',
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6,
              }}
            >
              {f.icon && <i className={`bi ${f.icon}`} style={{ fontSize: 8, color: f.iconColor }} aria-hidden />}
              {f.label}
            </button>
          ))}
        </div>
      </div>

      <input
        value={search}
        onChange={e => setSearch(e.target.value)}
        placeholder="Поиск по СВ ID или названию гена…"
        style={{
          width: '100%', boxSizing: 'border-box',
          background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(201,162,39,0.3)',
          borderRadius: 6, color: 'var(--cream, #f5f0e8)', padding: '7px 12px',
          fontSize: 13, marginBottom: 14, outline: 'none',
        }}
      />

      <div style={{
        fontSize: 12, lineHeight: 1.6, opacity: 0.65, marginBottom: 14,
        padding: '8px 12px', borderRadius: 6, background: 'rgba(255,255,255,0.04)',
        borderLeft: '3px solid rgba(201,162,39,0.5)',
      }}>
        Гены отсортированы по уверенности модели: наиболее вероятно затронутые — вверху,
        нейтральные с высокой уверенностью — внизу. Неопределённые случаи — посередине.
      </div>

      <div style={{ maxHeight: 300, overflowY: 'auto', borderRadius: 6, marginBottom: 12, paddingRight: 4 }}>
        {filtered.length === 0 ? (
          <div style={{ textAlign: 'center', opacity: 0.45, padding: '24px 0', fontSize: 13 }}>
            Нет СВ, соответствующих фильтру
          </div>
        ) : (
          filtered.map((sv, i) => <SVCard key={sv.sv_id + i} sv={sv} />)
        )}
      </div>

      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginTop: 8, paddingTop: 12, borderTop: '1px solid rgba(255,255,255,0.08)' }}>
        <LegendPill color="#f6225a" label="Ген затронут (высокая уверенность)" />
        <LegendPill color="#d06080" label="Ген затронут (низкая уверенность)" />
        <LegendPill color="#2ff849" label="Ген не затронут (высокая уверенность)" />
        <LegendPill color="#3a9455" label="Ген не затронут (низкая уверенность)" />
        <LegendPill color="#c9a227" label="Неопределённо" />
      </div>
    </div>
  );
}
