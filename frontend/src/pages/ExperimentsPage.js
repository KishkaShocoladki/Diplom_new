import React, { useEffect, useState, useRef, useCallback } from 'react';
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import api from '../api';
import { exportAll } from '../utils/exportUtils';
import { useTaskContext } from '../TaskContext';
import { experimentKindLabel, taskStatusLabel } from '../uiLabels';

const STATUS_BADGE = {
  pending:   'sv-badge-dim',
  queued:    'sv-badge-blue',
  running:   'sv-badge-blue',
  completed: 'sv-badge-success',
  failed:    'sv-badge-red',
};

const KIND_BADGE = {
  train:     'sv-badge-gold',
  fine_tune: 'sv-badge-gold',
  retrain:   'sv-badge-red',
  predict:   'sv-badge-success',
};

function ComparisonMetricRow({ metric, comparison }) {
  if (!comparison) return null;
  const parentVal = comparison[`${metric}_parent`];
  const currentVal = comparison[`${metric}_current`];
  const delta = comparison[`${metric}_delta`];
  const improvementPct = comparison[`${metric}_improvement_percent`];
  if (parentVal === undefined || currentVal === undefined) return null;
  const isImprovement = delta > 0;
  const deltaColor = isImprovement ? 'var(--green)' : 'var(--red-btn)';
  return (
    <tr>
      <td style={{ fontWeight: 500, textTransform: 'uppercase', fontSize: 12 }}>{metric}</td>
      <td><code>{Number(parentVal).toFixed(4)}</code></td>
      <td><code>{Number(currentVal).toFixed(4)}</code></td>
      <td>
        <code style={{ color: deltaColor }}>
          {isImprovement ? '+' : ''}{Number(delta).toFixed(4)} ({isImprovement ? '+' : ''}{Number(improvementPct).toFixed(1)}%)
        </code>
      </td>
    </tr>
  );
}

export default function ExperimentsPage() {
  const [experiments,  setExperiments]  = useState([]);
  const [selectedExp,  setSelectedExp]  = useState(null);
  const [graphs,       setGraphs]       = useState({});
  const [datasetInfo,  setDatasetInfo]  = useState(null);
  const [predictions,  setPredictions]  = useState([]);
  const [predPage,     setPredPage]     = useState(1);
  const [predTotal,    setPredTotal]    = useState(0);
  const [loadingPreds, setLoadingPreds] = useState(false);
  const hasTrueLabel = predictions.some(p => p?.true_label != null);
  const hasExplanation = predictions.some(p => p?.explanation != null && p.explanation !== '');
  const [loading,      setLoading]      = useState(false);
  const [exportError,  setExportError]  = useState('');
  const [exporting,    setExporting]    = useState(false);

  const { startStream, stopStream, tasks, isStreaming, epochData } = useTaskContext();
  const [currentTaskId, setCurrentTaskId] = useState(null);
  const logsEndRef = useRef(null);

  const loadExperiments = async () => {
    const res = await api.listExperiments();
    setExperiments(res.data);
  };

  function ExplanationCell({ text }) {
    const [open, setOpen] = React.useState(false);
    if (!text || text === '—') return <td style={{ opacity: 0.4, fontSize: 12 }}>—</td>;

    const bracketEnd = text.indexOf(']. ');
    const verdict = bracketEnd !== -1 ? text.slice(0, bracketEnd + 1) : text;
    const reasons = bracketEnd !== -1 ? text.slice(bracketEnd + 2) : '';

    const isAffected = text.includes('ВЛИЯЕТ на ген') && !text.includes('НЕ ВЛИЯЕТ');
    const color = isAffected ? 'var(--red-btn)' : 'var(--blue)';

    return (
      <td style={{ maxWidth: 320, fontSize: 11, verticalAlign: 'top' }}>
        <div>
          <span style={{ color, fontWeight: 600 }}>{verdict}</span>
          {reasons && (
            <>
              {' '}
              <button
                onClick={() => setOpen(o => !o)}
                style={{
                  background: 'none', border: 'none', cursor: 'pointer',
                  color: 'var(--gold)', fontSize: 11, padding: 0, textDecoration: 'underline',
                }}
              >
                {open ? (<><i className="bi bi-chevron-up" aria-hidden /> скрыть</>) : (<><i className="bi bi-chevron-down" aria-hidden /> подробнее</>)}
              </button>
              {open && (
                <div style={{
                  marginTop: 6, padding: '8px 10px',
                  background: 'var(--panel-bg, rgba(255,255,255,0.04))',
                  borderLeft: `2px solid ${color}`,
                  borderRadius: 4,
                  lineHeight: 1.55,
                  color: 'var(--text, inherit)',
                  whiteSpace: 'normal',
                }}>
                  {reasons}
                </div>
              )}
            </>
          )}
        </div>
      </td>
    );
  }

  useEffect(() => { loadExperiments(); }, []);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [currentTaskId, tasks[currentTaskId]?.logs]);

  const loadPredictions = useCallback(async (expId, page = 1) => {
    if (!expId) return;
    setLoadingPreds(true);
    try {
      const res = await api.getExperimentPredictions(expId, page, 50);
      setPredictions(res.data.predictions || []);
      setPredTotal(res.data.total || 0);
      setPredPage(page);
    } catch { setPredictions([]); setPredTotal(0); }
    finally { setLoadingPreds(false); }
  }, []);

  const selectExperiment = async (expId) => {
    setLoading(true);
    setExportError('');
    try {
      const exp = await api.getExperiment(expId);
      const data = { ...exp.data };

      const coerceMetrics = (raw) => {
        if (!raw) return [];
        if (Array.isArray(raw)) return raw;
        if (typeof raw === 'string') { try { return JSON.parse(raw); } catch { return []; } }
        if (typeof raw === 'object') {
          if (raw.epoch !== undefined) return [raw];
          return Object.values(raw).filter(v => typeof v === 'object');
        }
        return [];
      };

      data.metrics = coerceMetrics(data.metrics).map(m => {
        const out = { ...m };
        Object.keys(out).forEach(k => {
          const v = out[k];
          if (v === null || v === undefined || typeof v === 'object') return;
          if (typeof v === 'string' && v.trim() !== '' && !Number.isNaN(Number(v))) out[k] = Number(v);
        });
        return out;
      });

      setSelectedExp(data);

      try {
        const found = Object.entries(tasks).find(([, t]) => t && t.experiment_id === data.id);
        if (found) {
          const taskId = found[0];
          if (currentTaskId && currentTaskId !== taskId) stopStream(currentTaskId);
          setCurrentTaskId(taskId);
          if (!isStreaming(taskId)) {
            startStream(taskId, {
              onComplete: async () => {
                try { const fresh = await api.getExperiment(expId); setSelectedExp(fresh.data); } catch {}
              },
            });
          }
        } else {
          if (currentTaskId) { stopStream(currentTaskId); setCurrentTaskId(null); }
        }
      } catch (e) { console.error(e); }

      if (data.dataset_id) {
        try { const ds = await api.getDataset(data.dataset_id); setDatasetInfo(ds.data); }
        catch { setDatasetInfo(null); }
      } else setDatasetInfo(null);

      if (data.status === 'completed') {
        try {
          const gdata = await api.getExperimentGraphs(expId);
          setGraphs(gdata.data || {});
        } catch (e) {
          console.error(e);
          setGraphs({});
        }
      } else {
        setGraphs({});
      }

      if (data.kind === 'predict') {
        loadPredictions(expId, 1);
      } else {
        setPredictions([]);
        setPredTotal(0);
      }
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  useEffect(() => {
    return () => { if (currentTaskId) stopStream(currentTaskId); };
  }, [currentTaskId, stopStream]);

  const task      = currentTaskId ? tasks[currentTaskId] : null;
  const isActive  = task && ['pending', 'queued', 'running'].includes(task.status);
  const formatDate = (d) => new Date(d).toLocaleString();

  const handleExport = async () => {
    if (!selectedExp || exporting) return;
    setExportError('');
    setExporting(true);
    try {
      await exportAll(selectedExp.id, selectedExp.kind, (msg) => setExportError(msg));
    } finally {
      setExporting(false);
    }
  };

  const handleDeleteExperiment = async () => {
    if (!selectedExp) return;
    const confirmed = window.confirm(
      `Удалить эксперимент "${selectedExp.name}"? Это действие невозможно отменить.`
    );
    if (!confirmed) return;

    setLoading(true);
    try {
      await api.deleteExperiment(selectedExp.id);
      await loadExperiments();
      if (currentTaskId) {
        stopStream(currentTaskId);
      }
      setSelectedExp(null);
      setGraphs({});
      setDatasetInfo(null);
      setPredictions([]);
      setPredTotal(0);
      setCurrentTaskId(null);
    } catch (e) {
      console.error(e);
      window.alert('Не удалось удалить эксперимент. Попробуйте снова.');
    } finally {
      setLoading(false);
    }
  };

  const canExport = selectedExp?.status === 'completed';

  return (
    <div>
      <h2 className="sv-page-title">Эксперименты</h2>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 340px) 1fr', gap: 20, alignItems: 'start' }}>
        <div className="sv-panel" style={{ padding: 0, overflow: 'hidden', width: '100%', maxWidth: 340, boxSizing: 'border-box' }}>
          <div style={{ padding: '16px 18px', borderBottom: '2px solid rgba(242,183,5,0.3)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <strong>Список экспериментов</strong>
            <button className="sv-btn sv-btn-ghost" style={{ fontSize: 12, padding: '4px 12px' }}
              onClick={loadExperiments}>
              <i className="bi bi-arrow-clockwise"></i>
            </button>
          </div>
          <div style={{ maxHeight: 'min(580px, 70vh)', overflowY: 'auto', overflowX: 'hidden' }}>
            {experiments.length === 0 ? (
              <div className="sv-empty" style={{ padding: 24 }}>Нет экспериментов</div>
            ) : experiments.map(e => (
              <button
                key={e.id}
                type="button"
                className={`sv-list-item${selectedExp?.id === e.id ? ' selected' : ''}`}
                onClick={() => selectExperiment(e.id)}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4, gap: 8, minWidth: 0 }}>
                  <strong
                    style={{ fontSize: 14, minWidth: 0, flex: '1 1 auto', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                    title={e.name}
                  >{e.name}</strong>
                  <small style={{ opacity: 0.5, fontSize: 11, flexShrink: 0, whiteSpace: 'nowrap' }}>{formatDate(e.created_at).split(',')[0]}</small>
                </div>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  <span className={`sv-badge ${KIND_BADGE[e.kind] || 'sv-badge-dim'}`}>{experimentKindLabel(e.kind)}</span>
                  <span className={`sv-badge ${STATUS_BADGE[e.status] || 'sv-badge-dim'}`}>{taskStatusLabel(e.status)}</span>
                </div>
              </button>
            ))}
          </div>
        </div>

        {selectedExp ? (
          <div>
            {loading && <div className="sv-panel sv-empty">Загрузка…</div>}

            {!loading && (
              <>
                <div className="sv-panel" style={{ marginBottom: 0 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                      <div style={{ fontSize: 17, fontWeight: 700, color: 'var(--cream)', marginBottom: 6 }}>{selectedExp.name}</div>
                      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                        <span className={`sv-badge ${KIND_BADGE[selectedExp.kind] || 'sv-badge-dim'}`}>{experimentKindLabel(selectedExp.kind)}</span>
                        <span className={`sv-badge ${STATUS_BADGE[selectedExp.status] || 'sv-badge-dim'}`}>{taskStatusLabel(selectedExp.status)}</span>
                        {datasetInfo && <span className="sv-badge sv-badge-dim"><i className="bi bi-folder2-open me-1" aria-hidden />{datasetInfo.name}</span>}
                        <span style={{ fontSize: 11, opacity: 0.5, alignSelf: 'center' }}>{formatDate(selectedExp.created_at)}</span>
                      </div>
                    </div>

                    <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
                      {canExport && (
                        <button
                          className="sv-btn sv-btn-ghost"
                          style={{ fontSize: 13, padding: '7px 18px', display: 'flex', alignItems: 'center', gap: 7, flexShrink: 0 }}
                          onClick={handleExport}
                          disabled={exporting}
                        >
                          {exporting
                            ? <><span className="sv-spinner" style={{ width: 13, height: 13 }} /> Экспорт…</>
                            : selectedExp.kind === 'predict'
                              ? <><i className="bi bi-download" aria-hidden /> Экспорт (PDF + CSV + XLSX)</>
                              : <><i className="bi bi-download" aria-hidden /> Экспорт PDF</>}
                        </button>
                      )}

                      <button
                        className="sv-btn sv-btn-ghost"
                        style={{ fontSize: 13, padding: '7px 18px', display: 'flex', alignItems: 'center', gap: 7, flexShrink: 0, color: 'var(--red-btn)', borderColor: 'var(--red-btn)' }}
                        onClick={handleDeleteExperiment}
                        disabled={loading}
                        type="button"
                      >
                        <i className="bi bi-trash" aria-hidden />
                        Удалить
                      </button>
                    </div>
                  </div>

                  {exportError && (
                    <div style={{ fontSize: 12, color: 'var(--red-btn)', marginTop: 10 }}>{exportError}</div>
                  )}
                </div>

                {selectedExp.kind === 'predict' && (
                  <div className="sv-panel" style={{ marginTop: 0 }}>
                    <div style={{ fontWeight: 700, marginBottom: 12 }}>Метрики предсказания</div>
                    <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                      {[
                        ['Всего СВ',   selectedExp.result_data.total,     v => Number(v).toLocaleString()],
                        ['Affected',   selectedExp.result_data.pred_1,    v => Number(v).toLocaleString()],
                        ['Neutral',    selectedExp.result_data.pred_0,    v => Number(v).toLocaleString()],
                        ['Accuracy',   selectedExp.result_data.accuracy,  v => Number(v).toFixed(4)],
                        ['AUC-ROC',    selectedExp.result_data.auc,       v => Number(v).toFixed(4)],
                        ['F1',         selectedExp.result_data.f1,        v => Number(v).toFixed(4)],
                        ['Precision',  selectedExp.result_data.precision, v => Number(v).toFixed(4)],
                        ['Recall',     selectedExp.result_data.recall,    v => Number(v).toFixed(4)],
                      ].filter(([, v]) => v != null).map(([l, v, fmt]) => (
                        <div key={l} className="sv-metric" style={{ flex: '1 1 80px' }}>
                          <div className="sv-metric-label">{l}</div>
                          <div className="sv-metric-value" style={{ fontSize: 16 }}>{(fmt || (x => x))(v)}</div>
                        </div>
                      ))}
                    </div>

                    {graphs.roc_curve && (
                      <div style={{ marginTop: 16 }}>
                        <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 8 }}>
                          Кривая ROC (AUC={graphs.roc_curve.auc?.toFixed(3)})
                        </div>
                        <ResponsiveContainer width="100%" height={200}>
                          <LineChart data={graphs.roc_curve.fpr?.map((fpr, i) => ({ fpr, tpr: graphs.roc_curve.tpr[i] })) || []}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="fpr" type="number" domain={[0, 1]} />
                            <YAxis type="number" domain={[0, 1]} />
                            <Tooltip />
                            <Line type="monotone" dataKey="tpr" stroke="#2FF849" dot={false} strokeWidth={2} isAnimationActive={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    )}

                    {predictions.length > 0 && (
                    <div style={{ marginTop: 14 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                        <strong>
                          Предсказания по SV <span style={{ opacity: 0.6, fontSize: 13 }}>(всего {predTotal.toLocaleString()})</span>
                        </strong>
                        
                      </div>

                      <div className="sv-table-wrap" style={{ maxHeight: 420, overflowY: 'auto' }}>
                        <table className="sv-table">
                          <thead>
                            <tr>
                              <th>#</th>
                              <th>SV ID</th>
                              <th>Тип</th>
                              <th style={{width: 50}}>Ген</th>
                              <th style={{width: 50}}>Предсказание</th>
                              <th>Уверенность</th>
                              {hasTrueLabel && <th>Метка</th>}
                              {hasExplanation && <th>Объяснение</th>}
                            </tr>
                          </thead>
                          <tbody>
                            {loadingPreds ? (
                              <tr><td colSpan={10} style={{ textAlign: 'center', padding: '40px 0', opacity: 0.55 }}>
                                <span className="sv-spinner" style={{ marginRight: 8 }} />Загрузка предсказаний…
                              </td></tr>
                            ) : predictions.length === 0 ? (
                              <tr><td colSpan={10} style={{ textAlign: 'center', padding: '40px 0', opacity: 0.45 }}>
                                Нет данных
                              </td></tr>
                            ) : predictions.map((p, i) => (
                              <tr key={p.id ?? i}>
                                <td style={{ opacity: 0.5, fontSize: 12 }}>{(predPage - 1) * 50 + i + 1}</td>
                                <td><code style={{ fontSize: 12 }}>{p.sv_id ?? p.name ?? `SV_${i + 1}`}</code></td>
                                <td><span className="sv-badge sv-badge-dim">{p.sv_type ?? '—'}</span></td>

                                <td style={{ fontSize: 13, maxWidth: 200 }}>
                                  {p.gene ? (
                                    <span title={p.gene_id || p.gene_name || ''}>{p.gene}</span>
                                  ) : p.gene_name ? (
                                    <code style={{ 
                                      fontSize: 13, 
                                      maxWidth: 200, 
                                      whiteSpace: 'nowrap', 
                                      overflow: 'hidden', 
                                      textOverflow: 'ellipsis',
                                      display: 'inline-block'
                                    }} title={p.gene_name}>
                                      {p.gene_name}
                                    </code>
                                  ) : '—'}
                                </td>

                                <td>
                                  {p.prediction === 1 || p.prediction === '1'
                                    ? <span className="sv-badge sv-badge-red">1</span>
                                    : <span className="sv-badge sv-badge-blue">0</span>}
                                </td>

                                <td>
                                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                    <div className="sv-progress-track" style={{ flex: 1, height: 8 }}>
                                      <div className="sv-progress-bar" style={{
                                        width: `${((p.confidence ?? 0) * 100).toFixed(0)}%`,
                                        background: (p.confidence ?? 0) >= 0.7 ? 'var(--green)' : (p.confidence ?? 0) >= 0.5 ? 'var(--gold)' : 'var(--red-btn)',
                                      }} />
                                    </div>
                                    <small>{((p.confidence ?? 0) * 100).toFixed(1)}%</small>
                                  </div>
                                </td>

                                {hasTrueLabel && (
                                  <td>
                                    {p.true_label === 1 || p.true_label === '1'
                                      ? <span className="sv-badge sv-badge-red">1</span>
                                      : <span className="sv-badge sv-badge-blue">0</span>}
                                  </td>
                                )}

                                {hasExplanation && (
                                  <ExplanationCell text={p.explanation} />
                                )}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>

                      {predTotal > 50 && (
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 14, fontSize: 13 }}>
                          <span style={{ opacity: 0.65 }}>
                            Показано {(predPage - 1) * 50 + 1}–{Math.min(predPage * 50, predTotal)} из {predTotal.toLocaleString()}
                          </span>
                          <div style={{ display: 'flex', gap: 8 }}>
                            <button className="sv-btn sv-btn-ghost" style={{ fontSize: 12, padding: '5px 14px' }}
                              disabled={predPage === 1 || loadingPreds}
                              onClick={() => loadPredictions(selectedExp.id, predPage - 1)}>← Назад</button>
                            <button className="sv-btn sv-btn-ghost" style={{ fontSize: 12, padding: '5px 14px' }}
                              disabled={predPage * 50 >= predTotal || loadingPreds}
                              onClick={() => loadPredictions(selectedExp.id, predPage + 1)}>Вперёд →</button>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                  </div>
                )}

                {selectedExp.result_data?.test_metrics && (
                  <div className="sv-panel" style={{ marginTop: 0 }}>
                    <div style={{ fontWeight: 700, marginBottom: 12 }}>Тестовые метрики</div>
                    {(() => {
                      const t = selectedExp.result_data.test_metrics;
                      return (
                        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                          {[
                            ['Accuracy',  t.accuracy ?? t.acc ?? 0],
                            ['AUC-ROC',   t.auc ?? 0],
                            ['F1',        t.f1 ?? 0],
                            ['Precision', t.precision ?? 0],
                            ['Recall',    t.recall ?? 0],
                          ].map(([l, v]) => (
                            <div key={l} className="sv-metric" style={{ flex: '1 1 80px' }}>
                              <div className="sv-metric-label">{l}</div>
                              <div className="sv-metric-value" style={{ fontSize: 16 }}>{Number(v).toFixed(4)}</div>
                            </div>
                          ))}
                        </div>
                      );
                    })()}

                    {(selectedExp.kind === 'fine_tune' || selectedExp.kind === 'retrain') && selectedExp.comparison_with_parent && (
                      <div style={{ marginTop: 20 }}>
                        <div className="sv-panel">
                          <div className="sv-panel-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <i className="bi bi-bar-chart-line" aria-hidden />
                            Сравнение метрик с базовой версией
                          </div>
                          {selectedExp.parent_experiment_id != null && (
                            <div style={{ fontSize: 12, opacity: 0.75, marginBottom: 12 }}>
                              Базовый эксперимент: №{selectedExp.parent_experiment_id}
                            </div>
                          )}
                          <div style={{ overflowX: 'auto' }}>
                            <table className="sv-table" style={{ width: '100%' }}>
                              <thead>
                                <tr>
                                  <th>Метрика</th>
                                  <th>Базовая версия</th>
                                  <th>Текущая версия</th>
                                  <th>Изменение (Δ%)</th>
                                </tr>
                              </thead>
                              <tbody>
                                <ComparisonMetricRow metric="accuracy" comparison={selectedExp.comparison_with_parent} />
                                <ComparisonMetricRow metric="precision" comparison={selectedExp.comparison_with_parent} />
                                <ComparisonMetricRow metric="recall" comparison={selectedExp.comparison_with_parent} />
                                <ComparisonMetricRow metric="f1" comparison={selectedExp.comparison_with_parent} />
                                <ComparisonMetricRow metric="auc" comparison={selectedExp.comparison_with_parent} />
                              </tbody>
                            </table>
                          </div>
                        </div>
                      </div>
                    )}

                    {selectedExp.metrics && selectedExp.metrics.length > 0 && (
                      <div style={{ marginTop: 20 }}>
                        <h3 style={{ color: 'var(--cream)', margin: '0 0 16px 0', fontSize: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
                          <i className="bi bi-graph-up-arrow" aria-hidden />
                          Метрики обучения
                        </h3>
                        
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
                          <div className="sv-panel">
                            <div style={{ textAlign: 'center', marginBottom: 12, fontWeight: 700 }}>Loss <small style={{ opacity: 0.6, fontSize: 12 }}>(ниже — лучше)</small></div>
                            <ResponsiveContainer width="100%" height={240}>
                              <LineChart data={selectedExp.metrics}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="epoch" />
                                <YAxis />
                                <Tooltip formatter={(v) => v?.toFixed(4)} labelFormatter={v => `Epoch ${v}`} />
                                <Legend />
                                <Line dataKey="train_loss" stroke="#F2B705" name="Train" dot={false} strokeWidth={2} isAnimationActive={false} />
                                <Line dataKey="val_loss" stroke="#2FF849" name="Val" dot={false} strokeWidth={2} isAnimationActive={false} />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                          <div className="sv-panel">
                            <div style={{ textAlign: 'center', marginBottom: 12, fontWeight: 700 }}>Точность и AUC <small style={{ opacity: 0.6, fontSize: 12 }}>(выше — лучше)</small></div>
                            <ResponsiveContainer width="100%" height={240}>
                              <LineChart data={selectedExp.metrics}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="epoch" />
                                <YAxis domain={[0, 1]} />
                                <Tooltip formatter={(v) => v?.toFixed(4)} labelFormatter={v => `Epoch ${v}`} />
                                <Legend />
                                <Line dataKey="val_acc" stroke="#F2B705" name="Accuracy" dot={false} strokeWidth={2} isAnimationActive={false} />
                                <Line dataKey="val_auc" stroke="#2FF849" name="AUC" dot={false} strokeWidth={2} isAnimationActive={false} />
                                {selectedExp.metrics.some(r => r.val_f1 != null) && <Line dataKey="val_f1" stroke="#2B72FB" name="F1" dot={false} strokeWidth={2} isAnimationActive={false} />}
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        </div>

                        <div className="sv-panel">
                          <div className="sv-panel-title">Лог по эпохам</div>
                          <div style={{ maxHeight: 300, overflowY: 'auto' }}>
                            <table className="sv-table">
                              <thead>
                                <tr>
                                  <th>Epoch</th>
                                  <th>Train Loss</th>
                                  <th>Val Loss</th>
                                  <th>Acc</th>
                                  <th>AUC</th>
                                  {selectedExp.metrics.some(r => r.val_f1 != null) && <th>F1</th>}
                                </tr>
                              </thead>
                              <tbody>
                                {[...selectedExp.metrics].reverse().map((m, i) => (
                                  <tr key={i}>
                                    <td><code style={{ color: 'var(--gold)' }}>{m.epoch}/{m.epochs}</code></td>
                                    <td><code>{m.train_loss?.toFixed(4)}</code></td>
                                    <td><code>{m.val_loss?.toFixed(4)}</code></td>
                                    <td><code>{m.val_acc?.toFixed(4)}</code></td>
                                    <td><code>{m.val_auc?.toFixed(4)}</code></td>
                                    {selectedExp.metrics.some(r => r.val_f1 != null) && <td><code>{m.val_f1?.toFixed(4) ?? '—'}</code></td>}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      </div>
                    )}

                    {graphs.roc_curve && (
                      <div style={{ marginTop: 20 }}>
                        <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 8 }}>
                          Кривая ROC (AUC={graphs.roc_curve.auc?.toFixed(3)})
                        </div>
                        <ResponsiveContainer width="100%" height={200}>
                          <LineChart data={graphs.roc_curve.fpr?.map((fpr, i) => ({ fpr, tpr: graphs.roc_curve.tpr[i] })) || []}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="fpr" type="number" domain={[0, 1]} />
                            <YAxis type="number" domain={[0, 1]} />
                            <Tooltip />
                            <Line type="monotone" dataKey="tpr" stroke="#2FF849" dot={false} strokeWidth={2} isAnimationActive={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    )}

                    {graphs.confusion_matrix && (
                      <div style={{ marginTop: 16 }}>
                        <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 8 }}>
                          Матрица ошибок
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2, maxWidth: 200 }}>
                          <div style={{ background: '#f0f0f0', padding: 8, textAlign: 'center', fontSize: 12 }}>TN: {graphs.confusion_matrix.tn}</div>
                          <div style={{ background: '#ffebee', padding: 8, textAlign: 'center', fontSize: 12 }}>FP: {graphs.confusion_matrix.fp}</div>
                          <div style={{ background: '#ffebee', padding: 8, textAlign: 'center', fontSize: 12 }}>FN: {graphs.confusion_matrix.fn}</div>
                          <div style={{ background: '#f0f0f0', padding: 8, textAlign: 'center', fontSize: 12 }}>TP: {graphs.confusion_matrix.tp}</div>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {task && (
                  <div className="sv-panel" style={{ borderColor: task.status === 'completed' ? 'var(--green)' : task.status === 'failed' ? 'var(--red-btn)' : 'var(--gold)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                      <strong>Активная задача</strong>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        {isActive && <span className="sv-spinner" />}
                        <span className={`sv-badge ${task.status === 'completed' ? 'sv-badge-success' : task.status === 'failed' ? 'sv-badge-red' : 'sv-badge-blue'}`}>
                          {taskStatusLabel(task.status)}
                        </span>
                      </div>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, opacity: 0.8, marginBottom: 6 }}>
                      <span>{task.message}</span><span>{task.progress ?? 0}%</span>
                    </div>
                    <div className="sv-progress-track" style={{ marginBottom: 12 }}>
                      <div className={`sv-progress-bar ${task.status === 'failed' ? 'danger' : task.status === 'completed' ? 'success' : ''}`} style={{ width: `${task.progress ?? 0}%` }} />
                    </div>
                    {(task.logs?.length ?? 0) > 0 && (
                      <div className="sv-logs">
                        {task.logs.map((l, i) => <div key={i}>{l}</div>)}
                        <div ref={logsEndRef} />
                      </div>
                    )}

                    {currentTaskId && epochData[currentTaskId]?.length > 0 && (
                      <div style={{ marginTop: 20 }}>
                        <div style={{ color: 'var(--gold)', fontSize: 12, marginBottom: 8, fontWeight: 500, display: 'flex', alignItems: 'center', gap: 6 }}>
                          <i className="bi bi-broadcast" aria-hidden />
                          Онлайн — графики обновляются по мере эпох
                        </div>
                        
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
                          <div className="sv-panel">
                            <div style={{ textAlign: 'center', marginBottom: 12, fontWeight: 700 }}>Loss <small style={{ opacity: 0.6, fontSize: 12 }}>(ниже — лучше)</small></div>
                            <ResponsiveContainer width="100%" height={200}>
                              <LineChart data={epochData[currentTaskId]}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="epoch" />
                                <YAxis />
                                <Tooltip formatter={(v) => v?.toFixed(4)} labelFormatter={v => `Epoch ${v}`} />
                                <Legend />
                                <Line dataKey="train_loss" stroke="#F2B705" name="Train" dot={false} strokeWidth={2} isAnimationActive={false} />
                                <Line dataKey="val_loss" stroke="#2FF849" name="Val" dot={false} strokeWidth={2} isAnimationActive={false} />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                          <div className="sv-panel">
                            <div style={{ textAlign: 'center', marginBottom: 12, fontWeight: 700 }}>Точность и AUC <small style={{ opacity: 0.6, fontSize: 12 }}>(выше — лучше)</small></div>
                            <ResponsiveContainer width="100%" height={200}>
                              <LineChart data={epochData[currentTaskId]}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="epoch" />
                                <YAxis domain={[0, 1]} />
                                <Tooltip formatter={(v) => v?.toFixed(4)} labelFormatter={v => `Epoch ${v}`} />
                                <Legend />
                                <Line dataKey="val_acc" stroke="#F2B705" name="Accuracy" dot={false} strokeWidth={2} isAnimationActive={false} />
                                <Line dataKey="val_auc" stroke="#2FF849" name="AUC" dot={false} strokeWidth={2} isAnimationActive={false} />
                                {epochData[currentTaskId]?.some(r => r.val_f1 != null) && <Line dataKey="val_f1" stroke="#2B72FB" name="F1" dot={false} strokeWidth={2} isAnimationActive={false} />}
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        ) : (
          <div className="sv-panel sv-empty">
            Выберите эксперимент для просмотра деталей
          </div>
        )}
      </div>
    </div>
  );
}
