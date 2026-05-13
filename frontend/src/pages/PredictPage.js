import React, { useEffect, useState, useRef, useCallback } from 'react';
import {
  LineChart, Line, PieChart, Pie, Cell,
  CartesianGrid, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import api from '../api';
import { exportAll } from '../utils/exportUtils';
import { useTaskContext } from '../TaskContext';
import BiologicalStatsPanel from '../components/BiologicalStatsPanel';
import { taskStatusLabel } from '../uiLabels';

const PIE_COLORS = ['#F6225A', '#2B72FB'];

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

function MetricCard({ label, value, fmt = (v) => Number(v).toFixed(4) }) {
  return (
    <div style={{ flex: '1 1 120px' }}>
      <div className="sv-metric">
        <div className="sv-metric-label">{label}</div>
        <div className="sv-metric-value">{value != null ? fmt(value) : '—'}</div>
      </div>
    </div>
  );
}

export default function PredictPage() {
  const { startStream, tasks, updateTask } = useTaskContext();

  const [datasets,      setDatasets]      = useState([]);
  const [models,        setModels]        = useState([]);
  const [datasetId,     setDatasetId]     = useState('');
  const [modelId,       setModelId]       = useState('');
  const [currentTaskId, setCurrentTaskId] = useState(null);
  const [experimentId,  setExperimentId]  = useState(null);
  const [result,        setResult]        = useState(null);
  const [graphs,        setGraphs]        = useState(null);
  const [predictions,   setPredictions]   = useState([]);
  const [predPage,      setPredPage]      = useState(1);
  const [predTotal,     setPredTotal]     = useState(0);
  const [loadingPreds,  setLoadingPreds]  = useState(false);
  const [exportError,   setExportError]   = useState('');
  const [exporting,     setExporting]     = useState(false);

  const logsEndRef = useRef(null);

  const refresh = useCallback(() => {
    api.listDatasets().then(r => setDatasets(r.data)).catch(console.error);
    api.listModels().then(r => setModels(r.data)).catch(console.error);
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const visibleDatasets = modelId
    ? datasets.filter(d => !(d.used_by_model_ids || []).includes(Number(modelId)))
    : datasets;

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [tasks[currentTaskId]?.logs]);

  useEffect(() => {
    if (!modelId) return;
    const d = datasets.find(x => String(x.id) === String(datasetId));
    if (d && (d.used_by_model_ids || []).includes(Number(modelId))) setDatasetId('');
  }, [modelId, datasets]); // eslint-disable-line

  const task     = currentTaskId ? tasks[currentTaskId] : null;
  const isActive = task && ['pending', 'queued', 'running'].includes(task.status);
  const isDone   = task?.status === 'completed';
  const hasTrueLabel = predictions.some(p => p?.true_label != null);
  const hasExplanation = predictions.some(p => p?.explanation != null && p.explanation !== '');

  const loadPredictions = useCallback(async (expId, page = 1) => {
    if (!expId) return;
    setLoadingPreds(true);
    setExportError('');

    try {
      const res = await api.getExperimentPredictions(expId, page, 50);
      let data = res.data;

      if (typeof data === 'string') {
        try {
          data = JSON.parse(data);
        } catch (e) {
          console.error(e);
          throw e;
        }
      }

      const preds = data.predictions || data || [];
      const total = data.total ?? preds.length;

      setPredictions(preds);
      setPredTotal(total);
      setPredPage(page);
    } catch (e) {
      console.error(e);
      if (e.response?.status === 404) {
        setExportError('Файл с предсказаниями ещё не готов или не найден');
      } else {
        setExportError('Ошибка при загрузке детальных предсказаний');
      }
      setPredictions([]);
    } finally {
      setLoadingPreds(false);
    }
  }, []);

  const onPredict = async () => {
    if (!datasetId || !modelId) return;
    setCurrentTaskId(null); setResult(null); setGraphs(null);
    setPredictions([]); setExportError('');

    const payload = { dataset_id: Number(datasetId), model_id: Number(modelId) };
    try {
      const res = await api.startPrediction(payload);
      const { task_id, experiment_id } = res.data;
      setCurrentTaskId(task_id); setExperimentId(experiment_id);

      updateTask(task_id, { task_id, experiment_id, status: 'queued', progress: 0, message: 'В очереди…', logs: [], kind: 'predict' });

      startStream(task_id, {
        onComplete: async (data) => {
          if (data.status === 'completed' && experiment_id) {
            try {
              const [expRes, graphRes] = await Promise.all([
                api.getExperiment(experiment_id),
                api.getExperimentGraphs(experiment_id),
              ]);
              setResult(expRes.data.result_data || null);
              setGraphs(graphRes.data || null);
              loadPredictions(experiment_id, 1);
            } catch (e) { console.error(e); }
          }
        },
      });
    } catch (err) { console.error(err); }
  };

  const handleExport = async () => {
    if (!experimentId || exporting) return;
    setExportError('');
    setExporting(true);
    try {
      await exportAll(experimentId, 'predict', (msg) => setExportError(msg));
    } finally {
      setExporting(false);
    }
  };

  const pieData = result
    ? [{ name: 'Affected (1)', value: result.pred_1 || 0 }, { name: 'Neutral (0)', value: result.pred_0 || 0 }]
    : [];

  return (
    <div>
      <h2 className="sv-page-title">Предсказание</h2>

      <div className="sv-panel">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr auto', gap: 20, alignItems: 'end' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: 8 }}>
            <div>
              <label className="sv-label">Датасет</label>
              <select className="sv-select" value={datasetId} onChange={e => setDatasetId(e.target.value)} disabled={isActive}>
                <option value="">Выберите датасет…</option>
                {visibleDatasets.map(d => (
                  <option key={d.id} value={d.id}>{d.name} ({(d.row_count || 0).toLocaleString()} строк)</option>
                ))}
              </select>
            </div>
            <div style={{ alignSelf: 'end' }}>
              <button
                className="sv-btn sv-btn-ghost"
                style={{ fontSize: 12, padding: '8px 12px' }}
                onClick={refresh}
                disabled={isActive}
              ><i className="bi bi-arrow-clockwise me-1" aria-hidden />Обновить</button>
            </div>
          </div>
          <div>
            <label className="sv-label">Модель</label>
            <select className="sv-select" value={modelId} onChange={e => setModelId(e.target.value)} disabled={isActive}>
              <option value="">Выберите модель…</option>
              {models.map(m => (
                <option key={m.id} value={m.id}>{m.name}{m.accuracy ? ` (Accuracy ${Number(m.accuracy).toFixed(3)})` : ''}</option>
              ))}
            </select>
          </div>
          <button className="sv-btn sv-btn-green" onClick={onPredict} disabled={!datasetId || !modelId || isActive}>
            {isActive ? (<><span className="spinner-border spinner-border-sm me-1" role="status" />{task?.progress ?? 0}%</>) : ('Предсказать')}
          </button>
        </div>
      </div>

      {task && (
        <div className="sv-panel" style={{ borderColor: task.status === 'completed' ? 'var(--green)' : task.status === 'failed' ? 'var(--red-btn)' : 'var(--gold)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
            <strong>Статус предсказания</strong>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              {isActive && <span className="sv-spinner" />}
              <span className={`sv-badge sv-badge-${task.status === 'completed' ? 'success' : task.status === 'failed' ? 'red' : task.status === 'running' ? 'blue' : 'dim'}`}>
                {taskStatusLabel(task.status)}
              </span>
              {isDone && experimentId && (
                <button
                  className="sv-btn sv-btn-ghost"
                  style={{ fontSize: 12, padding: '5px 14px', display: 'flex', alignItems: 'center', gap: 6 }}
                  onClick={handleExport}
                  disabled={exporting}
                >
                  {exporting
                    ? <><span className="sv-spinner" style={{ width: 12, height: 12 }} /> Экспорт…</>
                    : <><i className="bi bi-download" aria-hidden /> Экспорт (PDF + CSV + XLSX)</>}
                </button>
              )}
            </div>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, opacity: 0.8, marginBottom: 6 }}>
            <span>{task.message}</span><span>{task.progress ?? 0}%</span>
          </div>
          <div className="sv-progress-track" style={{ marginBottom: 12 }}>
            <div className={`sv-progress-bar ${task.status === 'failed' ? 'danger' : task.status === 'completed' ? 'success' : ''}`} style={{ width: `${task.progress ?? 0}%` }} />
          </div>
          {exportError && (
            <div style={{ fontSize: 12, color: 'var(--red-btn)', marginTop: 6 }}>{exportError}</div>
          )}
          {(task.logs?.length ?? 0) > 0 && (
            <div className="sv-logs" style={{ marginTop: 8 }}>
              {task.logs.map((l, i) => <div key={i}>{l}</div>)}
              <div ref={logsEndRef} />
            </div>
          )}
        </div>
      )}

      {result && (isDone || predictions.length > 0) && (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
            <h3 style={{ color: 'var(--cream)', margin: 0, fontSize: 20, display: 'flex', alignItems: 'center', gap: 8 }}>
              <i className="bi bi-clipboard-data" aria-hidden />
              Результаты предсказания
            </h3>
          </div>

          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
            {result.total   != null && <MetricCard label="Всего СВ"    value={result.total}   fmt={v => Number(v).toLocaleString()} />}
            {result.pred_1  != null && <MetricCard label="Affected"    value={result.pred_1}  fmt={v => Number(v).toLocaleString()} />}
            {result.pred_0  != null && <MetricCard label="Neutral"     value={result.pred_0}  fmt={v => Number(v).toLocaleString()} />}
            {result.accuracy != null && <MetricCard label="Accuracy"   value={result.accuracy} />}
            {result.auc      != null && <MetricCard label="AUC-ROC"    value={result.auc} />}
            {result.f1       != null && <MetricCard label="F1"         value={result.f1} />}
            {result.precision != null && <MetricCard label="Precision" value={result.precision} />}
            {result.recall   != null && <MetricCard label="Recall"     value={result.recall} />}
          </div>

          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: '1fr 2fr', 
            gap: 20, 
            marginBottom: 20 
          }}>
            
            <div className="sv-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
              <div style={{ textAlign: 'center', fontWeight: 700, marginBottom: 10 }}>Распределение</div>
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <ResponsiveContainer width="100%" height={260}>
                  <PieChart>
                    <Pie 
                      data={pieData} 
                      dataKey="value" 
                      nameKey="name" 
                      cx="50%" 
                      cy="50%" 
                      outerRadius={90}
                      label={({ percent }) => `${(percent * 100).toFixed(1)}%`}
                    >
                      {pieData.map((_, i) => <Cell key={i} fill={PIE_COLORS[i]} />)}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="sv-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
              <BiologicalStatsPanel
                svStatistics={result.sv_statistics || []}
                predictions={predictions}
              />
            </div>
          </div>

          {(graphs?.roc_curve || graphs?.pr_curve || result.tp != null) && (
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: '2fr 2fr 1fr', 
              gap: 20, 
              marginBottom: 20 
            }}>
              
              {graphs?.roc_curve && (
                <div className="sv-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                  <div style={{ textAlign: 'center', fontWeight: 700, marginBottom: 10 }}>
                    Кривая ROC {graphs.roc_curve.auc && <small style={{ opacity: 0.7 }}>(AUC={Number(graphs.roc_curve.auc).toFixed(3)})</small>}
                  </div>
                  <div style={{ flex: 1 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={(graphs.roc_curve.fpr || []).map((fpr, i) => ({ fpr: +fpr.toFixed(3), tpr: +((graphs.roc_curve.tpr[i] || 0)).toFixed(3) }))}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="fpr" type="number" domain={[0, 1]} />
                        <YAxis type="number" domain={[0, 1]} />
                        <Tooltip formatter={v => v.toFixed(3)} />
                        <Line type="monotone" dataKey="tpr" stroke="#2FF849" dot={false} strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {graphs?.pr_curve && (
                <div className="sv-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                  <div style={{ textAlign: 'center', fontWeight: 700, marginBottom: 10 }}>Кривая Precision–Recall</div>
                  <div style={{ flex: 1 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={(graphs.pr_curve.recall || []).map((r, i) => ({ recall: +r.toFixed(3), precision: +((graphs.pr_curve.precision[i] || 0)).toFixed(3) }))}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="recall" type="number" domain={[0, 1]} />
                        <YAxis type="number" domain={[0, 1]} />
                        <Tooltip formatter={v => v.toFixed(3)} />
                        <Line type="monotone" dataKey="precision" stroke="#F2B705" dot={false} strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {result.tp != null && (
                <div className="sv-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                  <div style={{ fontWeight: 700, marginBottom: 12, textAlign: 'center' }}>Матрица ошибок</div>
                  <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <table className="sv-table" style={{ width: '100%', maxWidth: 280, fontSize: 14 }}>
                      <thead><tr><th></th><th>Pred 0</th><th>Pred 1</th></tr></thead>
                      <tbody>
                        <tr><td><strong>True 0</strong></td><td style={{ color: 'var(--green)' }}>{result.tn ?? '—'}</td><td style={{ color: 'var(--red-btn)' }}>{result.fp ?? '—'}</td></tr>
                        <tr><td><strong>True 1</strong></td><td style={{ color: 'var(--red-btn)' }}>{result.fn ?? '—'}</td><td style={{ color: 'var(--green)' }}>{result.tp ?? '—'}</td></tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

        </div>
      )}

      {result && (isDone || predictions.length > 0) && (
        <div className="sv-panel">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <strong>
              Предсказания по SV <span style={{ opacity: 0.6, fontSize: 13 }}>(всего {predTotal.toLocaleString()})</span>
            </strong>
            <button
              className="sv-btn sv-btn-ghost"
              style={{ fontSize: 12, padding: '5px 12px', display: 'flex', alignItems: 'center', gap: 6 }}
              onClick={handleExport}
              disabled={exporting}
            >
              {exporting
                ? <><span className="sv-spinner" style={{ width: 12, height: 12 }} /> Экспорт…</>
                : <><i className="bi bi-download" aria-hidden /> Экспорт (PDF + CSV + XLSX)</>}
            </button>
          </div>
          <div className="sv-table-wrap" style={{ maxHeight: 360, overflowY: 'auto' }}>
            <table className="sv-table">
              <thead>
                <tr>
                  <th>#</th><th>SV ID</th><th>Тип</th><th>Ген</th>
                  <th>Предсказание</th>
                  <th>Уверенность</th>
                  {hasTrueLabel && <th>Метка</th>}
                  {hasExplanation && <th>Объяснение</th>}
                  
                </tr>
              </thead>
              <tbody>
                {loadingPreds ? (
                  <tr><td colSpan={10} style={{ textAlign: 'center', padding: '24px 0', opacity: 0.55 }}>
                    <span className="sv-spinner" style={{ marginRight: 8 }} />Загрузка предсказаний…
                  </td></tr>
                ) : predictions.length === 0 ? (
                  <tr><td colSpan={10} style={{ textAlign: 'center', padding: '24px 0', opacity: 0.45 }}>
                    Нет данных
                  </td></tr>
                ) : predictions.map((p, i) => (
                  <tr key={p.id ?? i}>
                    <td style={{ opacity: 0.5, fontSize: 12 }}>{(predPage - 1) * 50 + i + 1}</td>
                    <td><code style={{ fontSize: 12 }}>{p.sv_id ?? p.name ?? `SV_${i + 1}`}</code></td>
                    <td><span className="sv-badge sv-badge-dim">{p.sv_type ?? '—'}</span></td>
                    <td style={{ fontSize: 13, maxWidth: 180 }}> {p.gene ? (
                      <span title={p.gene_name || ''}>{p.gene}</span>
                    ) : p.gene_name ? (
                    <code style={{ fontSize: 13, 
                      maxWidth: 200, 
                      whiteSpace: 'nowrap', 
                      overflow: 'hidden', 
                      textOverflow: 'ellipsis' 
                    }}>{p.gene_name}</code>
                    ) : '—'}
                    </td>
                    <td>
                      {p.prediction === 1 || p.prediction === '1'
                        ? <span className="sv-badge sv-badge-red">Affected (1)</span>
                        : <span className="sv-badge sv-badge-blue">Neutral (0)</span>}
                    </td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div className="sv-progress-track" style={{ flex: 1, height: 8 }}>
                          <div className="sv-progress-bar" style={{
                            width: `${((p.confidence ?? 0) * 100).toFixed(0)}%`, height: '100%',
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
                  onClick={() => loadPredictions(experimentId, predPage - 1)}>← Назад</button>
                <button className="sv-btn sv-btn-ghost" style={{ fontSize: 12, padding: '5px 14px' }}
                  disabled={predPage * 50 >= predTotal || loadingPreds}
                  onClick={() => loadPredictions(experimentId, predPage + 1)}>Вперёд →</button>
              </div>
            </div>
          )}
        </div>
      )}

      {!task && !result && (
        <div className="sv-panel sv-empty">
          Выберите датасет и модель, затем нажмите <strong>Предсказать</strong>.
        </div>
      )}
    </div>
  );
}