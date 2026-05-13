import React, { useEffect, useState, useRef, useCallback } from 'react';
import {
  LineChart, Line, CartesianGrid, XAxis, YAxis,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import api from '../api';
import { exportAll } from '../utils/exportUtils';
import { useTaskContext } from '../TaskContext';
import { experimentKindLabel, taskStatusLabel } from '../uiLabels';

const MODE_INFO = {
  train:     { label: 'Обучение с нуля',                         color: 'green' },
  fine_tune: { label: 'Дообучение (датасет различается ≤ 50%)',  color: 'gold' },
  retrain:   { label: 'Переобучение (датасет различается > 10%)', color: 'blue' },
};

function MetricCard({ label, value }) {
  return (
    <div style={{ flex: '1 1 120px' }}>
      <div className="sv-metric">
        <div className="sv-metric-label">{label}</div>
        <div className="sv-metric-value">{value != null ? Number(value).toFixed(4) : '—'}</div>
      </div>
    </div>
  );
}

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
      <td><code style={{ color: deltaColor }}>
        {isImprovement ? '+' : ''}{Number(delta).toFixed(4)} ({isImprovement ? '+' : ''}{Number(improvementPct).toFixed(1)}%)
      </code></td>
    </tr>
  );
}

export default function TrainingPage() {
  const { startStream, tasks, epochData, updateTask, datasetsVersion } = useTaskContext();

  const [datasets,       setDatasets]       = useState([]);
  const [models,         setModels]         = useState([]);
  const [datasetId,      setDatasetId]      = useState('');
  const [modelName,      setModelName]      = useState('sv_classifier');
  const [mode,           setMode]           = useState('train');
  const [epochs,         setEpochs]         = useState(20);
  const [batchSize,      setBatchSize]      = useState(64);
  const [lr,             setLr]             = useState(0.001);
  const [currentTaskId,  setCurrentTaskId]  = useState(null);
  const [experimentId,   setExperimentId]   = useState(null);
  const [finalMetrics,   setFinalMetrics]   = useState(null);
  const [exportError,    setExportError]    = useState('');
  const [exporting,      setExporting]      = useState(false);
  const [modelHistory,   setModelHistory]   = useState([]);
  const [parentExpId,    setParentExpId]    = useState(null);
  const [useEarlyStopping, setUseEarlyStopping] = useState(false);
  const [earlyStoppingPatience, setEarlyStoppingPatience] = useState(5);
  const [deletingModelId, setDeletingModelId] = useState(null);
  const logsEndRef = useRef(null);

  const refresh = useCallback(() => {
    api.listDatasets().then(r => setDatasets(r.data)).catch(console.error);
    api.listModels().then(r => setModels(r.data)).catch(console.error);
  }, []);

  useEffect(() => {
    if (mode !== 'train' && models.length > 0) {
      const names = models.map(m => m.name);
      if (!names.includes(modelName)) setModelName(models[0].name);
    }
    if (mode !== 'train' && models.length === 0) {
      setModelName('');
    }
  }, [mode, models]); // eslint-disable-line

  useEffect(() => { refresh(); }, [refresh, datasetsVersion]);

  useEffect(() => {
    const selectedModel = models.find(m => m.name === modelName);
    if (selectedModel && (mode === 'fine_tune' || mode === 'retrain')) {
      api.getModelTrainingHistory(selectedModel.id)
        .then(r => {
          setModelHistory(r.data);
          setParentExpId(null);
        })
        .catch(e => {
          console.error(e);
          setModelHistory([]);
        });
    } else {
      setModelHistory([]);
      setParentExpId(null);
    }
  }, [mode, modelName, models]);

  const normalizeMetrics = (metrics) => {
    if (!Array.isArray(metrics)) return [];
    return metrics.map(m => ({
      ...m,
      epoch: m.epoch != null ? Number(m.epoch) : m.epoch,
      epochs: m.epochs != null ? Number(m.epochs) : m.epochs,
      train_loss: m.train_loss != null ? Number(m.train_loss) : null,
      val_loss: m.val_loss != null ? Number(m.val_loss) : null,
      val_acc: m.val_acc != null ? Number(m.val_acc) : null,
      val_auc: m.val_auc != null ? Number(m.val_auc) : null,
      val_f1: m.val_f1 != null ? Number(m.val_f1) : null,
      val_precision: m.val_precision != null ? Number(m.val_precision) : null,
      val_recall: m.val_recall != null ? Number(m.val_recall) : null,
    }));
  };

  const parseExperimentMetrics = (raw) => {
    if (!raw) return [];
    if (Array.isArray(raw)) return raw;
    if (typeof raw === 'string') {
      try { return parseExperimentMetrics(JSON.parse(raw)); } catch { return []; }
    }
    if (typeof raw === 'object') {
      if (raw.epoch !== undefined) return [raw];
      const vals = Object.values(raw).filter(v => typeof v === 'object');
      return vals.sort((a, b) => (a.epoch ?? 0) - (b.epoch ?? 0));
    }
    return [];
  };

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [tasks[currentTaskId]?.logs]);

  const task        = currentTaskId ? tasks[currentTaskId] : null;
  const liveMetrics = (currentTaskId ? epochData[currentTaskId] : null) || [];
  const chartData   = finalMetrics || liveMetrics;
  const isActive    = task && ['pending', 'queued', 'running'].includes(task.status);
  const isCompleted = task?.status === 'completed';
  const lastEpoch   = chartData.length > 0 ? chartData[chartData.length - 1] : null;

  const onStart = async () => {
    if (!datasetId) return;
    if ((mode === 'fine_tune' || mode === 'retrain') && !parentExpId) {
      alert(`Для режима ${MODE_INFO[mode]?.label} выберите базовый эксперимент`);
      return;
    }
    setCurrentTaskId(null);
    setFinalMetrics(null);
    setExportError('');

    const payload = { 
      dataset_id: Number(datasetId), 
      model_name: modelName, 
      mode, 
      epochs, 
      batch_size: batchSize, 
      lr,
      parent_experiment_id: parentExpId,
      use_early_stopping: useEarlyStopping,
      early_stopping_patience: earlyStoppingPatience,
    };
    try {
      const res = await api.startExperiment(payload);
      const { task_id, experiment_id } = res.data;
      setCurrentTaskId(task_id);
      setExperimentId(experiment_id);

      updateTask(task_id, { task_id, experiment_id, status: 'queued', progress: 0, message: 'В очереди на обучение…', logs: [], kind: mode });

      startStream(task_id, {
        onComplete: async (data) => {
          if (data.status === 'completed' && experiment_id) {
            try {
              const exp = await api.getExperiment(experiment_id);
              const parsed = parseExperimentMetrics(exp.data.metrics);
              if (parsed.length) {
                setFinalMetrics(normalizeMetrics(parsed));
              } else if (exp.data.result_data?.test_metrics) {
                const t = exp.data.result_data.test_metrics;
                setFinalMetrics(normalizeMetrics([{
                  epoch: 1, epochs: 1,
                  train_loss: t.train_loss ?? null,
                  val_loss: t.val_loss ?? null,
                  val_acc: t.accuracy ?? t.acc ?? null,
                  val_auc: t.auc ?? null,
                  val_f1: t.f1 ?? null,
                  val_precision: t.precision ?? null,
                  val_recall: t.recall ?? null,
                }]));
              }
              
              if (exp.data.comparison_with_parent) {
                updateTask(task_id, { comparison_data: exp.data.comparison_with_parent });
              }
            } catch (e) { console.error('getExperiment:', e); }
          }
        },
      });
    } catch (err) { console.error('startExperiment:', err); }
  };

  const handleExport = async () => {
    if (!experimentId || exporting) return;
    setExportError('');
    setExporting(true);
    try {
      await exportAll(experimentId, mode, (msg) => setExportError(msg));
    } finally {
      setExporting(false);
    }
  };

  const onDeleteModel = async (m) => {
    if (!window.confirm(`Удалить модель «${m.name}»? Файл весов будет удалён; эксперименты останутся в архиве без привязки к модели (model_id станет пустым).`)) return;
    setDeletingModelId(m.id);
    try {
      await api.deleteModel(m.id);
      await refresh();
    } catch (err) {
      const msg = err.response?.data?.error || err.message || 'Не удалось удалить модель';
      window.alert(msg);
    } finally {
      setDeletingModelId(null);
    }
  };

  const chartLineProps = { dot: false, strokeWidth: 2, isAnimationActive: false };

  return (
    <div>
      <h2 className="sv-page-title">Обучение модели</h2>

      <div className="sv-panel">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 32 }}>
          <div>
            <div style={{ marginBottom: 16 }}>
              <label className="sv-label">Датасет</label>
              <select className="sv-select" value={datasetId} onChange={e => setDatasetId(e.target.value)} disabled={isActive}>
                <option value="">Выберите датасет…</option>
                {datasets.map(d => (
                  <option key={d.id} value={d.id}>{d.name} ({(d.row_count || 0).toLocaleString()} строк)</option>
                ))}
              </select>
            </div>

            <div style={{ marginBottom: 16 }}>
              <label className="sv-label">Название модели</label>
              {mode === 'train' ? (
                <input className="sv-input" value={modelName} onChange={e => setModelName(e.target.value)} disabled={isActive} placeholder="sv_classifier" />
              ) : (
                <select className="sv-select" value={modelName} onChange={e => setModelName(e.target.value)} disabled={isActive || models.length === 0}>
                  <option value="">Выберите модель…</option>
                  {models.map(m => <option key={m.id} value={m.name}>{m.name} (id:{m.id})</option>)}
                </select>
              )}
            </div>

            <div style={{ marginBottom: 16 }}>
              <label className="sv-label">Режим</label>
              <select className="sv-select" value={mode} onChange={e => setMode(e.target.value)} disabled={isActive}>
                {Object.entries(MODE_INFO).map(([k, v]) => (
                  <option key={k} value={k}>{v.label}</option>
                ))}
              </select>
            </div>

            {(mode === 'fine_tune' || mode === 'retrain') && modelHistory.length > 0 && (
              <div style={{ marginBottom: 16 }}>
                <label className="sv-label">Базовый эксперимент</label>
                <select className="sv-select" value={parentExpId || ''} onChange={e => setParentExpId(e.target.value ? Number(e.target.value) : null)} disabled={isActive}>
                  <option value="">Выберите базовый эксперимент…</option>
                  {modelHistory.map(exp => (
                    <option key={exp.id} value={exp.id}>
                      {experimentKindLabel(exp.kind)} {exp.model_version_at_time ? `(v${exp.model_version_at_time})` : ''} ({new Date(exp.created_at).toLocaleString()})
                    </option>
                  ))}
                </select>
              </div>
            )}

            {models.length > 0 && (
              <div style={{ marginTop: 4 }}>
                <div style={{ fontSize: 13, opacity: 0.65, marginBottom: 8 }}>Сохранённые модели</div>
                <div className="sv-models-scroll">
                  <div className="sv-models-list">
                    {models.map(m => (
                      <span key={m.id} className="sv-model-chip sv-badge sv-badge-gold">
                        <span className="sv-model-chip-name" title={m.name}>{m.name}</span>
                        <button
                          type="button"
                          className="sv-model-chip-remove"
                          onClick={() => onDeleteModel(m)}
                          disabled={isActive || deletingModelId === m.id}
                          aria-label={`Удалить модель ${m.name}`}
                        >
                          {deletingModelId === m.id ? <span className="sv-spinner" style={{ width: 12, height: 12 }} /> : <i className="bi bi-x-lg" aria-hidden />}
                        </button>
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          <div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
              <div>
                <label className="sv-label" style={{ fontSize: 12 }}>Эпохи</label>
                <input className="sv-input" type="number" min="1" value={epochs} onChange={e => setEpochs(Number(e.target.value))} disabled={isActive} />
              </div>
              <div>
                <label className="sv-label" style={{ fontSize: 12 }}>Размер батча</label>
                <input className="sv-input" type="number" min="1" value={batchSize} onChange={e => setBatchSize(Number(e.target.value))} disabled={isActive} />
              </div>
            </div>
            <div style={{ marginBottom: 16 }}>
              <label className="sv-label" style={{ fontSize: 12 }}>Скорость обучения</label>
              <input className="sv-input" type="number" step="0.00001" value={lr} onChange={e => setLr(Number(e.target.value))} disabled={isActive} />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16, alignItems: 'start' }}>
              <label className="sv-checkbox-label">
                <input type="checkbox" checked={useEarlyStopping} onChange={e => setUseEarlyStopping(e.target.checked)} disabled={isActive} />
                Ранняя остановка
              </label>
              <div>
                <label className="sv-label" style={{ fontSize: 12 }}>Patience</label>
                <input className="sv-input" type="number" min="1" value={earlyStoppingPatience} onChange={e => setEarlyStoppingPatience(Number(e.target.value))} disabled={!useEarlyStopping || isActive} />
              </div>
            </div>

            <button className="sv-btn sv-btn-green" style={{ width: '100%', justifyContent: 'center' }} onClick={onStart} disabled={!datasetId || ((mode === 'fine_tune' || mode === 'retrain') && !parentExpId) || isActive}>
              {isActive
                ? `${MODE_INFO[mode]?.label || 'Обучение'}… (${task?.progress ?? 0}%)`
                : (MODE_INFO[mode]?.label || 'Запустить')}
            </button>
          </div>
        </div>
      </div>

      {task && (
        <div className="sv-panel" style={{ borderColor: task.status === 'completed' ? 'var(--green)' : task.status === 'failed' ? 'var(--red-btn)' : 'var(--gold)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <strong>Статус обучения</strong>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              {isActive && <span className="sv-spinner" />}
              <span className={`sv-badge sv-badge-${task.status === 'completed' ? 'success' : task.status === 'failed' ? 'red' : task.status === 'running' ? 'blue' : 'dim'}`}>
                {taskStatusLabel(task.status)}
              </span>

              {isCompleted && experimentId && (
                <button
                  className="sv-btn sv-btn-ghost"
                  style={{ fontSize: 12, padding: '5px 14px', display: 'flex', alignItems: 'center', gap: 6 }}
                  onClick={handleExport}
                  disabled={exporting}
                >
                  {exporting ? <><span className="sv-spinner" style={{ width: 12, height: 12 }} /> Экспорт…</> : <><i className="bi bi-download" aria-hidden /> Экспорт PDF</>}
                </button>
              )}
            </div>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, opacity: 0.8, marginBottom: 6 }}>
            <span>{task.message}</span><span>{task.progress ?? 0}%</span>
          </div>
          <div className="sv-progress-track" style={{ marginBottom: 14 }}>
            <div className={`sv-progress-bar ${task.status === 'failed' ? 'danger' : task.status === 'completed' ? 'success' : ''}`} style={{ width: `${task.progress ?? 0}%` }} />
          </div>
          {exportError && (
            <div style={{ fontSize: 12, color: 'var(--red-btn)', marginBottom: 8 }}>{exportError}</div>
          )}
          {(task.logs?.length ?? 0) > 0 && (
            <div>
              <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 6, opacity: 0.7 }}>Журнал</div>
              <div className="sv-logs">
                {task.logs.map((l, i) => <div key={i}>{l}</div>)}
                <div ref={logsEndRef} />
              </div>
            </div>
          )}
        </div>
      )}

      {chartData.length > 0 && (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
            <h3 style={{ color: 'var(--cream)', margin: 0, fontSize: 20, display: 'flex', alignItems: 'center', gap: 8 }}>
              <i className="bi bi-graph-up-arrow" aria-hidden />
              Метрики обучения
            </h3>
            {isActive && <span className="sv-badge sv-badge-blue"><i className="bi bi-broadcast me-1" aria-hidden />Онлайн — epoch {lastEpoch?.epoch}/{lastEpoch?.epochs}</span>}
            {isCompleted && <span className="sv-badge sv-badge-success">Финальные результаты</span>}
          </div>

          {lastEpoch && (
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
              <MetricCard label="Train Loss"  value={lastEpoch.train_loss} />
              <MetricCard label="Val Loss"    value={lastEpoch.val_loss} />
              <MetricCard label="Accuracy"    value={lastEpoch.val_acc} />
              <MetricCard label="AUC-ROC"     value={lastEpoch.val_auc} />
              {lastEpoch.val_f1 != null       && <MetricCard label="F1"        value={lastEpoch.val_f1} />}
              {lastEpoch.val_precision != null && <MetricCard label="Precision" value={lastEpoch.val_precision} />}
              {lastEpoch.val_recall != null   && <MetricCard label="Recall"    value={lastEpoch.val_recall} />}
            </div>
          )}

          {isCompleted && experimentId && (
            <div style={{ marginBottom: 20 }}>
              <button 
                className="sv-btn sv-btn-ghost"
                style={{ fontSize: 12, padding: '8px 14px', marginBottom: 12 }}
                onClick={async () => {
                  try {
                    const exp = await api.getExperiment(experimentId);
                    if (exp.data.comparison_with_parent) {
                      document.getElementById('comparison-table')?.scrollIntoView({ behavior: 'smooth' });
                    } else {
                      alert('Нет данных для сравнения. Это может быть первое обучение модели.');
                    }
                  } catch (e) {
                    console.error(e);
                  }
                }}
              >
                <i className="bi bi-arrow-left-right me-1" aria-hidden />
                Сравнить с предыдущей версией
              </button>
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
            <div className="sv-panel">
              <div style={{ textAlign: 'center', marginBottom: 12, fontWeight: 700 }}>Loss <small style={{ opacity: 0.6, fontSize: 12 }}>(ниже — лучше)</small></div>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" />
                  <YAxis />
                  <Tooltip formatter={(v) => v?.toFixed(4)} labelFormatter={v => `Epoch ${v}`} />
                  <Legend />
                  <Line dataKey="train_loss" stroke="#F2B705" name="Train" {...chartLineProps} />
                  <Line dataKey="val_loss"   stroke="#2FF849" name="Val"   {...chartLineProps} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="sv-panel">
              <div style={{ textAlign: 'center', marginBottom: 12, fontWeight: 700 }}>Точность и AUC <small style={{ opacity: 0.6, fontSize: 12 }}>(выше — лучше)</small></div>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" />
                  <YAxis domain={[0, 1]} />
                  <Tooltip formatter={(v) => v?.toFixed(4)} labelFormatter={v => `Epoch ${v}`} />
                  <Legend />
                  <Line dataKey="val_acc" stroke="#F2B705" name="Accuracy" {...chartLineProps} />
                  <Line dataKey="val_auc" stroke="#2FF849" name="AUC"      {...chartLineProps} />
                  {lastEpoch?.val_f1 != null && <Line dataKey="val_f1" stroke="#2B72FB" name="F1" {...chartLineProps} />}
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
                    {chartData.some(r => r.val_f1 != null) && <th>F1</th>}
                  </tr>
                </thead>
                <tbody>
                  {[...chartData].reverse().map((m, i) => (
                    <tr key={i}>
                      <td><code style={{ color: 'var(--gold)' }}>{m.epoch}/{m.epochs}</code></td>
                      <td><code>{m.train_loss?.toFixed(4)}</code></td>
                      <td><code>{m.val_loss?.toFixed(4)}</code></td>
                      <td><code>{m.val_acc?.toFixed(4)}</code></td>
                      <td><code>{m.val_auc?.toFixed(4)}</code></td>
                      {chartData.some(r => r.val_f1 != null) && <td><code>{m.val_f1?.toFixed(4) ?? '—'}</code></td>}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {isCompleted && experimentId && (
            <div id="comparison-table" style={{ marginTop: 20 }}>
              {task?.comparison_data && (
                <div className="sv-panel">
                  <div className="sv-panel-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <i className="bi bi-bar-chart-line" aria-hidden />
                    Сравнение метрик с базовой версией
                  </div>
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
                        <ComparisonMetricRow metric="accuracy" comparison={task?.comparison_data} />
                        <ComparisonMetricRow metric="precision" comparison={task?.comparison_data} />
                        <ComparisonMetricRow metric="recall" comparison={task?.comparison_data} />
                        <ComparisonMetricRow metric="f1" comparison={task?.comparison_data} />
                        <ComparisonMetricRow metric="auc" comparison={task?.comparison_data} />
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {!task && chartData.length === 0 && (
        <div className="sv-panel sv-empty">
          Выберите датасет и нажмите <strong>Запустить обучение</strong> для старта.
        </div>
      )}
    </div>
  );
}
