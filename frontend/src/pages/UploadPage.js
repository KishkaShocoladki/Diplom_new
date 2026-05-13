import React, { useEffect, useState, useCallback, useRef } from 'react';
import api from '../api';
import { useTaskContext } from '../TaskContext';
import { taskStatusLabel } from '../uiLabels';

const toDisplayProgress = (phase, raw) =>
  phase === 'transfer' ? Math.round(raw / 2) : 50 + Math.round(raw / 2);

const statusBarClass = (task) => {
  if (!task) return '';
  if (task.status === 'failed')    return 'danger';
  if (task.status === 'completed') return 'success';
  if (task.status === 'cancelled') return 'warning';
  return '';
};

export default function UploadPage() {
  const { startStream, tasks, updateTask, notifyDatasetsChanged } = useTaskContext();

  const [file, setFile] = useState(null);
  const [name, setName] = useState('');
  const [datasets, setDatasets] = useState([]);
  const [deleting, setDeleting] = useState(null);
  const [error, setError] = useState(null);

  const [currentTaskId, setCurrentTaskId] = useState(() =>
    localStorage.getItem('upload_taskId') || null
  );
  const [phase, setPhase] = useState('idle');
  const [transferPct, setTransferPct] = useState(0);
  const logsEndRef = useRef(null);

  const loadDatasets = useCallback(() => {
    api.listDatasets()
      .then(res => setDatasets(res.data))
      .catch(e => console.error('loadDatasets:', e));
  }, []);

  useEffect(() => { loadDatasets(); }, [loadDatasets]);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [tasks[currentTaskId]?.logs]);

  useEffect(() => {
    if (!currentTaskId) return;
    const existing = tasks[currentTaskId];
    if (existing && ['pending', 'queued', 'running'].includes(existing.status)) {
      setPhase('processing');
    }
  }, [currentTaskId]); // eslint-disable-line

  const task = currentTaskId ? tasks[currentTaskId] : null;
  const isActive = task && ['pending', 'queued', 'running'].includes(task.status);

  const displayProgress =
    phase === 'transfer'
      ? toDisplayProgress('transfer', transferPct)
      : phase === 'processing'
      ? toDisplayProgress('processing', task?.progress ?? 0)
      : task?.status === 'completed' ? 100 : 0;

  const displayMessage =
    phase === 'transfer' ? `Передача файла… ${transferPct}%` : task?.message || '';

  const onUpload = async (e) => {
    e.preventDefault();
    if (!file) return;
    setError(null);
    setTransferPct(0);
    setPhase('transfer');
    const prevId = localStorage.getItem('upload_taskId');
    if (prevId) localStorage.removeItem('upload_taskId');
    setCurrentTaskId(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', name || file.name);

    try {
      const res = await api.uploadDataset(formData, (pct) => setTransferPct(pct));
      const { task_id, dataset_id } = res.data;
      localStorage.setItem('upload_taskId', task_id);
      setCurrentTaskId(task_id);
      setPhase('processing');

      updateTask(task_id, {
        task_id, dataset_id, status: 'queued', progress: 0,
        message: 'Ожидание предобработки…', logs: [], kind: 'upload',
      });

      startStream(task_id, {
        onComplete: (data) => {
          if (data.status === 'completed') {
            loadDatasets();
            notifyDatasetsChanged();
            setTimeout(() => {
              setFile(null); setName(''); setPhase('idle');
              setCurrentTaskId(null);
              localStorage.removeItem('upload_taskId');
            }, 2500);
          } else if (data.status === 'cancelled') {
            loadDatasets();
            notifyDatasetsChanged();
            setPhase('idle'); setCurrentTaskId(null);
            localStorage.removeItem('upload_taskId');
          }
        },
      });
    } catch (err) {
      setPhase('idle');
      setError(err.response?.data?.message || err.response?.data?.error || err.message);
    }
  };

  const onCancel = async () => {
    if (!currentTaskId) return;
    try { await api.cancelTask(currentTaskId); } catch (err) { console.error(err); }
  };

  const onDelete = async (dsId, dsName) => {
    if (!window.confirm(`Удалить датасет "${dsName}" и все связанные файлы?`)) return;
    setDeleting(dsId);
    try { await api.deleteDataset(dsId); loadDatasets(); }
    catch (err) { setError(err.response?.data?.error || err.message); }
    finally { setDeleting(null); }
  };

  const showProgress = phase !== 'idle' || (task && task.status !== undefined);

  return (
    <div>
      <h2 className="sv-page-title">Загрузка датасетов</h2>

      <div className="sv-panel">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
          <div>
            <label className="sv-label">Файл датасета (TSV)</label>
            <input
              type="file"
              className="sv-input"
              accept=".tsv,.txt"
              disabled={!!isActive || phase === 'transfer'}
              onChange={e => setFile(e.target.files?.[0] || null)}
              style={{ padding: '8px 16px', cursor: 'pointer' }}
            />
            {file && (
              <small style={{ color: 'var(--cream)', opacity: 0.6, marginTop: 4, display: 'block' }}>
                {file.name} — {(file.size / 1024 / 1024).toFixed(2)} МБ
              </small>
            )}
          </div>
          <div>
            <label className="sv-label">Название (опционально)</label>
            <input
              className="sv-input"
              value={name}
              onChange={e => setName(e.target.value)}
              disabled={!!isActive || phase === 'transfer'}
              placeholder="Например, Dataset_v1"
            />
          </div>
        </div>
        <div style={{ display: 'flex', gap: 12 }}>
          <button
            className="sv-btn sv-btn-green"
            onClick={onUpload}
            disabled={!file || !!isActive || phase === 'transfer'}
          >{phase === 'transfer' ? (<><i className="bi bi-broadcast me-1"></i>Отправка…</>) : isActive ? (<><span className="spinner-border spinner-border-sm me-1" role="status" />Обработка…</>) : (<><i className="bi bi-upload me-1"></i>Загрузить и предобработать</>)}
          </button>
          {(!!isActive || phase === 'transfer') && (
            <button className="sv-btn sv-btn-red" onClick={onCancel} type="button"><i className="bi bi-x-circle me-1" aria-hidden />Отменить</button>
          )}
        </div>
      </div>

      {error && (
        <div className="sv-alert sv-alert-danger" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span><strong>Ошибка:</strong> {error}</span>
          <button type="button" onClick={() => setError(null)} style={{ background: 'none', border: 'none', color: 'var(--cream)', cursor: 'pointer', fontSize: 18 }} aria-label="Закрыть"><i className="bi bi-x-lg" aria-hidden /></button>
        </div>
      )}

      {showProgress && (
        <div className="sv-panel" style={{ borderColor: task?.status === 'failed' ? 'var(--red-btn)' : task?.status === 'completed' ? 'var(--green)' : 'var(--gold)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <strong style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              {phase === 'transfer' && (<><i className="bi bi-broadcast" aria-hidden />Передача файла</>)}
              {phase !== 'transfer' && task?.status === 'completed' && (<><i className="bi bi-check-circle-fill" style={{ color: 'var(--green)' }} aria-hidden />Готово</>)}
              {phase !== 'transfer' && task?.status === 'failed' && (<><i className="bi bi-exclamation-octagon-fill" style={{ color: 'var(--red-btn)' }} aria-hidden />Ошибка</>)}
              {phase !== 'transfer' && task?.status === 'cancelled' && (<><i className="bi bi-slash-circle" aria-hidden />Отменено</>)}
              {phase !== 'transfer' && task?.status && !['completed', 'failed', 'cancelled'].includes(task.status) && (<><i className="bi bi-gear-wide-connected" aria-hidden />Предобработка</>)}
            </strong>
            {task?.status && (
              <span className={`sv-badge sv-badge-${
                task.status === 'completed' ? 'success' :
                task.status === 'failed' ? 'red' :
                task.status === 'running' ? 'blue' : 'dim'
              }`}>
                {taskStatusLabel(task.status)}
              </span>
            )}
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 13, opacity: 0.8 }}>
            <span>{displayMessage}</span>
            <span>{displayProgress}%</span>
          </div>

          <div className="sv-progress-track" style={{ marginBottom: 16 }}>
            <div
              className={`sv-progress-bar ${statusBarClass(task)}`}
              style={{ width: `${displayProgress}%` }}
            >
              {displayProgress > 10 && `${displayProgress}%`}
            </div>
          </div>

          <div style={{ display: 'flex', gap: 24, marginBottom: 12, fontSize: 13 }}>
            <span style={{ color: displayProgress >= 50 ? 'var(--green)' : phase === 'transfer' ? 'var(--gold)' : 'var(--cream)', opacity: phase === 'transfer' ? 1 : 0.5, display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              {displayProgress >= 50 ? <i className="bi bi-check2" aria-hidden /> : <i className="bi bi-circle" aria-hidden />}
              Передача файла (0–50%)
            </span>
            <span style={{ color: task?.status === 'completed' ? 'var(--green)' : phase === 'processing' && isActive ? 'var(--gold)' : 'var(--cream)', opacity: phase === 'processing' ? 1 : 0.5, display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              {task?.status === 'completed' ? <i className="bi bi-check2" aria-hidden /> : phase === 'processing' && isActive ? <i className="bi bi-record-fill" aria-hidden /> : <i className="bi bi-circle" aria-hidden />}
              Серверная предобработка (50–100%)
            </span>
          </div>

          {(task?.logs?.length ?? 0) > 0 && (
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

      {datasets.length > 0 ? (
        <div className="sv-panel">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
            <span className="sv-panel-title" style={{ marginBottom: 0 }}>Датасеты ({datasets.length})</span>
            <button
              className="sv-btn sv-btn-ghost"
              style={{ fontSize: 13, padding: '6px 14px' }}
              onClick={loadDatasets}>
              <i className="bi bi-arrow-clockwise me-1"></i>
              Обновить
            </button>
          </div>
          <div className="sv-table-wrap">
            <table className="sv-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Название</th>
                  <th>Строк</th>
                  <th>Метки</th>
                  <th>Создан</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {datasets.map(d => (
                  <tr key={d.id}>
                    <td><code style={{ color: 'var(--gold)' }}>{d.id}</code></td>
                    <td>{d.name}</td>
                    <td>{d.row_count ? d.row_count.toLocaleString() : '—'}</td>
                    <td>
                      {d.has_label
                        ? <span className="sv-badge sv-badge-success">Да</span>
                        : <span className="sv-badge sv-badge-dim">Нет</span>}
                    </td>
                    <td style={{ fontSize: 13, opacity: 0.75 }}>{new Date(d.created_at).toLocaleString()}</td>
                    <td>
                      <button
                        className="sv-btn sv-btn-red"
                        style={{ fontSize: 13, padding: '5px 14px' }}
                        onClick={() => onDelete(d.id, d.name)}
                        disabled={deleting === d.id}>
                        {deleting === d.id ? (
                          <span className="spinner-border spinner-border-sm me-1" role="status" />
                        ) : (
                          <i className="bi bi-trash me-1"></i>
                        )}
                        Удалить
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        !showProgress && (
          <div className="sv-panel sv-empty">
            Нет датасетов. Загрузите датасет, чтобы начать.
          </div>
        )
      )}
    </div>
  );
}
