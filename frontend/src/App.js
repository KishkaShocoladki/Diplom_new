import React, { useState } from 'react';
import { TaskProvider, useTaskContext } from './TaskContext';
import { taskStatusLabel, taskKindLabel } from './uiLabels';
import UploadPage from './pages/UploadPage';
import TrainingPage from './pages/TrainingPage';
import PredictPage from './pages/PredictPage';
import ExperimentsPage from './pages/ExperimentsPage';
import DatasetInfoPage from './pages/DatasetInfoPage';
import './App.css';

const sections = [
  { key: 'upload',      label: 'Загрузка' },
  { key: 'train',       label: 'Обучение' },
  { key: 'predict',     label: 'Предсказание' },
  { key: 'experiments', label: 'Эксперименты' },
  { key: 'dataset',     label: 'Структурные варианты' },
];

function GlobalTaskBadge() {
  const { tasks, activeCount } = useTaskContext();
  if (activeCount === 0) return null;

  const active = Object.values(tasks).filter(
    t => t && ['pending', 'queued', 'running'].includes(t.status)
  );

  return (
    <div className="sv-global-badge">
      <span className="sv-spinner" />
      <span>
        <strong style={{ color: 'var(--green)' }}>
          {activeCount} фоновых задач{activeCount > 1 ? '' : 'а'} выполняется
        </strong>
        {active.map(t => (
          <span key={t.task_id || t.id} style={{ marginLeft: 16, opacity: 0.7, fontSize: 12 }}>
            {t.kind ? `${taskKindLabel(t.kind)}: ` : ''}{t.message || taskStatusLabel(t.status)} — {t.progress ?? 0}%
          </span>
        ))}
      </span>
    </div>
  );
}

function AppShell() {
  const [active, setActive] = useState('upload');

  return (
    <div>
      <GlobalTaskBadge />

      <div className="app-header">
        <h1 className="app-title">
          Модуль предсказания влияния структурных вариантов на экспрессию генов
        </h1>
        <p className="app-subtitle">Классификация SV и прогноз экспрессии генов</p>
      </div>

      <nav className="sv-nav">
        {sections.map(sec => (
          <button
            key={sec.key}
            className={`sv-nav-btn${active === sec.key ? ' active' : ''}`}
            onClick={() => setActive(sec.key)}
          >
            {sec.label}
          </button>
        ))}
      </nav>

      <div className="sv-page">
        <div style={{ display: active === 'upload'      ? 'block' : 'none' }}><UploadPage /></div>
        <div style={{ display: active === 'train'       ? 'block' : 'none' }}><TrainingPage /></div>
        <div style={{ display: active === 'predict'     ? 'block' : 'none' }}><PredictPage /></div>
        <div style={{ display: active === 'experiments' ? 'block' : 'none' }}><ExperimentsPage /></div>
        <div style={{ display: active === 'dataset'     ? 'block' : 'none' }}><DatasetInfoPage /></div>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <TaskProvider>
      <AppShell />
    </TaskProvider>
  );
}
