import React, { useState } from 'react';

export default function ExportModal({ open, onClose, onConfirm }) {
  const [json, setJson] = useState(true);
  const [pdf, setPdf] = useState(false);

  if (!open) return null;

  const handleConfirm = () => {
    const formats = [];
    if (json) formats.push('json');
    if (pdf) formats.push('pdf');
    if (formats.length === 0) return;
    onConfirm(formats);
    onClose();
  };

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 9999 }}>
      <div style={{ background: '#0b0b0f', padding: 18, borderRadius: 8, width: 420, boxShadow: '0 6px 18px rgba(0,0,0,0.6)' }}>
        <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 10 }}>Экспорт результатов</div>
        <div style={{ fontSize: 13, opacity: 0.9, marginBottom: 12 }}>Выберите форматы экспорта:</div>
        <div style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <input type="checkbox" checked={json} onChange={e => setJson(e.target.checked)} /> JSON
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <input type="checkbox" checked={pdf} onChange={e => setPdf(e.target.checked)} /> PDF
          </label>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button className="sv-btn sv-btn-ghost" onClick={onClose}>Отмена</button>
          <button className="sv-btn sv-btn-green" onClick={handleConfirm} disabled={!json && !pdf}>Скачать</button>
        </div>
      </div>
    </div>
  );
}
