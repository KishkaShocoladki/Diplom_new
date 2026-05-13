import api from '../api';

const MIME = {
  pdf:  'application/pdf',
  csv:  'text/csv',
  xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
};

const EXT = { pdf: 'pdf', csv: 'csv', xlsx: 'xlsx' };

export async function downloadExperimentExports(experimentId, formats, onError) {
  for (const fmt of formats) {
    try {
      const res = await api.exportExperiment(experimentId, [fmt]);

      const blobData = res.data instanceof Blob
        ? res.data
        : new Blob([res.data], { type: MIME[fmt] || 'application/octet-stream' });

      const url = URL.createObjectURL(blobData);
      const a = document.createElement('a');
      a.href = url;
      a.download = `experiment_${experimentId}.${EXT[fmt] || fmt}`;
      document.body.appendChild(a);
      a.click();
      a.remove();

      setTimeout(() => URL.revokeObjectURL(url), 10_000);

      if (formats.indexOf(fmt) < formats.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 600));
      }
    } catch (err) {
      console.error('Export error for format', fmt, err);
      const msg = err?.response?.data?.error || err?.message || 'Неизвестная ошибка';
      if (onError) onError(`Ошибка экспорта ${fmt.toUpperCase()}: ${msg}`);
    }
  }
}

export async function exportAll(experimentId, kind, onError) {
  const formats = kind === 'predict' ? ['pdf', 'csv', 'xlsx'] : ['pdf'];
  await downloadExperimentExports(experimentId, formats, onError);
}
