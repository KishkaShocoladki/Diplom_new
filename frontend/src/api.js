import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:5000/api';

const api = {
  uploadDataset: (formData, onUploadProgress) =>
    axios.post(`${API_BASE}/datasets/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: onUploadProgress
        ? (e) => {
            if (e.total) {
              onUploadProgress(Math.round((e.loaded * 100) / e.total));
            }
          }
        : undefined,
    }),

  listDatasets: () => axios.get(`${API_BASE}/datasets`),
  getDataset: (datasetId) => axios.get(`${API_BASE}/datasets/${datasetId}`),
  getDatasetSvInfo: (datasetId, params = {}) => axios.get(`${API_BASE}/datasets/${datasetId}/sv-info`, { params }),
  deleteDataset: (datasetId) => axios.delete(`${API_BASE}/datasets/${datasetId}`),

  startExperiment: (payload) => axios.post(`${API_BASE}/experiments/start`, payload),
  startPrediction: (payload) => axios.post(`${API_BASE}/experiments/predict`, payload),
  listExperiments: () => axios.get(`${API_BASE}/experiments`),
  getExperiment: (id) => axios.get(`${API_BASE}/experiments/${id}`),
  getExperimentGraphs: (id) => axios.get(`${API_BASE}/experiments/${id}/graphs`),
  getExperimentPredictions: (id, page = 1, per_page = 100) =>
    axios.get(`${API_BASE}/experiments/${id}/predictions`, { params: { page, per_page } }),
  deleteExperiment: (id) => axios.delete(`${API_BASE}/experiments/${id}`),
  exportExperiment: (id, formats = ['csv','json','pdf']) => axios.post(`${API_BASE}/experiments/${id}/export`, { formats }, { responseType: 'blob' }),
  getModelTrainingHistory: (modelId) => axios.get(`${API_BASE}/experiments/model/${modelId}/history`),

  listModels: () => axios.get(`${API_BASE}/models`),
  deleteModel: (modelId) => axios.delete(`${API_BASE}/models/${modelId}`),

  getTask: (taskId) => axios.get(`${API_BASE}/tasks/${taskId}`),
  cancelTask: (taskId) => axios.post(`${API_BASE}/tasks/${taskId}/cancel`),

  streamTask: (taskId, onMessage) => {
    const evtSource = new EventSource(`${API_BASE}/tasks/${taskId}/stream`);

    evtSource.onmessage = (event) => {
      try {
        onMessage(JSON.parse(event.data));
      } catch (e) {
        console.error('SSE parse error:', e, event.data);
      }
    };

    evtSource.onerror = (err) => {
      console.error('SSE error for task', taskId, err);
      evtSource.close();
    };

    return evtSource;
  },
};

export default api;
