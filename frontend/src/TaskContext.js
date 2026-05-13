import React, { createContext, useContext, useState, useRef, useCallback } from 'react';
import api from './api';

const TaskContext = createContext(null);

export function TaskProvider({ children }) {
  const [tasks, setTasks] = useState({});
  const sources = useRef({});
  const [epochData, setEpochData] = useState({});
  const [datasetsVersion, setDatasetsVersion] = useState(0);

  const updateTask = useCallback((taskId, patch) => {
    setTasks(prev => ({
      ...prev,
      [taskId]: { ...(prev[taskId] || {}), ...patch },
    }));
  }, []);

  const appendLog = useCallback((taskId, message) => {
    const timestamp = new Date().toLocaleTimeString();
    setTasks(prev => ({
      ...prev,
      [taskId]: {
        ...(prev[taskId] || {}),
        logs: [...((prev[taskId] || {}).logs || []), `[${timestamp}] ${message}`],
      },
    }));
  }, []);

  const appendEpoch = useCallback((taskId, epochRow) => {
    setEpochData(prev => {
      const existing = prev[taskId] || [];
      const idx = existing.findIndex(r => r.epoch === epochRow.epoch);
      if (idx >= 0) {
        const copy = [...existing];
        copy[idx] = epochRow;
        return { ...prev, [taskId]: copy };
      }
      return { ...prev, [taskId]: [...existing, epochRow] };
    });
  }, []);

  const startStream = useCallback((taskId, { onUpdate, onComplete } = {}) => {
    if (sources.current[taskId]) {
      sources.current[taskId].close();
    }

    const source = api.streamTask(taskId, (data) => {
      updateTask(taskId, data);
      if (data.message) appendLog(taskId, data.message);

      if (data.epoch !== undefined && data.train_loss !== undefined) {
        appendEpoch(taskId, {
          epoch: data.epoch,
          epochs: data.epochs,
          train_loss: data.train_loss,
          val_loss: data.val_loss,
          val_acc: data.val_acc,
          val_auc: data.val_auc,
          val_f1: data.val_f1,
          val_precision: data.val_precision,
          val_recall: data.val_recall,
        });
      }

      onUpdate?.(data);

      if (['completed', 'failed', 'cancelled'].includes(data.status)) {
        source.close();
        delete sources.current[taskId];
        onComplete?.(data);
      }
    });

    sources.current[taskId] = source;
    return source;
  }, [updateTask, appendLog, appendEpoch]);

  const stopStream = useCallback((taskId) => {
    if (sources.current[taskId]) {
      sources.current[taskId].close();
      delete sources.current[taskId];
    }
  }, []);

  const isStreaming = useCallback(
    (taskId) => !!sources.current[taskId],
    []
  );

  const clearTask = useCallback((taskId) => {
    stopStream(taskId);
    setTasks(prev => {
      const copy = { ...prev };
      delete copy[taskId];
      return copy;
    });
    setEpochData(prev => {
      const copy = { ...prev };
      delete copy[taskId];
      return copy;
    });
  }, [stopStream]);

  const notifyDatasetsChanged = useCallback(() => {
    setDatasetsVersion(prev => prev + 1);
  }, []);

  const activeCount = Object.values(tasks).filter(
    t => t && ['pending', 'queued', 'running'].includes(t.status)
  ).length;

  return (
    <TaskContext.Provider value={{
      tasks,
      epochData,
      startStream,
      stopStream,
      clearTask,
      updateTask,
      appendLog,
      isStreaming,
      activeCount,
      datasetsVersion,
      notifyDatasetsChanged,
    }}>
      {children}
    </TaskContext.Provider>
  );
}

export const useTaskContext = () => {
  const ctx = useContext(TaskContext);
  if (!ctx) throw new Error('useTaskContext must be used inside TaskProvider');
  return ctx;
};
