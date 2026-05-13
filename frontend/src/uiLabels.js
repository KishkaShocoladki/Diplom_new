export function taskStatusLabel(status) {
  const m = {
    pending: 'Ожидание',
    queued: 'В очереди',
    running: 'Выполняется',
    completed: 'Завершено',
    failed: 'Ошибка',
    cancelled: 'Отменено',
  };
  return m[status] ?? status ?? '';
}

export function experimentKindLabel(kind) {
  const m = {
    train: 'Обучение',
    fine_tune: 'Дообучение',
    retrain: 'Переобучение',
    predict: 'Предсказание',
  };
  return m[kind] ?? kind ?? '';
}

export function taskKindLabel(kind) {
  const m = {
    upload: 'Загрузка',
    train: 'Обучение',
    fine_tune: 'Дообучение',
    retrain: 'Переобучение',
    predict: 'Предсказание',
  };
  return m[kind] ?? kind ?? '';
}
