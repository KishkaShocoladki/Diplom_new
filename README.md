# Веб-приложение: анализ влияния структурных вариантов на экспрессию генов

Flask API + React UI: загрузка датасетов, предобработка, обучение/дообучение/переобучение модели, предсказание и просмотр экспериментов.

## Требования

- Python 3.10+ (рекомендуется 3.12)
- Node.js 18+ и npm
- PostgreSQL (по умолчанию строка подключения ниже)

## Установка

### Бэкенд

Из корня репозитория:

```bash
python -m venv venv
```
или
```bash
# Windows: venv\Scripts\activate
```
```bash
pip install -r requirements.txt
```

Далее необходимо задать БД DATABASE_URL (иначе используется значение по умолчанию из кода):
```text
postgresql://postgres:postgres@localhost:5432/diplom
```

Папки для файлов (при отсутствии создаются частично автоматически; пути можно переопределить):

| Переменная | Назначение |
|------------|------------|
| DATABASE_URL | Строка подключения SQLAlchemy |
| DATASETS_DIR | Корень датасетов (raw / preprocessed) |
| MODELS_DIR | Сохраненные веса моделей |
| ARTIFACTS_DIR | Артефакты предобработки |
| PREDICTIONS_DIR | Выгрузка предсказаний |

### Фронтенд

```bash
cd frontend
npm install
```

При другом адресе API (не http://localhost:5000/api) создайте frontend/.env:

```text
REACT_APP_API_BASE=http://localhost:5000/api
```

## Запуск

1. API (из корня репозитория - иначе не разрешится пакет backend):

   ```bash
   python -m backend.app
   ```

   API слушает http://0.0.0.0:5000 (в т.ч. http://localhost:5000).

2. Интерфейс:

   ```bash
   cd frontend
   npm start
   ```

   Откроется http://localhost:3000.

## Демонстрационный пример

1. Запуск приложения:
   - Запуск бэкенда: ![enter image description here](https://i.ibb.co/pr0kyVGk/st1.png)
   - Запуск фронтенда: ![enter image description here](https://i.ibb.co/3YFzytVb/st2.png)

2. Первая страница веб-приложения - загрузка датасетов. Выберите датасет, дайте ему название (при желании). В папке проекта есть тестовые датасеты как для обучения (с лейблами), так и для предсказания.
   ![загрузка датасетов](https://i.ibb.co/FkmyVfJ8/load.png)
   
   Файлы в проекте:
   ![Файлы в проекте](https://i.ibb.co/qM4jHx3Y/files.png)
   
   Присутствуют как крупные датасеты для обучения, так и небольшие для предсказания.

3. После загрузки датасета переходите в обучение.
   ![обучение](https://i.ibb.co/ZR40dVmZ/train.png)
   
   Здесь выберите датасет, укажите название модели. Остальные поля заполнены по умолчанию, но их можно редактировать. Для начала обучения кликните на зеленую кнопку.
   ![enter image description here](https://i.ibb.co/C36WSMyd/train-res.png)
   
   Во время обучения метрики выводятся в реальном времени на графиках. После завершения обучения выведется таблица метрик по эпохам.

Помимо обучения с нуля доступны:

- Дообучение. Здесь необходимо выбрать соответствующую опцию и модель, которую нужно дообучить.
  ![enter image description here](https://i.ibb.co/wNYv88my/finetun.png)
  
  Вывод метрик в целом остается прежним, но доступна таблица сравнения метрик с прошлой версией:
  ![enter image description here](https://i.ibb.co/Q3XfWNXP/finetunres.png)

- Переобучение. Также выбирается соответствующая опция и модель, которая будет переобучаться.
  ![enter image description here](https://i.ibb.co/PZHGQ5gw/retraint.png)
  
  По итогу доступна таблица сравнения метрик с прошлой версией:
  ![enter image description here](https://i.ibb.co/bgcHyZcC/retrainres.png)

- После того, как модель обучена, переходите к предсказанию.
  ![enter image description here](https://i.ibb.co/wZVwScJ2/pred.png)
  
  Выберите обученную модель, выберите датасет. Нажмите кнопку предсказать.
  
  По завершении предсказания будут выведены результаты:
  - Логи предсказания и метрики
    ![enter image description here](https://i.ibb.co/v4FJTbD5/predres1.png)
  
  - Распределение результатов и биологическая интерпретация. В данном случае для каждого СВ выводятся затронутые им гены и то, с какой вероятностью данный СВ повлияет на ген патогенно или нейтрально.
    ![enter image description here](https://i.ibb.co/zW04k87W/predres2.png)
  
  - Таблица предсказаний с объяснением результатов (можно посмотреть как краткое объяснение, так и развернутое).
    ![enter image description here](https://i.ibb.co/tpd2W5ss/predres3.png)
  
  - Доступен экспорт (файлы экспорта как пример находятся в папке проекта).

- Все обучения или предсказания считаются экспериментами и отображаются на соответствующей вкладке со всеми прошедшими экспериментами:
  ![enter image description here](https://i.ibb.co/PdkGRpR/exp.png)
  
  - Результаты предсказания: ![enter image description here](https://i.ibb.co/VpVj08Fw/exppred1.png)
  - Результаты обучения: ![enter image description here](https://i.ibb.co/0j555G0h/exptrain.png)
  - Результаты дообучения/переобучения: ![enter image description here](https://i.ibb.co/0V2C50kP/expretrain.png) ![enter image description here](https://i.ibb.co/pB2tHJFX/2expretrain.png)

- Последняя страница веб-приложения - страница анализа загруженных датасетов:
  ![enter image description here](https://i.ibb.co/VcC0ZgPL/sv1.png)
  
  - Во вкладке "Статистика" отображается общая статистика по датасету:
    ![enter image description here](https://i.ibb.co/7dSbx8tb/sv2.png)
    ![enter image description here](https://i.ibb.co/XrLzJk4s/sv3.png)
    ![enter image description here](https://i.ibb.co/rKRySQPX/sv4.png)
  
  - Во вкладке "Данные" отображаются все СВ в датасете с возможностью их фильтрации:
    ![enter image description here](https://i.ibb.co/BHpJJF40/sv5.png)
    ![enter image description here](https://i.ibb.co/ccNZgPmc/sv6.png)