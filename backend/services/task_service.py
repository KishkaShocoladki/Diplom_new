from typing import Optional
import json
import time
from datetime import datetime

TASKS: dict = {}
TASK_DATASET_MAP: dict = {}
CANCEL_FLAGS: dict = {}


def create_task(initial_message: str = "started") -> str:
    import uuid

    task_id = str(uuid.uuid4())
    task_data = {
        "id": task_id,
        "status": "pending",
        "progress": 0,
        "message": initial_message,
        "updated_at": datetime.utcnow().isoformat(),
    }
    TASKS[task_id] = task_data
    return task_id


def update_task(task_id: str, status: Optional[str] = None, progress=None, message=None, **extra):
    if task_id not in TASKS:
        return
    if status is not None:
        TASKS[task_id]["status"] = status
    if progress is not None:
        TASKS[task_id]["progress"] = progress
    if message is not None:
        TASKS[task_id]["message"] = message
    for k, v in extra.items():
        TASKS[task_id][k] = v
    TASKS[task_id]["updated_at"] = datetime.utcnow().isoformat()


def get_task(task_id: str):
    return TASKS.get(task_id)


def task_stream(task_id: str):
    def generate():
        last = None
        while True:
            info = TASKS.get(task_id)
            if info is None:
                break
            data = json.dumps(info)
            if data != last:
                yield f"data: {data}\n\n"
                last = data
            if info.get("status") in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.1)

    return generate()


def cancel_task(task_id: str):
    CANCEL_FLAGS[task_id] = True
    if task_id in TASKS:
        update_task(task_id, status="cancelled", message="Cancellation requested")
