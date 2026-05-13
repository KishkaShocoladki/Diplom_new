from flask import Blueprint, jsonify, Response
from backend.services import task_service

bp = Blueprint("tasks", __name__, url_prefix="/api/tasks")


@bp.route("/<task_id>", methods=["GET"])
def get_task_controller(task_id):
    entry = task_service.get_task(task_id)
    if not entry:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(entry)


@bp.route("/<task_id>/stream", methods=["GET"])
def task_stream_controller(task_id):
    return Response(task_service.task_stream(task_id), mimetype="text/event-stream")


@bp.route("/<task_id>/cancel", methods=["POST"])
def cancel_task_controller(task_id):
    entry = task_service.get_task(task_id)
    if not entry:
        return jsonify({"error": "Task not found"}), 404
    task_service.cancel_task(task_id)
    return jsonify({"status": "cancelled"})


def register_blueprint(app):
    app.register_blueprint(bp)
