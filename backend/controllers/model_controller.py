import logging
from pathlib import Path

from flask import Blueprint, jsonify

from backend.extensions import db
from backend.models import Experiment, Model

logger = logging.getLogger(__name__)

bp = Blueprint("models", __name__, url_prefix="/api/models")


@bp.route("/", methods=["GET"])
def list_models_controller():
    models = Model.query.order_by(Model.created_at.desc()).all()
    return jsonify([
        {
            "id": m.id,
            "name": m.name,
            "input_dim": m.input_dim,
            "dropout": m.dropout,
            "path": m.path,
            "created_at": m.created_at.isoformat(),
            "updated_at": m.updated_at.isoformat(),
        }
        for m in models
    ])


@bp.route("/<int:model_id>", methods=["DELETE"])
def delete_model_controller(model_id):
    """Удаляет запись модели и файл весов; у всех экспериментов model_id сбрасывается в NULL."""
    model = db.session.get(Model, model_id)
    if not model:
        return jsonify({"error": "Модель не найдена"}), 404

    try:
        n = Experiment.query.filter_by(model_id=model_id).update(
            {"model_id": None},
            synchronize_session=False,
        )

        if model.path:
            try:
                p = Path(model.path)
                if p.is_file():
                    p.unlink()
            except OSError as exc:
                logger.warning("Не удалось удалить файл модели %s: %s", model.path, exc)

        db.session.delete(model)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        logger.exception("Ошибка удаления модели %s", model_id)
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "message": "Модель удалена",
        "model_id": model_id,
        "experiments_unlinked": n,
    }), 200


def register_blueprint(app):
    app.register_blueprint(bp)
