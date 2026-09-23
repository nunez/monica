"""Local HTTP service. Sole inference endpoint: POST /v1/systemone.

No other inference endpoints exist (GET returns 405, which doubles as a
liveness signal). Networking is never required at inference: all weights and
calibration files load from local disk.
"""

from __future__ import annotations

import json
import os
import threading

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .audioio import AudioInputError
from .model import MonicaModel, MonicaConfig
from .schemas import JevRequest, MAX_QUESTIONS

MODEL_DIR = os.environ.get("MONICA_BASE_MODEL", "data/models/gemma-4-E2B-it")
CKPT = os.environ.get("MONICA_HEAD_CKPT", "models/head-cremad.pt")
CALIB = os.environ.get("MONICA_CALIB", "models/calibration.json")
DEVICE = os.environ.get("MONICA_DEVICE", "mps")
MAX_BODY = int(os.environ.get("MONICA_MAX_BODY_BYTES", str(64 * 1024 * 1024)))


def create_app(model: MonicaModel | None = None) -> FastAPI:
    app = FastAPI(title="monica", docs_url=None, redoc_url=None, openapi_url=None)
    holder: dict = {"model": model, "lock": threading.Lock()}

    @app.on_event("startup")
    def _startup():
        if holder["model"] is None:
            cfg = MonicaConfig(model_dir=MODEL_DIR, ckpt=CKPT, device=DEVICE,
                            calibration_path=CALIB if os.path.exists(CALIB) else None)
            holder["model"] = MonicaModel(cfg)
            holder["model"].warmup()
        app.state.model = holder["model"]

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        body = await request.body()
        if len(body) > MAX_BODY:
            return _err(413, "request_too_large",
                       f"request body exceeds {MAX_BODY} bytes")
        try:
            payload = json.loads(body)
        except Exception:
            return _err(422, "invalid_json", "request body is not valid JSON")
        if not isinstance(payload, dict):
            return _err(422, "invalid_request", "request body must be a JSON object")
        state = payload.get("state")
        if not isinstance(state, dict) or not state.get("audio"):
            return _err(400, "empty_audio", "state.audio is required (base64 WAV data URL or file path)")
        try:
            req = JevRequest(**payload)
        except Exception as e:  # pydantic
            return _err(422, "invalid_request", str(e))
        if len(req.questions) > MAX_QUESTIONS:
            return _err(422, "too_many_questions", f"max {MAX_QUESTIONS} questions per request")
        m = holder["model"]
        with holder["lock"]:
            try:
                resp, _q = m.handle(req.to_plain())
            except AudioInputError as e:
                status = 415 if e.code == "unsupported_media_type" else 400
                return _err(status, e.code, e.message)
        return JSONResponse(resp)

    return app


def _err(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def main():
    import uvicorn

    uvicorn.run(create_app(), host=os.environ.get("MONICA_HOST", "127.0.0.1"),
                port=int(os.environ.get("MONICA_PORT", "8910")), log_level="info")


if __name__ == "__main__":
    main()


app = create_app()
