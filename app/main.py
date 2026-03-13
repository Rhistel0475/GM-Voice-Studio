"""
FastAPI application factory and entry point.
Load .env, create app, register middleware, routers, static mounts, and startup.
"""
import logging
import os
import time
import uuid
from pathlib import Path

from fastapi import Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api.dependencies.auth import limiter
from app.api.routers import config as config_router_module
from app.api.routers import health as health_router_module
from app.core.config import CORS_ORIGINS
from app.core.settings import get_settings
from app.core.logging import configure_logging
from app.core.metrics import record_request_duration
from app.infrastructure.database import init_db

# Load .env first
_REPO_ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

# Patch transformers before any kani_tts import (kani_tts.model needs TransformersKwargs/Unpack)
def _patch_transformers_utils():
    try:
        from typing import TypedDict, Unpack
        import transformers.utils as tf_utils
        import transformers.processing_utils as tf_proc
        if not hasattr(tf_utils, "TransformersKwargs"):
            tf_utils.TransformersKwargs = TypedDict("TransformersKwargs", {}, total=False)
        if not hasattr(tf_proc, "Unpack"):
            tf_proc.Unpack = Unpack
    except Exception as e:
        logging.warning("Transformers compat patch skipped: %s", e)


_patch_transformers_utils()

# Ensure LFM2 is available and patch Lfm2Config for kani_tts (expects config.rope_theta; transformers 5 uses rope_parameters)
def _check_lfm2_available_and_patch_config():
    try:
        from transformers.models.lfm2.configuration_lfm2 import Lfm2Config
        # kani_tts accesses config.rope_theta; transformers 5.x Lfm2Config has rope_parameters dict instead
        if not hasattr(Lfm2Config, "rope_theta"):
            @property
            def _rope_theta_prop(self):
                rp = getattr(self, "rope_parameters", None)
                if rp is not None and isinstance(rp, dict):
                    return rp.get("rope_theta", 10000.0)
                if rp is not None and hasattr(rp, "rope_theta"):
                    return rp.rope_theta
                return 10000.0

            Lfm2Config.rope_theta = _rope_theta_prop
    except ImportError:
        logging.warning(
            "transformers.models.lfm2 not found. TTS/play will fail. "
            "Upgrade: pip install -U 'transformers>=5.1.0'"
        )


_check_lfm2_available_and_patch_config()

# kani_tts uses _tied_weights_keys = ["lm_head.weight"] (list); transformers 5.x get_expanded_tied_weights_keys expects a dict
def _patch_tied_weights_keys_list():
    try:
        from transformers.modeling_utils import PreTrainedModel
        _orig = PreTrainedModel.get_expanded_tied_weights_keys

        def _patched(self, all_submodels=False):
            keys = getattr(self, "_tied_weights_keys", None)
            if isinstance(keys, list):
                self._tied_weights_keys = {k: "model.embed_tokens.weight" for k in keys}
            try:
                return _orig(self, all_submodels)
            finally:
                if isinstance(keys, list):
                    self._tied_weights_keys = keys

        PreTrainedModel.get_expanded_tied_weights_keys = _patched
    except Exception as e:
        logging.warning("Tied weights list patch skipped: %s", e)


_patch_tied_weights_keys_list()

# Import FastAPI and slowapi after env is loaded
from fastapi import FastAPI, HTTPException
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware


def create_app() -> FastAPI:
    app = FastAPI(title="Kani TTS API")
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    if CORS_ORIGINS:
        origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
        if origins:
            from fastapi.middleware.cors import CORSMiddleware
            app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=False,
                allow_methods=["*"],
                allow_headers=["*"],
            )

    @app.middleware("http")
    async def request_logging_and_metrics(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        path = request.scope.get("path") or request.url.path
        record_request_duration(path, duration)
        extra = {
            "request_id": request_id,
            "request_path": path,
            "status_code": response.status_code,
            "duration_seconds": round(duration, 4),
        }
        if getattr(request.state, "voice_id", None) is not None:
            extra["voice_id"] = request.state.voice_id
        if getattr(request.state, "job_id", None) is not None:
            extra["job_id"] = request.state.job_id
        logging.getLogger("request").info("request_finished", extra=extra)
        response.headers["X-Request-ID"] = request_id
        return response

    assets_dir = _REPO_ROOT / "static" / "campaign_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/campaign-assets", StaticFiles(directory=str(assets_dir)), name="campaign_assets")

    img_dir = _REPO_ROOT / "static" / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static/img", StaticFiles(directory=str(img_dir)), name="static_img")

    @app.on_event("startup")
    def startup():
        init_db()
        configure_logging()
        settings = get_settings()
        logging.info("Kani TTS API starting; models load on first request.")
        if not settings.hf_token:
            logging.warning("HF_TOKEN is not set. Voice cloning may fail.")
        else:
            logging.info("HF_TOKEN is set; voice cloning should be available.")

    app.include_router(health_router_module.router, tags=["health"])
    app.include_router(config_router_module.router, tags=["config"])

    # All remaining routes (voices, clone, tts, campaigns, adventure, rag, brain, npc, ai, ws, pages)
    from app.api.routers import routes_legacy
    app.include_router(routes_legacy.router, tags=["legacy"])

    # HTML and favicon
    static_index = _REPO_ROOT / "static" / "index.html"
    static_preview_index = _REPO_ROOT / "static" / "index.preview.html"
    static_preview_react_dir = _REPO_ROOT / "static" / "frontend"
    static_preview_react_index = static_preview_react_dir / "index.html"

    def _with_no_cache(resp: Response) -> Response:
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    def _serve_preview(subpath=None):
        if static_preview_react_index.is_file():
            if subpath:
                base = static_preview_react_dir.resolve()
                candidate = (base / subpath).resolve()
                try:
                    candidate.relative_to(base)
                except ValueError:
                    raise HTTPException(404, "Not found")
                if candidate.is_file():
                    return FileResponse(candidate)
            return _with_no_cache(FileResponse(static_preview_react_index, media_type="text/html"))
        return _with_no_cache(FileResponse(static_preview_index, media_type="text/html"))

    @app.get("/favicon.ico")
    def favicon():
        return Response(status_code=204)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return FileResponse(static_index, media_type="text/html")

    @app.get("/preview", response_class=HTMLResponse)
    def preview_index():
        return _serve_preview()

    @app.get("/preview/{subpath:path}", response_class=HTMLResponse)
    def preview_subpath(subpath: str):
        return _serve_preview(subpath)

    return app


app = create_app()
