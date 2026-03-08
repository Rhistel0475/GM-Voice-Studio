"""
Thin entrypoint: run the FastAPI app from app.main.
Usage: python server.py  (or: python -m app.main with uvicorn in __main__)
"""
import os as _os
try:
    from dotenv import load_dotenv
    load_dotenv(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

from app.main import app
from app.core.config import SERVER_NAME, PORT
from app.api.routers.health import health, ready

__all__ = ["app", "health", "ready"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_NAME, port=PORT)
